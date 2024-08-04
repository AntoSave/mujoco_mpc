// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/h1/trot/trot.h"

#include <iostream>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc::h1 {
constexpr float rgba_red[4] = {1.0, 0.0, 0.0, 1};
constexpr float rgba_blue[4] = {0.0, 0.0, 1.0, 1};
constexpr double marker_size = 0.025;

std::string Trot::XmlPath() const {
  return GetModelPath("h1/trot/task.xml");
}
std::string Trot::Name() const { return "H1 Trot"; }

// ------------------ Residuals for humanoid trot task ------------
//   Number of residuals:
//     Residual (0): Upright
//     Residual (1): Height
//     Residual (2): Position
//     Residual (3): Gait
//     Residual (4): Balance
//     Residual (5): Control
//     Residual (6): Posture up
//   Number of parameters:
//     Parameter (0): Cadence
//     Parameter (1): Amplitude
//     Parameter (2): Duty ratio
//     Parameter (3): Walk speed
//     Parameter (4): Walk turn
//     Parameter (5): Bipedal height
// ----------------------------------------------------------------
void Trot::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  int counter = 0;
  double height_goal = parameters_[5];
  double* foot_right = SensorByName(model, data, "foot_right");
  double* foot_left = SensorByName(model, data, "foot_left");
  double* foot_right_xbody = SensorByName(model, data, "foot_right_xbody");
  double* foot_left_xbody = SensorByName(model, data, "foot_left_xbody");
  double avg_foot_pos[3];
  mju_add3(avg_foot_pos, foot_right, foot_left);
  mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);

  // ---------- Upright ----------
  double *torso_up = SensorByName(model, data, "torso_up");
  residual[counter++] = torso_up[2] - 1;

  // ---------- Height ----------
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = (torso_height - avg_foot_pos[2]) - height_goal;

  //Set next 3 residuals to zero
  for (int i = 0; i < 3; i++) {
    residual[counter++] = 0;
  }

  // ---------- Gait ----------
  double phases[2] = {0, 0.5};
  double foot_radius = 0.066;
  double step[2];
  double amplitude = parameters_[1];
  double duty_ratio = parameters_[2];
  double feet_height[2] = {foot_left_xbody[2], foot_right_xbody[2]};

  for (int i = 0; i < 2; i++) {
    double footphase = 2*mjPI*phases[i];
    step[i] = amplitude * StepHeight(GetPhase(data->time), footphase, duty_ratio);
    double ground_height = 0;//Ground(model, data, query);
    double height_target = ground_height + foot_radius + step[i];
    double height_difference = feet_height[i] - height_target;
    residual[counter++] = height_difference;
  }

  // ---------- Balance ----------
  double *compos = SensorByName(model, data, "torso_subcom");
  double *comvel = SensorByName(model, data, "torso_subcomvel");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- posture up ----- //
  mju_copy(&residual[counter], data->qpos + 17, model->nq - 17); //First 7 are freejoint coord, the other 10 are lower body joints
  counter += model->nq - 17;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension"
        "and actual length of residual %d",
        counter);
  }
}

double Trot::ResidualFn::StepHeight(double time, double footphase,
                                             double duty_ratio) {
  double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

double Trot::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

void Trot::TransitionLocked(mjModel* model, mjData* data) {
  if (data->time < residual_.last_transition_time_ || residual_.last_transition_time_ == -1) {
    residual_.last_transition_time_ = residual_.phase_start_time_ = data->time;
    residual_.phase_start_ = 0;
  }

  double phase_velocity = 2 * mjPI * parameters[0];
  if (phase_velocity != residual_.phase_velocity_) {
    residual_.phase_start_ = residual_.GetPhase(data->time);
    residual_.phase_start_time_ = data->time;
    residual_.phase_velocity_ = phase_velocity;
  }
}

void Trot::ResetLocked(const mjModel* model) {
  std::cout << "Resetting Trot" << std::endl;
}

void Trot::ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const {
  double *foot_left = SensorByName(model, data, "foot_left_xbody");
  double *foot_right = SensorByName(model, data, "foot_right_xbody");
  double foot_left_target[3], foot_right_target[3];
  double phases[2] = {0, 0.5};
  double foot_radius = 0.066;
  double step[2];
  double amplitude = parameters[1]; //parameters_[1];
  double duty_ratio = parameters[2]; //parameters_[2];
  double height_targets[2];
  for (int i = 0; i < 2; i++) {
    double footphase = 2*mjPI*phases[i];
    step[i] = amplitude * Trot::ResidualFn::StepHeight(residual_.GetPhase(data->time), footphase, duty_ratio);
    double ground_height = 0;//Ground(model, data, query);
    height_targets[i] = ground_height + foot_radius + step[i];
  }
  mju_copy3(foot_left_target, foot_left);
  mju_copy3(foot_right_target, foot_right);
  foot_left_target[2] = height_targets[0];
  foot_right_target[2] = height_targets[1];
  AddGeom(scene, mjGEOM_SPHERE, &marker_size, foot_left_target, /*mat=*/nullptr, rgba_red);
  AddGeom(scene, mjGEOM_SPHERE, &marker_size, foot_right_target, /*mat=*/nullptr, rgba_red);
  AddGeom(scene, mjGEOM_SPHERE, &marker_size, foot_left, /*mat=*/nullptr, rgba_blue);
  AddGeom(scene, mjGEOM_SPHERE, &marker_size, foot_right, /*mat=*/nullptr, rgba_blue);

}

}  // namespace mjpc::h1
