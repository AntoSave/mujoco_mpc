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

#include "mjpc/tasks/h1/walk-obs/walk.h"

#include <iostream>
#include <string>
#include <Eigen/Geometry>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include <mujoco/mujoco.h>

namespace mjpc::h1 {
std::string WalkObs::XmlPath() const { return GetModelPath("h1/walk-obs/task.xml"); }
std::string WalkObs::Name() const { return "H1 WalkObs"; }

float normal_pdf(float x, float m, float s) {
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;

    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

// ------------------ Residuals for humanoid walkobs task ------------
//   Number of residuals:
//     Residual (0): torso height
//     Residual (1): pelvis-feet aligment
//     Residual (2): balance
//     Residual (3): upright
//     Residual (4): torso posture
//     Residual (5): arms posture
//     Residual (6): face towards goal
//     Residual (7): walk towards goal
//     Residual (8): move feet
//     Residual (9): control
//     Residual (10): feet distance
//     Residual (11): leg cross
//     Residual (12): slippage
//     Residual (13): obstacles
//   Number of parameters:
//     Parameter (0): torso height goal
//     Parameter (1): speed goal
//     Parameter (2): feet distance goal
//     Parameter (3): balance speed
// ----------------------------------------------------------------
void WalkObs::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                double *residual) const {
  int counter = 0;

  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters_[0];

  // ----- pelvis / feet ----- //
  double *foot_right = SensorByName(model, data, "foot_right");
  double *foot_left = SensorByName(model, data, "foot_left");
  double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] =
      0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // ----- balance ----- //
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, parameters_[3], 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment
  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right, foot_left);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right, foot_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing = torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- upright ----- //
  double *torso_up = SensorByName(model, data, "torso_up");
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  double *foot_right_up = SensorByName(model, data, "foot_right_up");
  double *foot_left_up = SensorByName(model, data, "foot_left_up");
  double z_ref[3] = {0.0, 0.0, 1.0};

  // torso
  residual[counter++] = torso_up[2] - 1.0;

  // pelvis
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

  // right foot
  mju_sub3(&residual[counter], foot_right_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  mju_sub3(&residual[counter], foot_left_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;
  // ----- torso & arms posture ----- //
  // We ignore the 7 dof of the freejoint and the 10 dof of the lower body joints
  mju_copy(&residual[counter], data->qpos + 17, model->nq - 17); 
                            
  counter += model->nq - 17;

  // ----- walk ----- //
  double robot_to_goal[2];
  double *goal_point = SensorByName(model, data, "goal");
  double *torso_position = SensorByName(model, data, "torso_position");
  mju_sub(robot_to_goal, goal_point, torso_position, 2);
  double goal_distance = mju_normalize(robot_to_goal, 2);
  residual[counter++] = goal_distance;

  // ----- speed ----- //
  double com_vel[2];
  mju_copy(com_vel, subcomvel, 2); // subcomvel is the velocity of the robot's CoM
  residual[counter++] = mju_normalize(com_vel, 2) - parameters_[1];

  // ----- speed orientation ----- //
  double *pelvis_forward = SensorByName(model, data, "pelvis_forward");
  mju_normalize(pelvis_forward, 2);
  residual[counter++] = 90.0*(mju_dot(pelvis_forward, com_vel, 2) - 1.0);

  // ----- move feet ----- //
  double move_feet[2];
  
  double *foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  double *foot_left_vel = SensorByName(model, data, "foot_left_velocity");

  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- obstacles ----- //
  double *pelvis_pos = SensorByName(model, data, "pelvis_position");
  Eigen::Vector2d pos = Eigen::Vector2d(pelvis_pos[0], pelvis_pos[1]);
  double cost = 0.0;
  //double dist = 0.0;
  int i = 0;
  double gaussian_x, gaussian_y;
  for(i = 0; i < this->obstacles.size(); i++) {
    // dist = std::min((pos - this->obstacles[i]).norm() - 1.0, 0.0);
    // cost += std::exp(10.8*(0.3-dist));
    gaussian_x = normal_pdf(pos[0], this->obstacles[i][0], 0.4);
    gaussian_y = normal_pdf(pos[1], this->obstacles[i][1], 0.4);
    cost += 30 * gaussian_x * gaussian_y;
  }
  residual[counter++] = cost;


  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension"
                "and actual length of residual %d",
                counter);
  }
}

} // namespace mjpc::h1
