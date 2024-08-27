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

#include "mjpc/tasks/h1/tracking/tracking.h"

#include <iostream>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/spline/spline.h"

#include "ndcurves/polynomial.h"

namespace mjpc::h1 {
std::string Tracking::XmlPath() const {
  return GetModelPath("h1/tracking/task.xml");
}
std::string Tracking::Name() const { return "H1 Tracking"; }

// ------------------ Residuals for humanoid tracking task ------------
  //   Number of residuals:
  //     Residual (0): Joint vel: minimize joint velocity
  //     Residual (1): Control: minimize control
  //     Residual (2-27): Tracking position: minimize tracking joint position error.
  //     Residual (28-52): Tracking velocity: minimize tracking joint velocity error.
  //   Number of parameters: 0
  // --------------------------------------------------------------------
void Tracking::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  int counter = 0;
  //std::cout << "Residual\tdata->time" << data->time << "ref_time" << ref_time << std::endl;
  //std::vector<double> curr_ref_qpos = ref_spline_qpos.Sample(data->time);
  ndcurves::pointX_t curr_ref_qpos;
  if(ref_time>0.0){
    std::cout << "Poly" << std::endl;
    curr_ref_qpos = poly(data->time);
  } else {
    curr_ref_qpos = Eigen::VectorXd::Zero(model->nq);
  }
  // Joint Position
  for (int i = 0; i < model->nq; i++) {
    residual[counter++] = curr_ref_qpos[i] - data->qpos[i];
  }

  // Joint Velocity
  for (int i = 0; i < model->nv; i++) {
    residual[counter++] = 0.0;
  }

  // Control
  for (int i = 0; i < 19; i++) {
    residual[counter++] = 0.0;
  }

  // std::cout << "nq: " << model->nq << " nv: " << model->nv << " na: " << model->na << std::endl;
  // std::cout << "counter: " << counter << std::endl;
  // std::cout << "nsensor: " << model->nsensor << std::endl;

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

Tracking::ResidualFn::ResidualFn(const Tracking* task) : mjpc::BaseResidualFn(task) {
  ref_spline_qpos.SetInterpolation(spline::SplineInterpolation::kCubicSpline);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Tracking::TransitionLocked(mjModel *model, mjData *d) {
  mjtNum ref_time = d->userdata[0];
  //std::cout << "TransitionLocked\td->time" << d->time << "\tref time" << ref_time << std::endl;
  if(ref_time > residual_.ref_time) {
    residual_.ref_time = ref_time;
    // residual_.ref_spline_qpos.Clear();
    // residual_.ref_spline_qpos.AddNode(d->time, absl::Span<const mjtNum>(d->qpos, model->nq));
    // residual_.ref_spline_qpos.AddNode(ref_time, absl::Span<const mjtNum>(d->userdata+1, model->nq));
    //std::cout << "Reference updated" << std::endl;
    Eigen::Map<Eigen::VectorXd> curr_qpos(d->qpos, model->nq);
    Eigen::Map<Eigen::VectorXd> ref_qpos(d->userdata+1, model->nq);
    std::flush(std::cout);
    ndcurves::t_pointX_t points;
    points.push_back(curr_qpos);
    points.push_back(ref_qpos);
    residual_.poly = ndcurves::polynomial_t(points.begin(),points.end(),d->time,ref_time);
  }
  //std::cout << "TransitionLocked\td->time" << d->time << "\tref time" << ref_time << std::endl;
  if(residual_.ref_time > 0.0){
    ndcurves::pointX_t curr_ref_qpos = residual_.poly(d->time);
    //std::vector<double> curr_ref_qpos = residual_.ref_spline_qpos.Sample(d->time)
    mju_copy(d->userdata+1+model->nq, curr_ref_qpos.data(), model->nq); // Copy the reference to userdata[:nq]
    //std::cout << "Position updated" << std::endl;
  }
}

}  // namespace mjpc::h1
