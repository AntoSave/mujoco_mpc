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
  residual[counter++] = 0.0;
  residual[counter++] = 0.0;

  for (int i = 0; i < model->nq; i++) {
    residual[counter++] = data->qpos[i];
  }

  for (int i = 0; i < model->nv; i++) {
    residual[counter++] = data->qvel[i];
  }

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

}  // namespace mjpc::h1
