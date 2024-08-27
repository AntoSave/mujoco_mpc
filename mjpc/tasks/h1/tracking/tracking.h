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

#ifndef MJPC_TASKS_H1_TRACKING_TASK_H_
#define MJPC_TASKS_H1_TRACKING_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/spline/spline.h"

#include "ndcurves/polynomial.h"

namespace mjpc {
namespace h1 {

class Tracking : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Tracking* task);

    // ------------------ Residuals for humanoid tracking task ------------
    //   Number of residuals:
    //     Residual (0): Joint vel: minimise joint velocity
    //     Residual (1): Control: minimise control
    //     Residual (2-27): Tracking position: minimise tracking joint position error.
    //     Residual (28-52): Tracking velocity: minimise tracking joint velocity error.
    //   Number of parameters: 0
    // --------------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    
    mjtNum ref_time = 0.0;
    mjtNum ref_qpos[26] = {0.0};
    mjtNum ref_qvel[25] = {0.0};
    spline::TimeSpline ref_spline_qpos = spline::TimeSpline(26);
    ndcurves::polynomial_t poly;
  };

  Tracking() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace h1
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_TRACKING_TASK_H_
