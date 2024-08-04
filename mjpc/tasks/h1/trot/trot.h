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

#ifndef MJPC_TASKS_H1_TROT_TASK_H_
#define MJPC_TASKS_H1_TROT_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace h1 {

class Trot : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Trot* task) : mjpc::BaseResidualFn(task) {}

    // ------------------ Residuals for humanoid walk task ------------
    //   Number of residuals:
    //     Residual (0): torso height
    //     Residual (1): pelvis-feet aligment
    //     Residual (2): balance
    //     Residual (3): upright
    //     Residual (4): posture
    //     Residual (5): walk
    //     Residual (6): move feet
    //     Residual (7): control
    //   Number of parameters:
    //     Parameter (0): torso height goal
    //     Parameter (1): speed goal
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    double get_param(int i) const { return parameters_[i]; }
    static double StepHeight(double time, double footphase, double duty_ratio);
    double GetPhase(double time) const;
    
    double last_transition_time_ = -1;
    double phase_start_       = 0;
    double phase_start_time_  = 0;
    double phase_velocity_    = 0;
  };

  Trot() : residual_(this) {}
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;
  void TransitionLocked(mjModel* model, mjData* data) override;
  void ResetLocked(const mjModel* model) override;

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

#endif  // MJPC_TASKS_H1_WALK_TASK_H_
