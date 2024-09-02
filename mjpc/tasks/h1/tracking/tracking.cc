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

#include "ndcurves/so3_linear.h"
#include "ndcurves/polynomial.h"

namespace mjpc::h1 {
std::string Tracking::XmlPath() const {
  return GetModelPath("h1/tracking/task.xml");
}
std::string Tracking::Name() const { return "H1 Tracking"; }

ndcurves::SE3Curve_t::point_t SampleSE3(const ndcurves::SE3Curve_t &curve, double time) {
  if(time < curve.T_min_){
    return curve(curve.T_min_);
  } else if(time > curve.T_max_) {
    return curve(curve.T_max_);
  } else {
    return curve(time);
  }
}

ndcurves::SE3Curve_t::point_derivate_t DerivateSE3(const ndcurves::SE3Curve_t &curve, double time) {
  if(time < curve.T_min_){
    return curve.derivate(curve.T_min_, 1);
  } else if(time > curve.T_max_) {
    return curve.derivate(curve.T_max_,1);
  } else {
    return curve.derivate(time, 1);
  }
}

ndcurves::pointX_t SamplePoly(const ndcurves::polynomial_t &poly, double time) {
  if(time < poly.T_min_){
    return poly(poly.T_min_);
  } else if(time > poly.T_max_) {
    return poly(poly.T_max_);
  } else {
    return poly(time);
  }
}

ndcurves::pointX_t DerivatePoly(const ndcurves::polynomial_t &poly, double time) {
  if(time < poly.T_min_){
    return poly.derivate(poly.T_min_,1);
  } else if(time > poly.T_max_) {
    return poly.derivate(poly.T_max_,1);
  } else {
    return poly.derivate(time,1);
  }
}

void Tracking::ResidualFn::GetCurrReference(const mjModel* model,
                      const mjData* data,
                      Eigen::Vector3d &curr_ref_transform_pos,
                      Eigen::Quaterniond &curr_ref_transform_rot,
                      Eigen::VectorXd &curr_ref_transform_d,
                      ndcurves::pointX_t &curr_ref_qpos,
                      ndcurves::pointX_t &curr_ref_qvel) const {
  ndcurves::SE3Curve_t::point_t curr_ref_transform;
  if(ref_time > 0.0 && data->time <= ref_time + 1e-4) {
    curr_ref_qpos = SamplePoly(poly, data->time);
    curr_ref_qvel = DerivatePoly(poly, data->time);
    curr_ref_transform = SampleSE3(se3_curve, data->time);
    curr_ref_transform_d = DerivateSE3(se3_curve, data->time);
    curr_ref_transform_pos = curr_ref_transform.translation();
    curr_ref_transform_rot = Eigen::Quaterniond(curr_ref_transform.linear());
  } else {
    //std::cout << "No reference time" << std::endl;
    curr_ref_qpos = Eigen::VectorXd(Eigen::Map<const Eigen::VectorXd>(data->qpos+7, model->nq-7));
    curr_ref_qvel = Eigen::VectorXd(Eigen::Map<const Eigen::VectorXd>(data->qvel+6, model->nv-6));
    curr_ref_transform_pos = Eigen::Vector3d(data->qpos[0],data->qpos[1],data->qpos[2]);
    curr_ref_transform_rot = Eigen::Quaterniond(data->qpos[3],data->qpos[4],data->qpos[5],data->qpos[6]);
    curr_ref_transform_d = Eigen::Map<const Eigen::VectorXd>(data->qvel, 6);
  }
}

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
  
  
  Eigen::Vector3d curr_ref_transform_pos;
  Eigen::Quaterniond curr_ref_transform_rot;
  Eigen::VectorXd curr_ref_transform_d;
  ndcurves::pointX_t curr_ref_qpos;
  ndcurves::pointX_t curr_ref_qvel;
  
  GetCurrReference(model, data, curr_ref_transform_pos, curr_ref_transform_rot, curr_ref_transform_d, curr_ref_qpos, curr_ref_qvel);
  
  //Eigen::Quaterniond rotation_error = curr_ref_transform_rot * Eigen::Quaterniond(data->qpos[3],data->qpos[4],data->qpos[5],data->qpos[6]).inverse();

  // Robot Position
  for (int i = 0; i < 3; i++) {
    residual[counter++] = curr_ref_transform_pos[i] - data->qpos[i];
  }
  // Robot Orientation
  for (int i = 0; i < 1; i++) {
    residual[counter++] = 0.0;//rotation_error.coeffs()[i];  //TODO: include orientation
  }
  // Robot Velocity
  for (int i = 0; i < 3; i++) { 
    residual[counter++] = curr_ref_transform_d[i] - data->qvel[i];
  }
  counter += 3; //TODO: include orientation
  // Joint Positions
  for (int i = 0; i < model->nq-7; i++) {
    residual[counter++] = curr_ref_qpos[i] - data->qpos[i+7];
  }
  // Joint Velocities
  for (int i = 0; i < model->nv-6; i++) {
    residual[counter++] = curr_ref_qvel[i] - data->qvel[i+6];
  }

  // Height
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters_[0];

  // ----- balance ----- //
  double *foot_right = SensorByName(model, data, "foot_right");
  double *foot_left = SensorByName(model, data, "foot_left");
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, parameters_[1], 3);
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
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

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
  //std::cout<<"ResidualFn INITIALIZATION"<<std::endl;
  start_transform = Eigen::Affine3d::Identity();
  ref_transform = Eigen::Affine3d::Identity();
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
    residual_.start_transform.translation() = Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2]);
    residual_.start_transform.linear() = Eigen::Quaterniond(d->qpos[3],d->qpos[4],d->qpos[5],d->qpos[6]).toRotationMatrix();
    residual_.ref_transform.translation() = Eigen::Vector3d(d->userdata[1],d->userdata[2],d->userdata[3]);
    residual_.ref_transform.linear() = Eigen::Quaterniond(d->userdata[4],d->userdata[5],d->userdata[6],d->userdata[7]).toRotationMatrix();
    Eigen::Map<Eigen::VectorXd> curr_qpos_map(d->qpos+7, model->nq-7);
    Eigen::Map<Eigen::VectorXd> final_qpos_map(d->userdata+1+7, model->nq-7);
    Eigen::Map<Eigen::VectorXd> curr_qvel_map(d->qvel+6, model->nv-6);
    Eigen::Map<Eigen::VectorXd> final_qvel_map(d->userdata+1+model->nq+6, model->nv-6);
    residual_.ref_qpos.clear();
    residual_.ref_qvel.clear();
    residual_.ref_qpos.push_back(Eigen::VectorXd(curr_qpos_map));
    residual_.ref_qpos.push_back(Eigen::VectorXd(final_qpos_map));
    residual_.ref_qvel.push_back(Eigen::VectorXd(curr_qvel_map));
    residual_.ref_qvel.push_back(Eigen::VectorXd(final_qvel_map));
    residual_.poly = ndcurves::polynomial_t(residual_.ref_qpos[0],residual_.ref_qvel[0],residual_.ref_qpos[1],residual_.ref_qvel[1],d->time,ref_time);
    residual_.se3_curve = ndcurves::SE3Curve_t(residual_.start_transform,residual_.ref_transform,d->time,ref_time);
  }
  if(residual_.ref_time > 0.0) {
    Eigen::Vector3d curr_ref_transform_pos;
    Eigen::Quaterniond curr_ref_transform_rot;
    Eigen::VectorXd curr_ref_transform_d;
    ndcurves::pointX_t curr_ref_qpos;
    ndcurves::pointX_t curr_ref_qvel;
  
    residual_.GetCurrReference(model, d, curr_ref_transform_pos, curr_ref_transform_rot, curr_ref_transform_d, curr_ref_qpos, curr_ref_qvel);
    // Copy the references to userdata
    auto userdata_ptr = d->userdata + 1 + model->nq + model->nv; // Skip the ref_time, qpos and qvel
    mju_copy(userdata_ptr, curr_ref_transform_pos.data(), 3);
    userdata_ptr += 3;
    Eigen::Vector4d curr_ref_transform_rot_coeffs = curr_ref_transform_rot.coeffs();
    // xyzw -> wxyz
    mju_copy(userdata_ptr, curr_ref_transform_rot_coeffs.data()+3, 1);
    userdata_ptr += 3;
    mju_copy(userdata_ptr, curr_ref_transform_rot_coeffs.data(), 3);
    userdata_ptr += 1;
    mju_copy(userdata_ptr, curr_ref_qpos.data(), model->nq-7);
    userdata_ptr += model->nq-7;
    mju_copy(userdata_ptr, curr_ref_transform_d.data(), 6);
    userdata_ptr += 6;
    mju_copy(userdata_ptr, curr_ref_qvel.data(), model->nv-6);
    //std::cout << "Position updated" << std::endl;
  }
}

}  // namespace mjpc::h1
