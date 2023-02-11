/**
* This file is part of SLICT.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/brytsknguyen/slict>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* SLICT is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* SLICT is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with SLICT.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility.h"
#include "PreintBase.h"

#include <ceres/ceres.h>

// using namespace Util;

class PreintFactor : public ceres::SizedCostFunction<15, 7, 3, 6, 7, 3, 6>
{
  public:
    PreintFactor() = delete;
    PreintFactor(PreintBase* _pre_integration) : pre_integration(_pre_integration), GRAV(_pre_integration->GRAV){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d    Vi(parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d   Bai(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Vector3d   Bgi(parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d    Pj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Quaterniond Qj(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

        Eigen::Vector3d    Vj(parameters[4][0], parameters[4][1], parameters[4][2]);

        Eigen::Vector3d   Baj(parameters[5][0], parameters[5][1], parameters[5][2]);
        Eigen::Vector3d   Bgj(parameters[5][3], parameters[5][4], parameters[5][5]);


#if 0
        if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
            (Bgi - pre_integration->linearized_bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }
#endif

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                             Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix<double, 15, 15>
            sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                            (pre_integration->covariance.inverse()).matrixL().transpose();

        // printf("IMU residual: \n");
        // std::cout << residual << std::endl;
        // printf("IMU pre_integration->covariance: \n");
        // std::cout << pre_integration->covariance << std::endl;
        // printf("IMU pre_integration->covariance.inverse(): \n");
        // std::cout << pre_integration->covariance.inverse() << std::endl;
        // printf("IMU sqrt_info: \n");
        // std::cout << sqrt_info << std::endl;

        // sqrt_info.setIdentity();
        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;

            Eigen::Matrix3d dp_dba = pre_integration->ddp_dba;
            Eigen::Matrix3d dp_dbg = pre_integration->ddp_dbg;

            Eigen::Matrix3d dv_dba = pre_integration->ddv_dba;
            Eigen::Matrix3d dv_dbg = pre_integration->ddv_dbg;

            Eigen::Matrix3d dq_dbg = pre_integration->ddq_dbg;

            Vector3d jacobian_extrema(0, 0, 0);
            pre_integration->jacobian_MaxAndMin(jacobian_extrema);
            if (jacobian_extrema(0) > 1e8 || jacobian_extrema(2) < -1e8)
            {
                // ROS_WARN("numerical unstable in preintegration");
                //std::cout << pre_integration->jacobian << std::endl;
                // ROS_BREAK();
            }

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();


                // Jacobians from preintegrated position (alpha)-----------------------------------------------

                jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(O_P, O_R) =  Util::skewSymmetric( Qi.inverse() 
                                                                              * ( 0.5 * GRAV * sum_dt * sum_dt
                                                                                      + Pj - Pi
                                                                                      - Vi * sum_dt
                                                                                )
                                                                            );


                // Jacobians from preintegrated velocity (beta)------------------------------------------------

                jacobian_pose_i.block<3, 3>(O_V, O_R) = Util::skewSymmetric(Qi.inverse() * (GRAV * sum_dt + Vj - Vi));


                // Jacobians from preintegrated quaternion (gamma)---------------------------------------------

                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q
                                                        * Util::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
                jacobian_pose_i.block<3, 3>(O_R, O_R) = -( Util::Qright(corrected_delta_q)
                                                           * Util::Qleft(Qj.inverse() * Qi)
                                                         ).bottomRightCorner<3, 3>();


                // Applying the weightage----------------------------------------------------------------------

                jacobian_pose_i = sqrt_info * jacobian_pose_i;


                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    // ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_velocity_i(jacobians[1]);
                jacobian_velocity_i.setZero();

                // Jacobians from preintegrated position (alpha)-----------------------------------------------

                jacobian_velocity_i.block<3, 3>(O_P, 0) = -Qi.inverse().toRotationMatrix() * sum_dt;


                // Jacobians from preintegrated velocity (beta)------------------------------------------------

                jacobian_velocity_i.block<3, 3>(O_V, 0)  = -Qi.inverse().toRotationMatrix();


                // Jacobians from preintegrated quaternion (gamma)---------------------------------------------
                // ... none

                
                // Jacobians from bias residuals --------------------------------------------------------------
                // ... none


                // Applying the weightage----------------------------------------------------------------------

                jacobian_velocity_i = sqrt_info * jacobian_velocity_i;

                //ROS_ASSERT(fabs(jacobian_velocity_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_velocity_i.minCoeff()) < 1e8);
            }

            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_bias_i(jacobians[2]);
                jacobian_bias_i.setZero();

                // Jacobians from preintegrated position (alpha)-----------------------------------------------

                jacobian_bias_i.block<3, 3>(O_P, 0) = -dp_dba;
                jacobian_bias_i.block<3, 3>(O_P, 3) = -dp_dbg;


                // Jacobians from preintegrated velocity (beta)------------------------------------------------

                jacobian_bias_i.block<3, 3>(O_V, 0) = -dv_dba;
                jacobian_bias_i.block<3, 3>(O_V, 3) = -dv_dbg;


                // Jacobians from preintegrated quaternion (gamma)---------------------------------------------

                jacobian_bias_i.block<3, 3>(O_R, 3) = -Util::Qright( pre_integration->delta_q.inverse()
                                                                     * Qi.inverse() * Qj
                                                                   ).bottomRightCorner<3, 3>()
                                                                   * dq_dbg;

                
                // Jacobians from bias residuals --------------------------------------------------------------

                jacobian_bias_i.block<3, 3>(O_BA, 0) = -Eigen::Matrix3d::Identity();
                jacobian_bias_i.block<3, 3>(O_BG, 3) = -Eigen::Matrix3d::Identity();


                // Applying the weightage----------------------------------------------------------------------

                jacobian_bias_i = sqrt_info * jacobian_bias_i;

                //ROS_ASSERT(fabs(jacobian_bias_i.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_bias_i.minCoeff()) < 1e8);
            }

            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[3]);
                jacobian_pose_j.setZero();


                // Jacobians from preintegrated position (alpha)-----------------------------------------------

                jacobian_pose_j.block<3, 3>(O_P, O_P) =  Qi.inverse().toRotationMatrix();

                
                // Jacobians from preintegrated velocity (beta)------------------------------------------------
                // ... none

                
                // Jacobians from preintegrated quaternion (gamma)---------------------------------------------

                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q
                                                       *
                                                       Util::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));

                jacobian_pose_j.block<3, 3>(O_R, O_R) = Util::Qleft( corrected_delta_q.inverse()
                                                                     * Qi.inverse() * Qj
                                                                   ).bottomRightCorner<3, 3>();


                // Jacobians from bias residuals --------------------------------------------------------------
                // ... none


                // Applying the weightage----------------------------------------------------------------------

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }

            if (jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_velocity_j(jacobians[4]);
                jacobian_velocity_j.setZero();

                // Jacobians from preintegrated position (alpha)-----------------------------------------------
                // ... none


                // Jacobians from preintegrated velocity (beta)------------------------------------------------

                jacobian_velocity_j.block<3, 3>(O_V, 0) = Qi.inverse().toRotationMatrix();


                // Jacobians from preintegrated quaternion (gamma)---------------------------------------------
                // ... none


                // Jacobians from bias residuals --------------------------------------------------------------
                // ... none


                // Applying the weightage----------------------------------------------------------------------

                jacobian_velocity_j = sqrt_info * jacobian_velocity_j;

                //ROS_ASSERT(fabs(jacobian_velocity_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_velocity_j.minCoeff()) < 1e8);
            }

            if (jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jacobian_biases_j(jacobians[5]);
                jacobian_biases_j.setZero();

                // Jacobians from preintegrated position (alpha)-----------------------------------------------
                // ... none


                // Jacobians from preintegrated velocity (beta)------------------------------------------------
                // ... none

                // Jacobians from preintegrated quaternion (gamma)---------------------------------------------
                // ... none


                // Jacobians from bias residuals --------------------------------------------------------------

                jacobian_biases_j.block<3, 3>(O_BA, 0) = Eigen::Matrix3d::Identity();
                jacobian_biases_j.block<3, 3>(O_BG, 3) = Eigen::Matrix3d::Identity();


                // Applying the weightage----------------------------------------------------------------------

                jacobian_biases_j = sqrt_info * jacobian_biases_j;

                //ROS_ASSERT(fabs(jacobian_biases_j.maxCoeff()) < 1e8);
                //ROS_ASSERT(fabs(jacobian_biases_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    PreintBase* pre_integration;
    Vector3d GRAV;

};

