/**
* This file is part of VIRALC.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/britsknguyen/VIRALC>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* VIRALC is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* VIRALC is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with VIRALC.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 15/12/20.
//

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include "../utility.h"

class Point2PlaneDisFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    Point2PlaneDisFactor() = delete;
    Point2PlaneDisFactor(const Eigen::Vector3d &point,
                         const Eigen::Vector4d &coeff) : point_(point), sqrt_info_static_(1.0)
    {
        normal_ << coeff.x(), coeff.y(), coeff.z();
        offset_ =  coeff.w();
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        double residual = normal_.transpose() * (Qi * point_ + Pi) + offset_;

        residuals[0] = sqrt_info_static_*residual;
        
        if(jacobians)
        {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            jacobian_pose.setZero();
            jacobian_pose.block<1, 3>(0, 0) =  normal_.transpose();
            jacobian_pose.block<1, 3>(0, 3) = -normal_.transpose()
                                               *Qi.toRotationMatrix()
                                               *Util::skewSymmetric(point_);
            jacobian_pose = sqrt_info_static_*jacobian_pose;
        }

        return true;
    }
    // void Check(double **parameters);

private:
    Eigen::Vector3d point_;
    Eigen::Vector3d normal_;
    double          offset_;
    
    double sqrt_info_static_;
};