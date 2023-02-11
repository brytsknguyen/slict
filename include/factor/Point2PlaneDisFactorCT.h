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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with SLICT.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#ifndef SLICT_PIVOTPOINTPLANEFACTORCT_H_
#define SLICT_PIVOTPOINTPLANEFACTORCT_H_

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include "../utility.h"

class Point2PlaneDisFactorCT : public ceres::SizedCostFunction<1, 7, 7>
{
public:
    Point2PlaneDisFactorCT() = delete;
    Point2PlaneDisFactorCT(const Eigen::Vector3d &f_, const Eigen::Vector4d &coef,
                           double s_, double dt, double w_ = 1.0)
    : f(f_), n(coef.head<3>()), m(coef.tail<1>()(0)), s(s_), sb(s_ - 1), w(w_)
    {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaternd Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Quaternd Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        
        Eigen::AngleAxis<double> Phi(Qi.inverse() * Qj);
        Eigen::AngleAxis<double> sxPhi(s * Phi.angle(), Phi.axis());
        Eigen::AngleAxis<double> sbxPhi(sb * Phi.angle(), Phi.axis());

        Matrix3d Rs(Qi*sxPhi);
        Vector3d Ps = (1-s)*Pi + s*Pj;

        residuals[0] = w*(n.transpose() * (Rs * f + Ps) + m);
        
        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<1, 3>(0, 0) = (1-s)*n.transpose();

                Eigen::Matrix<double, 1, 3> Dr_DRsxf    =  n.transpose();
                Eigen::Matrix<double, 3, 3> DRsxf_DRs   = -Rs * Util::skewSymmetric(f);
                Eigen::Matrix<double, 3, 3> DRs_DsbxPhi =  Util::SO3Jright(sbxPhi);
                Eigen::Matrix<double, 3, 3> DsbxPhi_DQi = -sb * Util::SO3JleftInv(Phi);

                jacobian_pose_i.block<1, 3>(0, 3) = Dr_DRsxf * DRsxf_DRs * DRs_DsbxPhi * DsbxPhi_DQi;

                // if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                // {
                //     printf("numerical unstable in factor jacobian pose i\n");
                //     std::cout << jacobian_pose_i << std::endl;
                //     // ROS_BREAK();
                // }
            }

            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<1, 3>(0, 0) = s*n.transpose();

                Eigen::Matrix<double, 1, 3> Dr_DRsxf    =  n.transpose();
                Eigen::Matrix<double, 3, 3> DRsxf_DRs   = -Rs * Util::skewSymmetric(f);
                Eigen::Matrix<double, 3, 3> DRs_DsxPhi  =  Util::SO3Jright(sxPhi);
                Eigen::Matrix<double, 3, 3> DsbxPhi_DQj =  s * Util::SO3JrightInv(Phi);

                jacobian_pose_j.block<1, 3>(0, 3) = Dr_DRsxf * DRsxf_DRs * DRs_DsxPhi * DsbxPhi_DQj;

                // if (jacobian_pose_j.maxCoeff() > 1e8 || jacobian_pose_j.minCoeff() < -1e8)
                // {
                //     printf("numerical unstable in uwb factor jacobian pose j\n");
                //     std::cout << jacobian_pose_j << std::endl;
                //     // ROS_BREAK();
                // }
            }
        }

        return true;
    }
    // void Check(double **parameters);

private:
    Eigen::Vector3d f;
    Eigen::Vector3d n;
    double m;
    double w;

    double s;
    double sb;
    double dt;

};

#endif //SLICT_PIVOTPOINTPLANEFACTORCT_H_
