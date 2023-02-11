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

#pragma once

#include <ceres/ceres.h>
#include "utility.h"

using namespace Eigen;

const int O_P  = 0;
const int O_R  = 3;
const int O_V  = 6;
const int O_BA = 9;
const int O_BG = 12;

class PreintBase
{
protected:

  public:

    ~PreintBase(){};
    PreintBase(const Eigen::Vector3d &_acc_0,
               const Eigen::Vector3d &_gyr_0,
               const Eigen::Vector3d &_linearized_ba,
               const Eigen::Vector3d &_linearized_bg,
               bool _show_init_cost,
               double _ACC_N, double _ACC_W,
               double _GYR_N, double _GYR_W,
               Vector3d _GRAV, int _id)
        : acc_0{_acc_0}, gyr_0{_gyr_0}, acc_0_{_acc_0}, gyr_0_{_gyr_0},
          linearized_acc{_acc_0},
          linearized_gyr{_gyr_0},
          linearized_ba{_linearized_ba},
          linearized_bg{_linearized_bg},
          show_init_cost{_show_init_cost},
          ACC_N{_ACC_N},
          ACC_W{_ACC_W},
          GYR_N{_GYR_N},
          GYR_W{_GYR_W},
          GRAV{_GRAV},
          // jacobian{Eigen::Matrix<double, 15, 15>::Identity()},
          covariance{Eigen::Matrix<double, 15, 15>::Zero()},
          sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()},
          delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()},
          id(_id)

    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0)   =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3)   =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6)   =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9)   =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();

        // printf("preint %d noise:\n", id);
        // std::cout << noise << std::endl;

        ddp_dba.setZero();
        ddp_dbg.setZero();

        ddv_dba.setZero();
        ddv_dbg.setZero();

        ddq_dbg.setZero();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        
        ddp_dba.setZero();
        ddp_dbg.setZero();
        ddv_dba.setZero();
        ddv_dbg.setZero();
        ddq_dbg.setZero();

        covariance.setZero();

        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        forwardPropagate(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        // delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;  
    }

    void forwardPropagate(double _dt, 
                            const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                            const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        //ROS_INFO("midpoint integration");
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        Quaterniond q_1_step_zoh = Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        q_1_step_zoh.normalize();
        result_delta_q = delta_q * q_1_step_zoh;
        // result_delta_q.normalize();
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;           

        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_x = 0.5 * (_acc_0 + _acc_1) - linearized_ba;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_x, R_a_0_x, R_a_1_x;

            R_w_x   << 0,      -w_x(2),   w_x(1),
                       w_x(2),  0,       -w_x(0),
                      -w_x(1),  w_x(0),   0;

            R_a_x   << 0,      -a_x(2),   a_x(1),
                       a_x(2),  0,       -a_x(0),
                      -a_x(1),  a_x(0),   0;

            R_a_0_x << 0,         -a_0_x(2),  a_0_x(1),
                       a_0_x(2),   0,        -a_0_x(0),
                      -a_0_x(1),   a_0_x(0),  0;

            R_a_1_x << 0,         -a_1_x(2),  a_1_x(1),
                       a_1_x(2),   0,        -a_1_x(0),
                      -a_1_x(1),   a_1_x(0),  0;

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0)   = Matrix3d::Identity();
            F.block<3, 3>(0, 3)   = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                    -0.25 * result_delta_q.toRotationMatrix()
                                          * R_a_1_x
                                          * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6)   = MatrixXd::Identity(3, 3) * _dt;
            F.block<3, 3>(0, 9)   = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12)  = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3)   = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12)  = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3)   = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt + 
                                    -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6)   = Matrix3d::Identity();
            F.block<3, 3>(6, 9)   = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12)  = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9)   = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0)   =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3)   =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6)   =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9)   =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3)   =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9)   =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0)   =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3)   =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6)   =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9)   =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12)  = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;

            covariance = F * covariance * F.transpose() + V * noise * V.transpose();

            ddp_dbg = ddp_dbg + ddv_dbg*dt - 0.5*delta_q.toRotationMatrix()*R_a_x*ddq_dbg*_dt*_dt;
            ddp_dba = ddp_dba + ddv_dba*dt - 0.5*delta_q.toRotationMatrix()*_dt*_dt;

            ddv_dbg = ddv_dbg - delta_q*R_a_x*ddq_dbg*_dt;
            ddv_dba = ddv_dba - delta_q.toRotationMatrix()*_dt;

            Matrix3d H = Matrix3d::Identity() - 0.5*R_w_x*_dt + R_w_x*R_w_x/6; //Approximation to avoid using cos and sin
                                                                               //which can lead to division by zero
            ddq_dbg = q_1_step_zoh.inverse()*ddq_dbg - H*_dt;
        }
    }

    void getPredictedDeltas(Eigen::Vector3d &dp, Eigen::Vector3d &dv, Eigen::Quaterniond &dq,
                            const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                            const Eigen::Vector3d Bai, const Eigen::Vector3d Bgi)
    {
        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        dq = (delta_q * Util::deltaQ(ddq_dbg * dbg));
        dq.normalize();
        dv = Qi*(delta_v + ddv_dba * dba + ddv_dbg * dbg - Qi.inverse()*GRAV*sum_dt);
        dp = Qi*(delta_p + ddp_dba * dba + ddp_dbg * dbg + Qi.inverse()*Vi*sum_dt - Qi.inverse()*GRAV*sum_dt*sum_dt/2);
    }

    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Util::deltaQ(ddq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + ddv_dba * dba + ddv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + ddp_dba * dba + ddp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * GRAV * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (GRAV * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        // printf("IMU residual: \n");
        // std::cout << residuals << std::endl;
        // printf("IMU Jacobian: \n");
        // std::cout << jacobian << std::endl;

        if(show_init_cost)
        {
            show_init_cost = false;
            Eigen::Matrix<double, 15, 15>
                sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>
                                (covariance.inverse()).matrixL().transpose();
            
            auto w_cost = sqrt_info*residuals;
            double cost = 0.5*w_cost.transpose()*w_cost;

            printf("preint %d. cost %f. buff: %d. ", id, cost, acc_buf.size());
            printf("Dp: %6.3f, %6.3f, %6.3f. "
                   "Dv: %6.3f, %6.3f, %6.3f. "
                   "Dq: %6.3f, %6.3f, %6.3f, %6.3f. "
                   "Ba: %6.3f, %6.3f, %6.3f. "
                   "Bg: %6.3f, %6.3f, %6.3f\n",
                    corrected_delta_p.x(), corrected_delta_p.y(), corrected_delta_p.z(),
                    corrected_delta_v.x(), corrected_delta_v.z(), corrected_delta_v.z(),
                    corrected_delta_q.x(), corrected_delta_q.y(), corrected_delta_q.z(), corrected_delta_q.w(),
                    linearized_ba(0), linearized_ba(1), linearized_ba(2),
                    linearized_bg(0), linearized_bg(1), linearized_bg(2));

            std::cout << "Pi:" << Pi.transpose()
                      << ", Vi: " << Vi.transpose()
                      << ", Pj:" << Pj.transpose()
                      << ", Vj: " << Vj.transpose() << std::endl;
            printf("residual: \n");
            std::cout << residuals.transpose() << std::endl << std::endl;

            // std::cout << covariance << std::endl;
        }

        return residuals;
    }

    void jacobian_MaxAndMin(Eigen::Vector3d &extrema)
    {
        Eigen::Matrix<double, 3, 15> jcb;

        jcb << ddp_dba, ddp_dbg, ddv_dba, ddv_dbg, ddq_dbg;

        extrema << jcb.maxCoeff(), 0, jcb.minCoeff();
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    // Saved for later reference
    Eigen::Vector3d acc_0_, gyr_0_;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d       linearized_ba, linearized_bg;

    Eigen::Matrix<double, 15, 15> covariance;
    Eigen::Matrix<double, 18, 18> noise;

    Eigen::Matrix<double, 3, 3> ddp_dba;
    Eigen::Matrix<double, 3, 3> ddp_dbg;
    
    Eigen::Matrix<double, 3, 3> ddv_dba;
    Eigen::Matrix<double, 3, 3> ddv_dbg;

    Eigen::Matrix<double, 3, 3> ddq_dbg;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

    int  id;
    bool show_init_cost = true;

    double ACC_N;
    double ACC_W;
    double GYR_N;
    double GYR_W;
    Vector3d GRAV;
};