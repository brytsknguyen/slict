#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

class GyroAcceBiasFactorTMN
{
public:
    
    // Destructor
    ~GyroAcceBiasFactorTMN() {};

    GyroAcceBiasFactorTMN(const ImuSample &imu_sample_, const ImuBias &imu_bias_, const Vector3d &GRAV_,
                          double wGyro_, double wAcce_, double wBiasGyro_, double wBiasAcce_,
                          int N_, double Dt_, double s_)
    :   imu_sample    (imu_sample_    ),
        imu_bias      (imu_bias_      ),
        GRAV          (GRAV_          ),
        wGyro         (wGyro_         ),
        wAcce         (wAcce_         ),
        wBiasGyro     (wBiasGyro_     ),
        wBiasAcce     (wBiasAcce_     ),
        N             (N_             ),
        Dt            (Dt_            ),
        s             (s_             )
    {
        // Copy the parameter blocks

        // Calculate the spline-defining quantities

        // Blending matrix for position, in standard form
        Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Blending matrix for rotation, in cummulative form
        Matrix<double, Dynamic, Dynamic> Btilde = basalt::computeBlendingMatrix<double, true>(N);

        // Inverse of knot length
        double Dt_inv = 1.0 / Dt;

        // Time powers
        Matrix<double, Dynamic, 1> U(N);
        for (int j = 0; j < N; j++)
            U(j) = std::pow(s, j);

        // Time power derivative
        Matrix<double, Dynamic, 1> Udot = Matrix<double, Dynamic, 1>::Zero(N);
        for (int j = 1; j < N; j++)
            Udot(j) = j * std::pow(s, j - 1);

        // Time power derivative
        Matrix<double, Dynamic, 1> Uddot = Matrix<double, Dynamic, 1>::Zero(N);
        for (int j = 2; j < N; j++)
            Uddot(j) = j * (j - 1) * std::pow(s, j - 2);

        // Lambda
        lambda_R = Btilde * U;

        // Lambda dot
        lambda_R_dot = Dt_inv * Btilde * Udot;

        // Lambda a
        lambda_P_ddot = Dt_inv * Dt_inv * B * Uddot;

        // Resize the residual and jacobian block;
        residual = Matrix<double, 12, 1>::Zero();
        jacobian = Matrix<double, 12, -1>::Zero(12, N*6 + 6);
    }

    Matrix3d rightJacobian(const Vector3d &phi) const
    {
        Matrix3d Jr;
        Sophus::rightJacobianSO3(phi, Jr);
        return Jr;
    }

    Matrix3d rightJacobianInv(const Vector3d &phi) const
    {
        Matrix3d Jr_inv;
        Sophus::rightJacobianInvSO3(phi, Jr_inv);
        return Jr_inv;
    }

    bool Evaluate(vector<SO3d> &Rot, vector<Vector3d> &Pos, Vector3d &biasW, Vector3d &biasA, bool computeJacobian = true)
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // // Indexing offsets for the states
        // size_t R_offset = 0;              // for quaternion
        // size_t P_offset = R_offset + N*3; // for position
        // size_t B_offset = P_offset + N*3; // for bias

        // // Map parameters to the control point states
        // SO3d Rot[N];
        // Vector3d Pos[N];
        // for (int j = 0; j < N; j++)
        // {
        //     Rot[j] = Eigen::Map<SO3d const>(PARAM_ROT[j]);
        //     Pos[j] = Eigen::Map<Vector3d const>(PARAM_POS[j]);
        // }

        // // Map parameters to the bias
        // Vector3d biasW = Eigen::Map<Vector3d const>(PARAM_BIG);
        // Vector3d biasA = Eigen::Map<Vector3d const>(PARAM_BIA);


        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        // The following use the formulations in the paper 2020 CVPR:
        // "Efficient derivative computation for cumulative b-splines on lie groups."
        // Sommer, Christiane, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers.
        // Some errors in the paper is corrected

        // Calculate the delta terms: delta(1) ... delta(N-1), where delta(j) = Log( R(j-1)^-1 * R(j) ),
        // delta(0) is an extension in the paper as the index starts at 1
        Vector3d delta[N];
        delta[0] = Rot[0].log();
        for (int j = 1; j < N; j++)
        {
            delta[j] = (Rot[j - 1].inverse() * Rot[j]).log();
            // if (isnan(delta[j].norm()))
            // {
            //     printf(KRED "delta[%d] is nan!" RESET, j);
            //     delta[j] = Vector3d(0, 0, 0);
            // }
        }

        // Calculate the A terms: A(1) ... A(N-1), where A(j) = Log( lambda(j) * d(j) ), A(0) is an extension
        SO3d A[N];
        A[0] = Rot[0];
        for (int j = 1; j < N; j++)
            A[j] = SO3d::exp(lambda_R(j) * delta[j]).matrix();

        // Calculate the P terms: P(0) ... P(N-1) = I, where P(j) = A(N-1)^-1 A(N-2)^-1 ... A(j+1)^-1
        SO3d P[N];
        P[N - 1] = SO3d(Quaternd(1, 0, 0, 0));
        for (int j = N - 1; j >= 1; j--)
            P[j - 1] = P[j] * A[j].inverse();

        // Calculate the omega terms: omega(1) ... omega(N), using equation (38), omega(N) is the angular velocity
        Vector3d omega[N + 1];
        omega[0] = Vector3d(0, 0, 0);
        omega[1] = Vector3d(0, 0, 0);
        for (int j = 1; j < N + 1; j++)
            omega[j] = A[j - 1].inverse() * omega[j - 1] + lambda_R_dot(j - 1) * delta[j - 1];

        // Predicted orientation from Rt^-1 = P(N-1)R(0)^-1
        SO3d R_W_Bt = (P[0] * Rot[0].inverse()).inverse();

        // // Predicted position
        // Vector3d p_inW_Bt(0, 0, 0);
        // for (int j = 0; j < N; j++)
        //     p_inW_Bt += lambda_P[j]*Pos[j];

        // Predicted gyro
        Vector3d gyro = omega[N];

        // Predicted acceleration
        Vector3d a_inW_Bt(0, 0, 0);
        for (int j = 0; j < N; j++)
            a_inW_Bt += lambda_P_ddot(j) * Pos[j];

        Vector3d a_plus_g = a_inW_Bt + GRAV;
        Vector3d acce = R_W_Bt.inverse() * a_plus_g;

        // Eigen::Map<Matrix<double, 12, 1>> residual(residual);
        residual.block<3, 1>(0, 0) = (gyro + biasW - imu_sample.gyro) * wGyro    ;
        residual.block<3, 1>(3, 0) = (acce + biasA - imu_sample.acce) * wAcce    ;
        residual.block<3, 1>(6, 0) = (biasW - imu_bias.gyro_bias)     * wBiasGyro;
        residual.block<3, 1>(9, 0) = (biasA - imu_bias.acce_bias)     * wBiasAcce;

        double resMax = residual.maxCoeff();
        double resMin = residual.minCoeff();

        // if (resMax > 1.0e9 || isnan(resMax) || isnan(resMin))
        // {
        //     ROS_WARN("numerical unstable in imu factor\n");
        //     cout << "gyro:  " << gyro << endl;
        //     cout << "acce:  " << acce << endl;
        //     cout << "biasW: " << biasW << endl;
        //     cout << "biasA: " << biasA << endl;
        //     cout << "gyro measurement: " << imu_sample.gyro << endl;
        //     cout << "acce measurement: " << imu_sample.acce << endl;            
        // }

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!computeJacobian)
            return true;

        /* #region Jacobian of angular velocity ---------------------------------------------------------------------*/

        // The inverse right Jacobian JrInv(dj) = d(deltaj)/d(Rj). Note that d( d[j+1] )/d( R[j] ) = -Jr(-d[j+1])
        Matrix3d ddelta_dR[N];
        for (int j = 0; j < N; j++)
            ddelta_dR[j] = rightJacobianInv(delta[j]);

        Matrix3d JrLambaDelta[N];
        for (int j = 0; j < N; j++)
            JrLambaDelta[j] = rightJacobian(lambda_R[j] * delta[j]);

        // Jacobian d(Rt)/d(Rj). Derived from the chain rule:
        // d(Rt)/d(Rj) = d(Rt(rho))/d(rho) [ d(rho)/d(dj) . d(dj)/d(Rj) + d(rho)/d(d[j+1]) d(d[j+1]))/d(Rj) ]
        // by using equation (57) in the TUM CVPR paper and some manipulation, we obtain
        // d(Rt)/d(R[j]) = lambda[j] * P[j] * Jr( lambda[j] delta[j] ) * Jr^-1(delta[j])
        //                 - lambda[j+1] * P[j+1] * Jr( lambda[j+1] delta[j+1] ) * Jr^-1( -delta[j+1] )
        Matrix3d dRt_dR[N];
        for (int j = 0; j < N; j++)
        {
            if (j == N - 1)
                dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j];
            else
                dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j] - lambda_R[j + 1] * P[j + 1].matrix() * JrLambaDelta[j + 1] * ddelta_dR[j + 1].transpose();
        }

        // Jacobian of d(omega)/d(deltaj)
        Matrix3d domega_ddelta[N];
        for (int j = 1; j < N; j++)
            domega_ddelta[j] = P[j].matrix() * (lambda_R(j) * A[j].matrix().transpose() * SO3d::hat(omega[j]) * JrLambaDelta[j].transpose()
                               + lambda_R_dot[j] * Matrix3d::Identity());
        // Jacobian of d(omega)/d(Rj)
        Matrix3d domega_dR[N];
        for (int j = 0; j < N; j++)
        {
            domega_dR[j].setZero();

            if (j == 0)
                domega_dR[j] = -domega_ddelta[1] * ddelta_dR[1].transpose();
            else if (j == N - 1)
                domega_dR[j] = domega_ddelta[j] * ddelta_dR[j];
            else
                domega_dR[j] = domega_ddelta[j] * ddelta_dR[j] - domega_ddelta[j + 1] * ddelta_dR[j + 1].transpose();

            // printf("slict: J_omega_R%d\n", j);
            // std::cout << domega_dR[j] << std::endl;
        }

        /* #endregion Jacobian of angular velocity ------------------------------------------------------------------*/

        /* #region Jacobian of acceleration -------------------------------------------------------------------------*/

        Matrix3d da_dRt = SO3d::hat(acce);

        // Jacobian of acceleration against rotation knots
        Matrix3d da_dR[N];
        for (int j = 0; j < N; j++)
        {
            da_dR[j] = da_dRt * dRt_dR[j];
            // printf("slict: J_a_R%d\n", j);
            // std::cout << da_dR[j] << std::endl;
        }

        /* #endregion Jacobian of acceleration ----------------------------------------------------------------------*/

        /// Rotation control point
        for (size_t j = 0; j < N; j++)
        {
            int ridx = j*6;
        
            Eigen::Block<Eigen::Matrix<double, 12, -1>, 12, 3> J_knot_R = jacobian.block<12, 3>(0, ridx);

            // from gyro residual
            J_knot_R.block<3, 3>(0, 0) = domega_dR[j] * wGyro;

            // from accel residual
            J_knot_R.block<3, 3>(3, 0) = da_dR[j] * wAcce;
        }

        /// Position control point
        for (size_t j = 0; j < N; j++)
        {
            int pidx = j*6 + 3;
            Eigen::Block<Eigen::Matrix<double, 12, -1>, 12, 3> J_knot_p = jacobian.block<12, 3>(0, pidx);

            // from accel residual
            J_knot_p.block<3, 3>(3, 0) = lambda_P_ddot[j] * R_W_Bt.inverse().matrix() * wAcce;
        }

        /// bias
        int bigidx = N*6;
        Eigen::Block<Eigen::Matrix<double, 12, -1>, 12, 3> J_bw = jacobian.block<12, 3>(0, bigidx);
        J_bw.block<3, 3>(0, 0) = Matrix3d::Identity() * wGyro;
        J_bw.block<3, 3>(6, 0) = Matrix3d::Identity() * wBiasGyro;

        int biaidx = N*6 + 3;
        Eigen::Block<Eigen::Matrix<double, 12, -1>, 12, 3> J_ba = jacobian.block<12, 3>(0, biaidx);
        J_ba.block<3, 3>(3, 0) = Matrix3d::Identity() * wAcce;
        J_ba.block<3, 3>(9, 0) = Matrix3d::Identity() * wBiasAcce;

        return true;
    }

    // Residual block
    Matrix<double, 12, 1> residual;
    
    // Jacobian block
    Matrix<double, 12, -1> jacobian;

private:
    ImuSample imu_sample;
    ImuBias imu_bias;
    Vector3d GRAV;

    double wGyro;
    double wAcce;
    double wBiasGyro;
    double wBiasAcce;

    int    N;
    double Dt; // Knot length
    double s;  // Normalized time (t - t_i)/Dt

    // Parameter blocks
    vector<double*> PARAM_ROT;
    vector<double*> PARAM_POS;
    double* PARAM_BIG;
    double* PARAM_BIA;

    // Lambda
    Matrix<double, Dynamic, 1> lambda_R;
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_R_dot;
    // Lambda ddot
    Matrix<double, Dynamic, 1> lambda_P_ddot;
};
