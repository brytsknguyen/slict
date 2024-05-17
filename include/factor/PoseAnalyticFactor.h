#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

class PoseAnalyticFactor : public ceres::CostFunction
{
public:

    PoseAnalyticFactor(const SE3d &pose_meas_, double wR_, double wP_, int N_, double Dt_, double s_)
    :   pose_meas   (pose_meas_      ),
        wR          (wR_             ),
        wP          (wP_             ),
        N           (N_              ),
        Dt          (Dt_             ),
        s           (s_              )
    {
        // 6-element residual: (3x1 omega, 3x1 a, 3x1 bw, 3x1 ba)
        set_num_residuals(6);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(4);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(3);

        // Calculate the spline-defining quantities:

        // Blending matrix for position, in standard form
        Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Blending matrix for rotation, in cummulative form
        Matrix<double, Dynamic, Dynamic> Btilde = basalt::computeBlendingMatrix<double, true>(N);

        // Inverse of knot length
        // double Dt_inv = 1.0/Dt;

        // Time powers
        Matrix<double, Dynamic, 1> U(N);
        for(int j = 0; j < N; j++)
            U(j) = std::pow(s, j);

        // Lambda for p
        lambda_P = B*U;

        // Lambda for R
        lambda_R = Btilde*U;
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

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Indexing offsets for the states
        size_t R_offset = 0;            // for quaternion
        size_t P_offset = R_offset + N; // for position
        size_t B_offset = P_offset + N; // for bias

        // Map parameters to the control point states
        SO3d Rot[N];
        Vector3d Pos[N];
        for (int j = 0; j < N; j++)
        {
            Rot[j] = Eigen::Map<SO3d const>(parameters[R_offset + j]);
            Pos[j] = Eigen::Map<Vector3d const>(parameters[P_offset + j]);

            // printf("slict: p%d. lambda_a: %f\n", j, lambda_P_ddot(j));
            // std::cout << pos[j].transpose() << std::endl;
        }

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

        // Predicted orientation from Rt^-1 = P(N-1)R(0)^-1
        SO3d R_W_Bt = (P[0] * Rot[0].inverse()).inverse();

        // Predicted position
        Vector3d p_inW_Bt(0, 0, 0);
        for (int j = 0; j < N; j++)
            p_inW_Bt += lambda_P[j]*Pos[j];

        // Rotational residual
        Vector3d phi = (pose_meas.so3().inverse()*R_W_Bt).log();

        // Positional residual
        Vector3d del = (p_inW_Bt - pose_meas.translation());

        // Residual
        Eigen::Map<Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wR*phi;
        residual.block<3, 1>(3, 0) = wP*del;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        /* #region Jacobian of r_deltaRot ---------------------------------------------------------------------------*/

        // The inverse right Jacobian Jr(dj) = d(deltaj)/d(Rj). Note that d( d[j+1] )/d( R[j] ) = -Jr(-d[j+1])
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

        // Jacobian on Rt
        Matrix3d dDeltaR_dRt = rightJacobianInv(phi);

        // Jacobian on Rj
        Matrix3d dDeltaR_dR[N];
        for (int j = 0; j < N; j++)
            dDeltaR_dR[j] = dDeltaR_dRt*dRt_dR[j];

        /* #endregion Jacobian of r_deltaRot ------------------------------------------------------------------------*/

        /* #region Jacobian of r_deltaPos ---------------------------------------------------------------------------*/
        
        Matrix3d ddeltaPos_dP[N];
        for (int j = 0; j < N; j++)
            ddeltaPos_dP[j] = Vector3d(lambda_P[j], lambda_P[j], lambda_P[j]).asDiagonal();

        /* #endregion Jacobian of r_deltaPos ------------------------------------------------------------------------*/

        size_t idx;

        /// Rotation control point
        for (size_t j = 0; j < N; j++)
        {
            idx = R_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J_knot_R(jacobians[idx]);
                J_knot_R.setZero();

                // for gyro residual
                J_knot_R.block<3, 3>(0, 0) = wR*dDeltaR_dR[j];
            }
        }

        /// Position control point
        for (size_t j = 0; j < N; j++)
        {
            idx = P_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J_knot_p(jacobians[idx]);
                J_knot_p.setZero();

                /// for accel residual
                J_knot_p.block<3, 3>(3, 0) = wP*ddeltaPos_dP[j];
            }
        }

        return true;
    }

private:

    SE3d pose_meas;

    double wR;
    double wP;

    int    N;
    double Dt;     // Knot length
    double s;      // Normalized time (t - t_i)/Dt

    // Lambda
    Matrix<double, Dynamic, 1> lambda_R;
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_P;
};
