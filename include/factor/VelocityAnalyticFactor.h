#pragma once

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "utility.h"

using SO3d = Sophus::SO3<double>;
using SE3d = Sophus::SE3<double>;

class VelocityAnalyticFactor : public ceres::CostFunction
{
public:

    VelocityAnalyticFactor(const Vector3d &vel_meas_, double wV_, int N_, double Dt_, double s_)
    :   vel_meas    (vel_meas_      ),
        wV          (wV_            ),
        N           (N_             ),
        Dt          (Dt_            ),
        s           (s_             )
    {
        // 6-element residual: (3x1 omega, 3x1 a, 3x1 bw, 3x1 ba)
        set_num_residuals(3);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(3);

        // Calculate the spline-defining quantities:

        // Blending matrix for position, in standard form
        Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Inverse of knot length
        double Dt_inv = 1.0/Dt;

        // Time power derivative
        Matrix<double, Dynamic, 1> Udot = Matrix<double, Dynamic, 1>::Zero(N);
        for (int j = 1; j < N; j++)
            Udot(j) = j * std::pow(s, j - 1);

        // Lambda dot
        lambda_P_dot = Dt_inv * B * Udot;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        /* #region Map the memory to control points -----------------------------------------------------------------*/

        // Indexing offsets for the states
        size_t P_offset = 0;

        // Map parameters to the control point states
        Vector3d Pos[N];
        for (int j = 0; j < N; j++)
            Pos[j] = Eigen::Map<Vector3d const>(parameters[P_offset + j]);

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        // The following use the formulations in the paper 2020 CVPR:
        // "Efficient derivative computation for cumulative b-splines on lie groups."
        // Sommer, Christiane, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers.
        // Some errors in the paper is corrected

        // Predicted velocity
        Vector3d v_inW_Bt(0, 0, 0);
        for (int j = 0; j < N; j++)
            v_inW_Bt += lambda_P_dot[j]*Pos[j];

        // Velocity residual
        Vector3d del = (v_inW_Bt - vel_meas);

        // Residual
        Eigen::Map<Matrix<double, 3, 1>> residual(residuals);
        residual = wV*del;

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        size_t idx;
        for (size_t j = 0; j < N; j++)
        {
            idx = P_offset + j;
            if (jacobians[idx])
            {
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_knot_p(jacobians[idx]);
                J_knot_p.setZero();

                double wV_lambda_P_dot = wV*lambda_P_dot[j];

                /// for velocity residual
                J_knot_p.block<3, 3>(0, 0) = Vector3d(wV_lambda_P_dot, wV_lambda_P_dot, wV_lambda_P_dot).asDiagonal();
            }
        }

        return true;
    }

private:

    Vector3d vel_meas;

    double wV;

    int    N;
    double Dt;     // Knot length
    double s;      // Normalized time (t - t_i)/Dt

    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_P_dot;
};
