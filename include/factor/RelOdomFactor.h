#ifndef SLICT_RELODOMFACTOR_H_
#define SLICT_RELODOMFACTOR_H_

#include <ceres/ceres.h>
#include <Eigen/Eigen>

#include "../utility.h"

class RelOdomFactor : public ceres::SizedCostFunction<6, 7, 7>
{
  private:

    Eigen::Vector3d     p_odom_i;
    Eigen::Vector3d     p_odom_j;
    
    Eigen::Quaterniond  q_odom_i;
    Eigen::Quaterniond  q_odom_j;

    double p_odom_n = 0.25;
    double q_odom_n = 0.25;

    Eigen::Matrix<double, 6, 6> sqrt_info;

  public:

    RelOdomFactor() = delete;
    RelOdomFactor(

                 const Eigen::Vector3d     &_p_odom_i,
                 const Eigen::Vector3d     &_p_odom_j,

                 const Eigen::Quaterniond  &_q_odom_i,
                 const Eigen::Quaterniond  &_q_odom_j,

                 const double _p_odom_n,
                 const double _q_odom_n)
        : p_odom_i(_p_odom_i), p_odom_j(_p_odom_j),
          q_odom_i(_q_odom_i), q_odom_j(_q_odom_j),
          p_odom_n(_p_odom_n), q_odom_n(_q_odom_n)
    {
      // Eigen::Matrix<double, 3, 3> Roti           = q_odom_i.toRotationMatrix();
      // Eigen::Matrix<double, 3, 3> Roti_inv       = Roti.transpose();
      // Eigen::Matrix<double, 3, 3> DeltaRot       = Roti_inv*(q_odom_j.toRotationMatrix());

      double q_odom_n_sq = q_odom_n*q_odom_n;
      Eigen::Matrix<double, 3, 3> Pphi;
      Pphi << q_odom_n_sq, 0, 0,
              0, q_odom_n_sq, 0,
              0, 0, q_odom_n_sq;

      double p_odom_n_sq = p_odom_n*p_odom_n;
      Eigen::Matrix<double, 3, 3> Pnu;
      Pnu << p_odom_n_sq, 0, 0,
             0, p_odom_n_sq, 0,
             0, 0, p_odom_n_sq;

      // Calculate the covariance matrix
      Eigen::Matrix<double, 6, 6> P_vio = Eigen::Matrix<double, 6, 6>::Zero();
      P_vio.block<3, 3>(0, 0) = Pphi;
      P_vio.block<3, 3>(3, 3) = Pnu;

      // printf("Hello 3. P_vio.: row %d, col %d\n", P_vio.rows(), P_vio.cols());
      // std::cout << P_vio << std::endl;

      sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(P_vio.inverse()).matrixL().transpose();
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d    Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Map < Eigen::Matrix<double, 6, 1> > residual(residuals);

        Eigen::Vector3d     delta_p = q_odom_i.inverse()*(p_odom_j - p_odom_i);
        Eigen::Quaterniond  delta_q = q_odom_i.inverse()*(q_odom_j);

        residual.block<3, 1>(0, 0) = Qi.inverse() * (Pj - Pi) - delta_p;
        residual.block<3, 1>(3, 0) = 2 * (delta_q.inverse() * (Qi.inverse() * Qj)).vec();

        residual = sqrt_info * residual;

        if (jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) = -Qi.inverse().toRotationMatrix();

                jacobian_pose_i.block<3, 3>(0, 3) = Util::skewSymmetric( Qi.inverse() * (Pj - Pi) );

                jacobian_pose_i.block<3, 3>(3, 3) = -(Util::Qright(delta_q) * Util::Qleft(Qj.inverse() * Qi)).bottomRightCorner<3, 3>();

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    printf("numerical unstable in vio factor jacobian.\n");
                    // std::cout << pre_integration->jacobian << std::endl;
                    // ROS_BREAK();
                }

            }

            if(jacobians[1])
            {

                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(0, 0) = Qi.inverse().toRotationMatrix();

                // jacobian_pose_i.block<3, 3>(0, 3) = Util::skewSymmetric( Qi.inverse() * (Pj - Pi) );

                jacobian_pose_j.block<3, 3>(3, 3) = Util::Qleft(delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                if (jacobian_pose_j.maxCoeff() > 1e8 || jacobian_pose_j.minCoeff() < -1e8)
                {
                    printf("numerical unstable in vio factor jacobian.\n");
                    // std::cout << pre_integration->jacobian << std::endl;
                    // ROS_BREAK();
                }

            }

        }

        return true;
    }
};

#endif //SLICT_RELODOMFACTOR_H_