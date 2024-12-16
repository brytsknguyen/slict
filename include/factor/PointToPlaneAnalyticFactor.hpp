/**
* This file is part of slict.
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
* slict is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* slict is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with slict.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include <ceres/ceres.h>
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/utils/sophus_utils.hpp"
#include "../utility.h"

using ufoSurfelMap    = ufo::map::SurfelMap;
using ufoSurfelMapPtr = boost::shared_ptr<ufoSurfelMap>;
using ufoNode         = ufo::map::NodeBV;
using ufoSphere       = ufo::geometry::Sphere;
using ufoPoint3       = ufo::map::Point3;

using namespace Eigen;

struct assocSettings
{
    assocSettings(bool    use_ufm_,
                  bool    reassoc_,
                  int     reassoc_rate_,
                  int     &surfel_min_point_,
                  double  &surfel_min_plnrty_,
                  double  &surfel_intsect_rad_,
                  double  &dis_to_surfel_max_,
                  double  &lidar_weight_,
                  uint32_t factor_idx_
                 )
    {
        use_ufm            = use_ufm_;
        reassoc            = reassoc_;
        reassoc_rate       = reassoc_rate_;
        surfel_min_point   = surfel_min_point_;
        surfel_min_plnrty  = surfel_min_plnrty_;
        surfel_intsect_rad = surfel_intsect_rad_;
        dis_to_surfel_max  = dis_to_surfel_max_;
        lidar_weight       = lidar_weight_;
        factor_idx         = factor_idx_;        
    };

    bool     use_ufm            = false;
    bool     reassoc            = true;
    int      reassoc_rate       = 2;
    int      surfel_min_point   = 6;
    double   surfel_min_plnrty  = 0.7;
    double   surfel_intsect_rad = 0.2;
    double   dis_to_surfel_max  = 0.3;
    double   lidar_weight       = 20;
    uint32_t factor_idx         = 0;
};

template <class Predicates>
class PointToPlaneAnalyticFactor: public ceres::CostFunction
{
public:

    // Destructor
    ~PointToPlaneAnalyticFactor()
    {
        delete finW_opt  ;
        delete n_opt     ;
        delete m_opt     ;
        delete w_opt     ;
        delete iteration ;
    };

    // Constructor
    PointToPlaneAnalyticFactor(const Vector3d &finW_, const Vector3d &f_, const Vector4d &coef, double w_,
                               int N_, double Dt_, double s_, ufoSurfelMapPtr &surfMap_, Predicates &commonPred_,
                               ikdtreePtr &ikdtMap_, assocSettings &settings_)
    :   finW       (finW_            ),
        f          (f_               ),
        n          (coef.head<3>()   ),
        m          (coef.tail<1>()(0)),
        w          (w_               ),
        N          (N_               ),
        Dt         (Dt_              ),
        s          (s_               ),
        surfMap    (surfMap_         ),
        ikdtMap    (ikdtMap_         ),
        commonPred (commonPred_      ),
        settings   (settings_        )
    {
        // 1-element residual: n^T*(Rt*f + pt) + m
        set_num_residuals(1);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(4);

        for (size_t j = 0; j < N; j++)
            mutable_parameter_block_sizes()->push_back(3);

        // Calculate the spline-defining quantities:

        // Blending matrix for position, in standard form
        Matrix<double, Dynamic, Dynamic> B = basalt::computeBlendingMatrix<double, false>(N);

        // Blending matrix for rotation, in cummulative form
        Matrix<double, Dynamic, Dynamic> Btilde = basalt::computeBlendingMatrix<double, true>(N);

        // // Inverse of knot length
        // double Dt_inv = 1.0/Dt;

        // Time powers
        Matrix<double, Dynamic, 1> U(N);
        for(int j = 0; j < N; j++)
            U(j) = std::pow(s, j);

        // Lambda for p
        lambda_P = B*U;

        // Lambda for R
        lambda_R = Btilde*U;
        
        // Initializing state-dependent variables
        finW_opt   = new Vector3d();
        n_opt      = new Vector3d();
        m_opt      = new double;
        w_opt      = new double;
        iteration  = new int;
        
        *finW_opt   =  finW;
        *n_opt      =  n;
        *m_opt      =  m;
        *w_opt      =  w;
        *iteration  = -1;
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
        *iteration += 1;

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
            Rot[j] = Map<SO3d const>(parameters[R_offset + j]);
            Pos[j] = Map<Vector3d const>(parameters[P_offset + j]);

            // printf("slict: p%d. lambda_a: %f\n", j, lambda_P_ddot(j));
            // std::cout << pos[j].transpose() << std::endl;
        }

        /* #endregion Map the memory to control points --------------------------------------------------------------*/

        /* #region Calculate the pose at sampling time --------------------------------------------------------------*/

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

        /* #endregion Calculate the pose at sampling time -----------------------------------------------------------*/
    
        /* #region Recalculate the normal, mean and noise -----------------------------------------------------------*/

        // Find the feature in the world frame
        Vector3d f_inW = R_W_Bt * f + p_inW_Bt;

        double shift = (f_inW - *finW_opt).norm();
        double currDis = (*n_opt).transpose() * (f_inW) + *m_opt;

        if( settings.reassoc 
            && shift > settings.surfel_intsect_rad
            && (*iteration != 0 && *iteration % settings.reassoc_rate == 0) // Only enforced after at least one iteration
            // && curr_depth > *last_depth
          )
        {
            if(settings.use_ufm)
            {
                // Query the surfel map with predicates
                namespace ufopred = ufo::map::predicate;
                auto pred = commonPred && ufopred::Intersects(ufoSphere(ufoPoint3(f_inW(0), f_inW(1), f_inW(2)), settings.surfel_intsect_rad));

                map<int, vector<ufoNode>> mapNodes;
                for (const ufoNode &node : surfMap->queryBV(pred))
                    mapNodes[node.depth()].push_back(node);

                double minDis = -1;
                ufoNode assocNode;
                bool point_associated = false;
                for(auto A : mapNodes)
                {
                    // Find the min distance
                    int depth = A.first;
                    vector<ufoNode> &nodes = A.second;

                    for(ufoNode &node : nodes)
                    {
                        auto const& surfel = surfMap->getSurfel(node);

                        Vector3d mean = ufo::math::toEigen(surfel.getMean());
                        Vector3d norm = ufo::math::toEigen(surfel.getNormal());

                        // printf("Surfel: %d. %d. %f.\n", depth, surfel.getNumPoints(), surfel.getPlanarity());

                        // ROS_ASSERT_MSG(planarity >= 0 && planarity <= 1.0, "plnrty: %f\n", planarity);
                        double d2pln = fabs(norm.dot(f_inW - mean));

                        if (d2pln > settings.dis_to_surfel_max)
                            continue;

                        if (minDis == -1 || d2pln < minDis)
                        {
                            minDis = d2pln;
                            point_associated = true;
                            assocNode = node;
                        }
                    }

                    if (point_associated)
                        break;
                }

                if(point_associated)
                {
                    auto const& surfel = surfMap->getSurfel(assocNode);
                    Vector3d mean = ufo::math::toEigen(surfel.getMean());
                    Vector3d norm = ufo::math::toEigen(surfel.getNormal());
                    // double score = (1 - 0.9 * minDis / f_inB.norm())*surfel.getPlanarity();

                    *finW_opt = f_inW;
                    *n_opt    = norm;
                    *m_opt    = -norm.dot(mean);
                    *w_opt    = settings.lidar_weight*surfel.getPlanarity();

                    // printf(KYEL "Factor %06d reassociating!. Shift: %f. Old d2p: %f. New d2p: %f.\n" RESET,
                    //    settings.factor_idx, shift, currDis, minDis);
                }
                // else
                // {
                //     // printf(KYEL "NO ASSOCIATION FOUND\n" RESET);
                //     *w_opt = 0;
                // }
            }
            else
            {
                PointXYZI pinW; pinW.x = f_inW(0); pinW.y = f_inW(1); pinW.z = f_inW(2);

                int numNbr = settings.surfel_min_point;
                ikdtPointVec nbrPoints;
                vector<float> knnSqDis;
                ikdtMap->Nearest_Search(pinW, numNbr, nbrPoints, knnSqDis);

                if (nbrPoints.size() < numNbr)
                    ;
                // else if (knnSqDis[numNbr - 1] > 3.0)
                //     ;
                else
                {
                    Vector4d pabcd;
                    double rho;
                    if(Util::fitPlane(nbrPoints, settings.surfel_min_plnrty, settings.dis_to_surfel_max, pabcd, rho))
                    {
                        float d2p = pabcd(0) * f_inW(0) + pabcd(1) * f_inW(1) + pabcd(2) * f_inW(2) + pabcd(3);
                        float score = (1 - 0.9 * fabs(d2p) / f.norm())*rho;
                        // float score = 1 - 0.9 * fabs(d2p) / (1 + pow(f.norm(), 4));
                        // float score = 1;
                        
                        if (score > 0.05)
                        {
                            // Add to coeff
                            *finW_opt = f_inW;
                            *n_opt    = pabcd.head<3>();
                            *m_opt    = pabcd(3);
                            *w_opt    = settings.lidar_weight*score;
                        }
                    }
                }
            }
        }

        /* #endregion Recalculate the normal, mean and noise --------------------------------------------------------*/

        /* #region Calculate the residual ---------------------------------------------------------------------------*/

        // Residual
        residuals[0] = (*w_opt)*((*n_opt).transpose() * (R_W_Bt * f + p_inW_Bt) + *m_opt);

        // if (residuals[0] > 1.0e9 || std::isnan(residuals[0]))
        // {
        //     ROS_WARN("numerical unstability in lidar factor\n");
        //     cout << "f: " << f << endl;
        //     cout << "n: " << n << endl;
        //     cout << "m: " << m << endl;
        //     cout << "s: " << s << endl;
        // }

        /* #endregion Calculate the residual ------------------------------------------------------------------------*/

        if (!jacobians)
            return true;

        /* #region Jacobian of dis on knot_R ------------------------------------------------------------------------*/

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
        Matrix<double, 1, 3> ddis_dRt = -n.transpose()*R_W_Bt.matrix()*SO3d::hat(f);

        // Jacobian on Rj
        Matrix<double, 1, 3> ddis_dR[N];
        for(int j = 0; j < N; j++)
            ddis_dR[j] = ddis_dRt*dRt_dR[j];

        /* #endregion Jacobian of dis on knot_R ---------------------------------------------------------------------*/

        /* #region Jacobian of dis on knot P ------------------------------------------------------------------------*/
        
        Matrix<double, 1, 3> ddis_dP[N];
        for (int j = 0; j < N; j++)
            ddis_dP[j] = (*n_opt).transpose()*Vector3d(lambda_P[j], lambda_P[j], lambda_P[j]).asDiagonal();

        /* #endregion Jacobian of dis on knot P ---------------------------------------------------------------------*/

        size_t idx;

        /// Rotation control point
        for (size_t j = 0; j < N; j++)
        {
            idx = R_offset + j;
            if (jacobians[idx])
            {
                Map<Matrix<double, 1, 4, RowMajor>> J_knot_R(jacobians[idx]);
                J_knot_R.setZero();

                // for gyro residual
                J_knot_R.block<1, 3>(0, 0) = (*w_opt)*ddis_dR[j];
            }
        }

        /// Position control point
        for (size_t j = 0; j < N; j++)
        {
            idx = P_offset + j;
            if (jacobians[idx])
            {
                Map<Matrix<double, 1, 3, RowMajor>> J_knot_p(jacobians[idx]);
                J_knot_p.setZero();

                /// for accel residual
                J_knot_p.block<1, 3>(0, 0) = (*w_opt)*ddis_dP[j];
            }
        }

        return true;
    }

private:
    
    // Settings for the associations
    assocSettings settings;

    // Feature coordinates in world frame
    Vector3d finW;

    // Feature coordinates in body frame
    Vector3d f;

    // Plane normal
    Vector3d n;

    // Plane offset
    double m;

    // Weight
    double w = 0.1;

    // Point coordinate in the world frame at the creation of the fact
    Vector3d *finW_opt;

    // Plane normal optimized
    Vector3d *n_opt;

    // Plane offset optimized
    double *m_opt;

    // Weight
    double *w_opt;

    // Iterations
    int *iteration;

    int    N;
    double Dt;
    double s;

    // Lambda
    Matrix<double, Dynamic, 1> lambda_R;
    
    // Lambda dot
    Matrix<double, Dynamic, 1> lambda_P;

    // Pointer to the map
    ufoSurfelMapPtr surfMap;

    // Pointer to the ikdtree
    ikdtreePtr ikdtMap;

    // Predicates
    Predicates commonPred;
};