/**
* This file is part of SLICT.
*
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
*
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/britsknguyen/SLICT>.
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
// Created by Thien-Minh Nguyen on 15/12/20.
//

#include <sys/stat.h>
#include <boost/format.hpp>
#include <deque>
#include <thread>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/kdtree/kdtree_flann.h>

#ifndef CLOUDMATCHER_HPP
#define CLOUDMATCHER_HPP

using namespace std;
using namespace Eigen;
using namespace pcl;

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "utility.h"

struct IOAOptions
{
    bool show_report;
    int  max_iterations = 3;
    double DJ_end = 0.001;
    string text;
    mytf init_tf;
    ceres::TrustRegionStrategyType trustRegType = ceres::DOGLEG;                                    // LEVENBERG_MARQUARDT, DOGLEG
    ceres::DenseLinearAlgebraLibraryType linAlgbLib = ceres::DenseLinearAlgebraLibraryType::EIGEN;  // EIGEN, LAPACK, CUDA
};

struct IOASummary
{
    bool converged = false;
    int iterations = 0;
    double JM = -1;
    double J0 = -1;
    double JK = -1;
    double JKavr = -1;
    double DJ = -1;
    double process_time = 0;
    mytf final_tf;
};

#ifndef POINTTOPLANDISFACTOR_H
#define POINTTOPLANDISFACTOR_H

class PointToPlaneFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    PointToPlaneFactor() = delete;
    PointToPlaneFactor(const Vector3d &point,
                       const Vector4d &coeff) : point_(point), sqrt_info_static_(1.0)
    {
        normal_ << coeff.x(), coeff.y(), coeff.z();
        offset_ =  coeff.w();
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Vector3d    Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

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
    double offset_;

    double sqrt_info_static_;
};

#endif

class CloudMatcher
{
private:

    double minMatchSqDis = 1.0;
    double minPlaneDis   = 0.05;

    int knn_size = 5;
    double knn_nbr_score_min = 0.1;


public:

    // Destructor
   ~CloudMatcher() {};

    CloudMatcher(double minMatchSqDis_, double minPlaneDis_)
    : minMatchSqDis(minMatchSqDis_), minPlaneDis(minPlaneDis_)
    {};

    // shared_ptr<CloudMatcher> ObjPtr();

    void IterateAssociateOptimize(IOAOptions &ioaOpt, IOASummary &ioaSum, const CloudXYZIPtr &refSurfMap, const CloudXYZIPtr &srcSurfCloud)
    {
        TicToc tt_proc_time;
        TicToc tt_assoc;
        TicToc tt_solve;

        double surf_assoc_time = 0, edge_assoc_time = 0;

        // Get the input parameters
        mytf tf_Ref_Src_init = ioaOpt.init_tf;
        int &maxF2MOptIters = ioaOpt.max_iterations;

        // Initialize the summary
        bool   &converged = ioaSum.converged;
        int    &actualItr = ioaSum.iterations;
        double &JM        = ioaSum.JM;
        double &J0        = ioaSum.J0;
        double &JK        = ioaSum.JK;
        double &JKavr     = ioaSum.JKavr;
        double &DJ        = ioaSum.DJ;
        mytf &tf_Ref_Src_iao = ioaSum.final_tf;

        JM = 0; J0 = 0; JK = 0;
        tf_Ref_Src_iao = tf_Ref_Src_init;

        // Back up the transform
        IOASummary ioaSumPrev = ioaSum;

        deque<Vector3d> srcSurfCoord;
        deque<Vector4d> srcSurfCoeff;
        int totalSurfF2M = 0, totalEdgeF2M = 0;

        pcl::KdTreeFLANN<PointXYZI> kdTreeSurfFromMap;
        kdTreeSurfFromMap.setInputCloud(refSurfMap);

        double *PARAM_POSE;
        PARAM_POSE = new double[7];

        int iter = -1;
        static int opt_num = -1; opt_num++;
        while(iter < maxF2MOptIters)
        {
            iter++;

            /* #region Find associations --------------------------------------------------------------------------*/

            tt_assoc.Tic();

            // Build the map
            CalSurfF2MCoeff(kdTreeSurfFromMap, refSurfMap, srcSurfCloud, tf_Ref_Src_init, srcSurfCoord, srcSurfCoeff, totalSurfF2M, surf_assoc_time);

            tt_assoc.Toc();

            /* #endregion Find associations -----------------------------------------------------------------------*/


            /* #region Solve the problem --------------------------------------------------------------------------*/

            tt_solve.Tic();

            // Create variable & initialize
            PARAM_POSE[0] = tf_Ref_Src_init.pos(0);
            PARAM_POSE[1] = tf_Ref_Src_init.pos(1);
            PARAM_POSE[2] = tf_Ref_Src_init.pos(2);
            PARAM_POSE[3] = tf_Ref_Src_init.rot.x();
            PARAM_POSE[4] = tf_Ref_Src_init.rot.y();
            PARAM_POSE[5] = tf_Ref_Src_init.rot.z();
            PARAM_POSE[6] = tf_Ref_Src_init.rot.w();

            // Create ceres problem and settings
            ceres::Problem problem;
            // Settings
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_SCHUR;
            options.trust_region_strategy_type = ioaOpt.trustRegType;
            options.dense_linear_algebra_library_type = ioaOpt.linAlgbLib;
            options.num_threads = MAX_THREADS;

            // Add parameter blocks
            ceres::Manifold *local_parameterization = new SO3xR3Manifold();
            problem.AddParameterBlock(PARAM_POSE, 7, local_parameterization);

            // Add factors
            ceres::LossFunction *huber_loss_function = new ceres::HuberLoss(1.0);

            // Surface factors
            int Nsurf = srcSurfCoord.size();
            double surf_init_cost = 0, surf_final_cost = -1;
            vector<ceres::internal::ResidualBlock *> res_ids_proj_surf;
            for (int j = 0; j < Nsurf; j++)
            {
                PointToPlaneFactor *f = new PointToPlaneFactor(srcSurfCoord[j], srcSurfCoeff[j]);
                ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f, huber_loss_function, PARAM_POSE);
                res_ids_proj_surf.push_back(res_id);
            }

            // Calculate the initial cost
            ceres::Problem::EvaluateOptions e_option;
            e_option.num_threads = MAX_THREADS;

            Util::ComputeCeresCost(res_ids_proj_surf, surf_init_cost, problem);
            // Util::ComputeCeresCost(res_ids_proj_edge, edge_init_cost, problem);

            // Solve the problem
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            Util::ComputeCeresCost(res_ids_proj_surf, surf_final_cost, problem);
            // Util::ComputeCeresCost(res_ids_proj_edge, edge_final_cost, problem);

            // Unload the estimate to the states
            Vector3d Ps(PARAM_POSE[0], PARAM_POSE[1], PARAM_POSE[2]);
            Quaternd Qs(PARAM_POSE[6], PARAM_POSE[3], PARAM_POSE[4], PARAM_POSE[5]);

            tf_Ref_Src_iao = mytf(Qs, Ps);

            double t_slv = tt_solve.Toc();

            /* #endregion Solve the problem -----------------------------------------------------------------------*/


            /* #region Report the estimate ------------------------------------------------------------------------*/

            Vector3d ypr_init = Util::Quat2YPR(tf_Ref_Src_init.rot);
            Vector3d ypr_last = Util::Quat2YPR(tf_Ref_Src_iao.rot);
            Vector3d pos_init = tf_Ref_Src_init.pos;
            Vector3d pos_last = tf_Ref_Src_iao.pos;

            int total_surf_features = srcSurfCloud->size();
            int total_surf_factors = res_ids_proj_surf.size();

            double JKavr_surf = (surf_final_cost == 0 || total_surf_factors == 0)
                                ? 0 : (surf_final_cost / total_surf_factors) * ((double)(total_surf_features) / total_surf_factors);

            double DJ = summary.initial_cost - summary.final_cost;

            JM = (iter == 0) ? summary.initial_cost : JM;
            J0 = summary.initial_cost;
            JK = summary.final_cost;
            JKavr = JKavr_surf;

            // Reload the initial guess with the optimized for the next iteration
            actualItr = iter;
            tf_Ref_Src_init = tf_Ref_Src_iao;

            if (ioaOpt.show_report)
            {
                printf(KBLU
                    "%s. Try: %2d. Itr: %2d. "
                    "t_assoc: %3.0f. t_slv: %3.0f.\n"
                    "Pinit: %7.2f. %7.2f. %7.2f. YPR: %4.0f. %4.0f. %4.0f.\n"
                    "Plast: %7.2f. %7.2f. %7.2f. YPR: %4.0f. %4.0f. %4.0f.\n"
                    "J0: %9.3f. Surf: %9.3f. Factors: Surf: %3d.\n"
                    "JK: %9.3f. Surf: %9.3f. Javrage: %f\n"
                    "DJ: %9.3f. Surf: %9.3f. %s\n"
                    RESET,
                    ioaOpt.text.c_str(), iter, static_cast<int>(summary.iterations.size()),
                    tt_assoc.GetLastStop(),
                    tt_solve.GetLastStop(),
                    pos_init.x(), pos_init.y(), pos_init.z(),
                    ypr_init.x(), ypr_init.y(), ypr_init.z(),
                    pos_last.x(), pos_last.y(), pos_last.z(),
                    ypr_last.x(), ypr_last.y(), ypr_last.z(),
                    summary.initial_cost, surf_init_cost,
                    res_ids_proj_surf.size(),
                    summary.final_cost, surf_final_cost,
                    JKavr,
                    summary.initial_cost - summary.final_cost,
                    surf_init_cost - surf_final_cost,
                    DJ < 0.001 ? "CONVERGED!" : "");
            }

            if (iter == 0)
            {
                ioaSumPrev.JM = JM;
                ioaSumPrev.J0 = J0;
                ioaSumPrev.JK = JK;
                ioaSumPrev.JKavr = JKavr;
            }

            /* #endregion Report the estimate ---------------------------------------------------------------------*/


            // Quit early if change is very small
            if (DJ < ioaOpt.DJ_end)
            {
                converged = true;
                break;
            }
        }

        // Add a line break
        if (ioaOpt.show_report)
            printf("\n");

        ioaSum.process_time = tt_proc_time.Toc();
    }

    bool CheckICP(CloudXYZIPtr &ref_pcl, CloudXYZIPtr &src_pcl, Matrix4f relPosIcpGuess, Matrix4f &relPosIcpEst,
                  double hisKFSearchRadius, int icp_max_iters, double icpFitnessThres, double &icpFitnessRes, double &ICPtime)
    {

        /* #region Calculate the relative pose constraint ---------------------------------------------------------------*/

        TicToc tt_icp;

        pcl::IterativeClosestPoint<PointXYZI, PointXYZI> icp;
        icp.setMaxCorrespondenceDistance(hisKFSearchRadius * 2);
        icp.setMaximumIterations(icp_max_iters);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        icp.setInputSource(src_pcl);
        icp.setInputTarget(ref_pcl);

        CloudXYZIPtr aligned_result(new CloudXYZI());

        icp.align(*aligned_result, relPosIcpGuess);

        bool icp_passed   = false;
        bool icpconverged = icp.hasConverged();
        icpFitnessRes     = icp.getFitnessScore();
        relPosIcpEst      = icp.getFinalTransformation();

        ICPtime = tt_icp.Toc();

        if (!icpconverged || icpFitnessRes > icpFitnessThres)
        {
            // if (extended_report)
            // {
            //     printf(KRED "\tICP time: %9.3f ms. ICP %s. Fitness: %9.3f, threshold: %3.1f\n" RESET,
            //     tt_icp.GetLastStop(),
            //     icpconverged ? "converged" : "fails to converge",
            //     icpFitnessRes, icpFitnessThres);
            // }
        }
        else
        {
            // if (extended_report)
            // {
            //     printf(KBLU "\tICP time: %9.3f ms. ICP %s. Fitness: %9.3f, threshold: %3.1f\n" RESET,
            //         tt_icp.GetLastStop(),
            //         icpconverged ? "converged" : "fails to converge",
            //         icpFitnessRes, icpFitnessThres);
            // }

            icp_passed = true;
        }

        return icp_passed;

        /* #endregion Calculate the relative pose constraint ------------------------------------------------------------*/

    }

    void CalSurfF2MCoeff(const pcl::KdTreeFLANN<PointXYZI> &kdTreeSurfFromMap, const CloudXYZIPtr &refSurfMap, const CloudXYZIPtr &srcSurfMap,
                         mytf tf_Ref_Src, deque<Vector3d> &srcSurfCoord, deque<Vector4d> &srcSurfCoeff,
                         int &totalSurfF2M, double &time_surf_assoc)
    {
        TicToc tt_surf_assoc;

        if (refSurfMap->size() > knn_size)
        {
            int pointsCount = srcSurfMap->points.size();
            deque<LidarCoef> CloudCoefTemp(pointsCount);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int i = 0; i < pointsCount; i++)
            {
                PointXYZI srcPointInFsrc = srcSurfMap->points[i];
                PointXYZI srcPointInFref = Util::transform_point(tf_Ref_Src, srcPointInFsrc);

                CloudCoefTemp[i].n = Vector4d(0, 0, 0, 0);
                CloudCoefTemp[i].t = -1;

                if(!Util::PointIsValid(srcPointInFsrc))
                {
                    // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                    srcPointInFsrc.x = 0; srcPointInFsrc.y = 0; srcPointInFsrc.z = 0; srcPointInFsrc.intensity = 0;
                    continue;
                }

                // Calculating the coefficients
                MatrixXd mat_A0(knn_size, 3);
                MatrixXd mat_B0(knn_size, 1);
                Vector3d mat_X0;
                Matrix3d mat_A1;
                MatrixXd mat_D1(1, 3);
                Matrix3d mat_V1;

                mat_A0.setZero();
                mat_B0.setConstant(-1);
                mat_X0.setZero();

                mat_A1.setZero();
                mat_D1.setZero();
                mat_V1.setZero();

                vector<int> knn_idx(knn_size, 0); vector<float> knn_sq_dis(knn_size, 0);
                kdTreeSurfFromMap.nearestKSearch(srcPointInFref, knn_size, knn_idx, knn_sq_dis);

                if (knn_sq_dis.back() < minMatchSqDis)
                {
                    for (int j = 0; j < knn_idx.size(); j++)
                    {
                        mat_A0(j, 0) = refSurfMap->points[knn_idx[j]].x;
                        mat_A0(j, 1) = refSurfMap->points[knn_idx[j]].y;
                        mat_A0(j, 2) = refSurfMap->points[knn_idx[j]].z;
                    }
                    mat_X0 = mat_A0.colPivHouseholderQr().solve(mat_B0);

                    float pa = mat_X0(0, 0);
                    float pb = mat_X0(1, 0);
                    float pc = mat_X0(2, 0);
                    float pd = 1;

                    float ps = sqrt(pa * pa + pb * pb + pc * pc);
                    pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                    // NOTE: plane as (x y z)*w+1 = 0

                    bool valid_plane = true;
                    for (int j = 0; j < knn_idx.size(); j++)
                    {
                        if (fabs(pa * refSurfMap->points[knn_idx[j]].x +
                                 pb * refSurfMap->points[knn_idx[j]].y +
                                 pc * refSurfMap->points[knn_idx[j]].z + pd) > minPlaneDis)
                        {
                            valid_plane = false;
                            break;
                        }
                    }

                    if (valid_plane)
                    {
                        float pd2 = pa * srcPointInFref.x + pb * srcPointInFref.y + pc * srcPointInFref.z + pd;

                        // Weightage based on close the point is to the plane ?
                        float score = 1 - 0.9f * fabs(pd2) / Util::pointDistance(srcPointInFref);

                        if (score > knn_nbr_score_min)
                        {
                            CloudCoefTemp[i].t    = 0;
                            CloudCoefTemp[i].f    = Vector3d(srcPointInFsrc.x, srcPointInFsrc.y, srcPointInFsrc.z);
                            CloudCoefTemp[i].fdsk = Vector3d(srcPointInFsrc.x, srcPointInFsrc.y, srcPointInFsrc.z);
                            CloudCoefTemp[i].n    = Vector4d(score * pa, score * pb, score * pc, score * pd);
                        }
                    }
                }
            }

            // Copy the coefficients to the buffer
            srcSurfCoord.clear();
            srcSurfCoeff.clear();

            for(auto &coef : CloudCoefTemp)
            {
                if (coef.t >= 0)
                {
                    srcSurfCoord.push_back(coef.f);
                    srcSurfCoeff.push_back(coef.n);
                    totalSurfF2M++;
                }
            }
        }

        time_surf_assoc = tt_surf_assoc.Toc();
    }

    void CalEdgeF2MCoeff(const CloudXYZIPtr &refEdgeMap, const CloudXYZIPtr &srcEdgeCloud,
                         mytf tf_Ref_Src, deque<Vector3d> &srcEdgeCoord, deque<Vector4d> &srcEdgeCoeff,
                         int &totalEdgeF2M, double &time_edge_assoc)
    {
        TicToc tt_edge_assoc;

        // tt_edge_assoc.Tic();

        // srcEdgeCoord.clear();
        // srcEdgeCoeff.clear();

        // vector<int> knn_idx(5, 0);
        // vector<float> knn_sq_dis(5, 0);
        // Matrix<float, 5, 3> mat_A0;
        // Matrix<float, 5, 1> mat_B0;
        // Vector3f mat_X0;
        // Matrix3f mat_A1;
        // Matrix<float, 1, 3> mat_D1;
        // Matrix3f mat_V1;

        // mat_A0.setZero();
        // mat_B0.setConstant(-1);
        // mat_X0.setZero();

        // mat_A1.setZero();
        // mat_D1.setZero();
        // mat_V1.setZero();

        // if (refEdgeMap->size() > 5)
        // {
        //     pcl::KdTreeFLANN<PointXYZI> kdTreeEdgeFromMap;
        //     kdTreeEdgeFromMap.setInputCloud(refEdgeMap);
        //     int edge_points_size = srcEdgeCloud->points.size();
        //     // region Corner points
        //     for (int i = 0; i < edge_points_size; i++)
        //     {
        //         PointXYZI srcEdgePoint = srcEdgeCloud->points[i];
        //         Vector3d srcEdgePoint_(srcEdgePoint.x, srcEdgePoint.y, srcEdgePoint.z);

        //         PointXYZI pivEdgePoint = Util::transform_point(tf_Ref_Src, srcEdgePoint);

        //         int knn_size = 5;
        //         kdTreeEdgeFromMap.nearestKSearch(pivEdgePoint, knn_size,
        //                                          knn_idx, knn_sq_dis);

        //         if (knn_sq_dis.back() < minMatchSqDis)
        //         {
        //             Vector3d centroid(0, 0, 0);

        //             for (int j = 0; j < knn_idx.size(); j++)
        //             {
        //                 const PointXYZI &point_sel_tmp = refEdgeMap->points[knn_idx[j]];
        //                 centroid.x() += point_sel_tmp.x;
        //                 centroid.y() += point_sel_tmp.y;
        //                 centroid.z() += point_sel_tmp.z;
        //             }
        //             centroid /= 5.0;

        //             Eigen::Matrix3f mat_a;
        //             mat_a.setZero();

        //             for (int j = 0; j < knn_idx.size(); j++)
        //             {
        //                 const PointXYZI &point_sel_tmp = refEdgeMap->points[knn_idx[j]];
        //                 Vector3d a;
        //                 a.x() = point_sel_tmp.x - centroid.x();
        //                 a.y() = point_sel_tmp.y - centroid.y();
        //                 a.z() = point_sel_tmp.z - centroid.z();

        //                 // sum of a*a^T, but since SelfAdjointEigenSolver below only uses
        //                 // the lower triangular part of the matrix, we only calculate the
        //                 // entries on this lower triangular part.
        //                 mat_a(0, 0) += a.x() * a.x();
        //                 mat_a(1, 0) += a.x() * a.y();
        //                 mat_a(2, 0) += a.x() * a.z();
        //                 mat_a(1, 1) += a.y() * a.y();
        //                 mat_a(2, 1) += a.y() * a.z();
        //                 mat_a(2, 2) += a.z() * a.z();
        //             }
        //             mat_A1 = mat_a / 5.0;

        //             Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> esolver(mat_A1);
        //             mat_D1 = esolver.eigenvalues().real();
        //             mat_V1 = esolver.eigenvectors().real();

        //             if (mat_D1(0, 2) > 9 * mat_D1(0, 1))
        //             {
        //                 Vector3f X0(pivEdgePoint.x, pivEdgePoint.y, pivEdgePoint.z);
        //                 Vector3f X1(centroid.x() + 0.1 * mat_V1(0, 2),
        //                             centroid.y() + 0.1 * mat_V1(1, 2),
        //                             centroid.z() + 0.1 * mat_V1(2, 2));
        //                 Vector3f X2(centroid.x() - 0.1 * mat_V1(0, 2),
        //                             centroid.y() - 0.1 * mat_V1(1, 2),
        //                             centroid.z() - 0.1 * mat_V1(2, 2));

        //                 Vector3f X12 = X1 - X2;
        //                 Vector3f X02 = X0 - X2;

        //                 Vector3f X10xX02 = (X0 - X1).cross(X02);

        //                 Vector3f n1 = (X12.cross(X10xX02)).normalized();
        //                 Vector3f n2 = X12.cross(n1);

        //                 // float X10xX02_norm = X10xX02.norm();
        //                 // float X12_norm = X12.norm();

        //                 float la = n1.x();
        //                 float lb = n1.y();
        //                 float lc = n1.z();

        //                 float ld2 = X10xX02.norm() / X12.norm();

        //                 PointXYZI point_proj = pivEdgePoint;
        //                 point_proj.x -= la * ld2;
        //                 point_proj.y -= lb * ld2;
        //                 point_proj.z -= lc * ld2;

        //                 // printf("pprj: %f, %f, %f. point_proj: %f, %f, %f\n",
        //                 //         pproj.x(), pproj.y(), pproj.z(),
        //                 //         point_proj.x, point_proj.y, point_proj.z);

        //                 float ld_p1 = -(n1.x() * point_proj.x + n1.y() * point_proj.y + n1.z() * point_proj.z);
        //                 float ld_p2 = -(n2.x() * point_proj.x + n2.y() * point_proj.y + n2.z() * point_proj.z);

        //                 // printf("ld_p1, ld_p2: %f, %f\n", ld_p1, ld_p2);

        //                 float s = 1 - 0.9f * fabs(ld2);

        //                 if (s > 0.1)
        //                 {
        //                     srcEdgeCoeff.push_back(Vector4d(0.5 * s * la, 0.5 * s * lb, 0.5 * s * lc, 0.5 * s * ld_p1));
        //                     srcEdgeCoord.push_back(srcEdgePoint_);
        //                     totalEdgeF2M++;

        //                     srcEdgeCoeff.push_back(Vector4d(0.5 * s * n2.x(), 0.5 * s * n2.y(), 0.5 * s * n2.z(), 0.5 * s * ld_p2));
        //                     srcEdgeCoord.push_back(srcEdgePoint_);
        //                     totalEdgeF2M++;
        //                 }
        //             }
        //         }
        //     }
        // }

        time_edge_assoc = tt_edge_assoc.Toc();
    }

};

#endif CLOUDMATCHER_HPP