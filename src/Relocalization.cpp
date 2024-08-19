/**
* This file is part of splio.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/brytsknguyen/splio>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* splio is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* splio is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with splio.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 01/08/22.
//

#include "splio/FeatureCloud.h"

#include "STDesc.h"
#include "utility.h"

#include <ceres/ceres.h>
#include "PoseLocalParameterization.h"

// Shorthands
typedef sensor_msgs::PointCloud2::ConstPtr rosCloudMsgPtr;
typedef sensor_msgs::PointCloud2 rosCloudMsg;
typedef Sophus::SO3d SO3d;

// Define the pose factor for this module
class PoseFactorAnalytic : public ceres::SizedCostFunction<6, 7>
{
public:
    PoseFactorAnalytic() = delete;
    PoseFactorAnalytic(const Vector3d &pbar_, const Quaternd &qbar_, double wp_ = 1.0, double wq_ = 1.0)
    : pbar(pbar_), qbar(qbar_), wp(wp_), wq(wq_)
    {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaternd Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::AngleAxis<double> Phir(qbar.inverse() * Qi);

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(0, 0) = wp*(Pi - pbar);
        residual.block<3, 1>(3, 0) = wq*Phir.axis()*Phir.angle();
        
        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) = Vector3d(wp, wp, wp).asDiagonal();
                jacobian_pose_i.block<3, 3>(3, 0) = wq *Util::SO3JrightInv(Phir);
            }
        }

        return true;
    }
    // void Check(double **parameters);

private:
    Vector3d pbar;
    Quaternd qbar;

    double wp;
    double wq;
};

// Groupings for next calculations
struct RelocStat
{
    int keyframe_id = -1;
    double t = -1;
    myTf<double> tf = myTf<double>();
    Vector3d p_var = Vector3d(0, 0, 0);
    Vector3d r_var = Vector3d(0, 0, 0);
    double PDM = -1; // positional mahalanobis distance
    double RDM = -1; // rotational mahalanobis distance
    double ceres_cost = -1;
    bool inlier = false;
};

class Relocalization
{

private:
        
    // Node handler
    ros::NodeHandlePtr nh_ptr;

    // Subcriber of lidar pointcloud
    ros::Subscriber lidarCloudSub;

    deque<PointPose> relocPoseBuf;
    mutex relocPoseBufMtx;

    int kf_merged = 1;
    int reloc_thres  = 10;
    double reloc_timespan = 10.0;
    double pose_percentile_thres = 0.5;

    ConfigSetting config_setting;
    STDescManager *std_manager;

    ros::Publisher relocPub;

    bool go_to_sleep = false;

public:

    // Destructor
    ~Relocalization() {}

    Relocalization(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        // Initialize the variables and subsribe/advertise topics here
        Initialize();
    }

    void Initialize()
    {

        /* #region Lidar --------------------------------------------------------------------------------------------*/
        
        // Subcribe to the lidar keyframes
        lidarCloudSub = nh_ptr->subscribe("/kfcloud_std", 100, &Relocalization::PCHandler, this);

        // Avertise the reloc pose topic
        relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/reloc_pose", 100);

        /* #endregion Lidar -----------------------------------------------------------------------------------------*/

        nh_ptr->param<int>   ("/relocalization/reloc_thres",    reloc_thres,    10);
        nh_ptr->param<double>("/relocalization/reloc_timespan", reloc_timespan, 10.0);
        nh_ptr->param<double>("/relocalization/pose_percentile_thres", pose_percentile_thres, 0.5);
        nh_ptr->param<int>   ("/relocalization/kf_merged", kf_merged, 1);

        /* #region Prior Map Database -------------------------------------------------------------------------------*/
        
        // Get the config
        read_parameters(*nh_ptr, config_setting);

        // Load the descriptor
        std::string descriptor_path = "";
        nh_ptr->param<std::string>("relocalization/descriptor_file_path", descriptor_path, "");

        cout << "Descriptor database: " << descriptor_path << endl;

        std_manager = new STDescManager(config_setting);
        std_manager->loadExistingSTD(descriptor_path);

        /* #endregion Prior Map Database ----------------------------------------------------------------------------*/

    }

    void PCHandler(const splio::FeatureCloud::ConstPtr &msg)
    {
        if(go_to_sleep)
            return;

        // Stash the pointclouds
        static deque<splio::FeatureCloud::ConstPtr> msg_stash;
        msg_stash.push_back(msg);

        CloudXYZIPtr lastKfCloud;
        CloudPosePtr kfPoseCloud;

        if (msg_stash.size() >= kf_merged)
        {   
            lastKfCloud = CloudXYZIPtr(new CloudXYZI());
            for(auto &msg : msg_stash)
            {
                CloudXYZI lastKfCloud_;
                pcl::fromROSMsg(msg->extracted_cloud, lastKfCloud_);
                *lastKfCloud += lastKfCloud_;
            }

            kfPoseCloud = CloudPosePtr(new CloudPose());
            pcl::fromROSMsg(msg_stash.back()->edge_cloud, *kfPoseCloud);

            msg_stash.clear();
        }
        else
        {
            printf("Clouds stashed: %d\n", msg_stash.size());
            return;
        }

        int keyframe_id = (int)(kfPoseCloud->back().intensity);        

        TicToc tt_relocalize;

        // Downsample the pointcloud
        down_sampling_voxel(*lastKfCloud, config_setting.ds_size_);

        // Find the cloud descriptor
        std::vector<STDesc> stds_vec;
        std_manager->GenerateSTDescsOneTime(lastKfCloud, stds_vec);

        // Search for match with previous scan
        std::pair<int, double> search_result(-1, 0);
        std::pair<Vector3d, Matrix3d> loop_transform = make_pair(Vector3d(0, 0, 0), Matrix3d::Identity());
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
        std_manager->SearchLoop(stds_vec, search_result, loop_transform, loop_std_pair);

        bool relocalized = search_result.first > 0;
        double score = search_result.second;
        myTf tf_Lprior_L0_est;

        if (relocalized)
        {
            // Compute Pose Estimation Error
            int match_frame = search_result.first;
            std_manager->PlaneGeomrtricIcp(std_manager->current_plane_cloud_,
                                           std_manager->plane_cloud_vec_[match_frame], loop_transform);

            tf_Lprior_L0_est = myTf(loop_transform.second, loop_transform.first);

            // printf(KGRN "Relocalized keyframe: %4d -- %4d. Score: %.5f. Pose: \n" RESET,
            //              keyframe_id, search_result.first, search_result.second);

            // cout << tf_Lprior_L0_est << endl;

            std::lock_guard<mutex> lg(relocPoseBufMtx);
            relocPoseBuf.push_back(tf_Lprior_L0_est.Pose6D(msg->header.stamp.toSec()));
            relocPoseBuf.back().intensity = keyframe_id;
        }
        else
            printf("Keyframe cannot be relocalized. ID %d. Buf: %d\n", keyframe_id, relocPoseBuf.size());

        string report1 = myprintf("Received KF of %6d pts. Id: %04d. ProcTime: %9.3f. ", lastKfCloud->size(), keyframe_id, tt_relocalize.Toc());
        string report2;
        if (relocalized)
        {
            report2 = myprintf(KBLU "Relocalized! TF_Lprior_L0: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f. Score: %.3f\n" RESET,
                               tf_Lprior_L0_est.pos.x(), tf_Lprior_L0_est.pos.y(), tf_Lprior_L0_est.pos.z(),
                               tf_Lprior_L0_est.yaw(),   tf_Lprior_L0_est.pitch(), tf_Lprior_L0_est.roll(),
                               score);

            cout << report1 << report2;
        }
    }

    void Relocalize()
    {
        while(ros::ok())
        {
            if (go_to_sleep)
            {
                this_thread::sleep_for(chrono::milliseconds(1000));
                continue;
            }

            // Loop if there is no relocalization
            if (relocPoseBuf.empty())
            {
                this_thread::sleep_for(chrono::milliseconds(1000));
                continue;
            }

            // Check if the pointclouds accumulated span 10 seconds to do relocalization
            if ( (relocPoseBuf.size() > 1 && (relocPoseBuf.back().t - relocPoseBuf.front().t) < reloc_timespan)
                    || relocPoseBuf.size() < reloc_thres )
            {
                this_thread::sleep_for(chrono::milliseconds(1000));
                continue;
            }

            // Extract the reloc poses
            deque<PointPose> relocPose;
            {
                std::lock_guard<std::mutex> lg(relocPoseBufMtx);
                relocPose = relocPoseBuf;

                // Remove the poses before the middle time
                // double mid_time = 0.5*(relocPoseBuf.back().t + relocPoseBuf.front().t);
                // while(relocPoseBuf.front().t < mid_time)
                //     relocPoseBuf.pop_front();

                relocPoseBuf.clear();
            }

            int N = relocPose.size();
            deque<RelocStat> relocStat;

            // Convert the pose points to pose tfs
            for(PointPose &pose : relocPose)
            {
                RelocStat stat;
                stat.keyframe_id = (int)pose.intensity;
                stat.t  = pose.t;
                stat.tf = myTf(pose);
                relocStat.push_back(stat);
            }

            // Calculate the mean 
            Vector3d p_mean(0, 0, 0);
            Vector3d r_mean(0, 0, 0);
            for(RelocStat &stat : relocStat)
            {
                p_mean += stat.tf.pos;
                r_mean += stat.tf.SO3Log();
            }

            p_mean /= N;
            r_mean /= N;
            Quaternd q_mean(Eigen::AngleAxis<double>(r_mean.norm(), r_mean/r_mean.norm()));

            OptimizePoseWithCeres(p_mean, q_mean, relocStat);

            // // Calulate the variance
            // Matrix3d Cp = Matrix3d::Zero();
            // Matrix3d Cr = Matrix3d::Zero();
            // for(RelocStat &stat : relocStat)
            // {
            //     stat.p_var = stat.tf.pos - p_mean;
            //     stat.r_var = stat.tf.SO3Log() - r_mean;

            //     Cp += stat.p_var*stat.p_var.transpose();
            //     Cr += stat.r_var*stat.r_var.transpose();
            // }

            // Cp /= N;
            // Cr /= N;

            // // Calculate the Mahalanobis distance of the reloc poses
            // for(RelocStat &stat : relocStat)
            // {
            //     stat.PDM = sqrt((stat.p_var.transpose()*Cp.inverse()*stat.p_var)[0]);
            //     stat.RDM = sqrt((stat.r_var.transpose()*Cr.inverse()*stat.r_var)[0]);
            // }

            // Sort the reloc by cost
            struct compareRes
            {
                bool const operator()(RelocStat a, RelocStat b) const
                {
                    return (a.ceres_cost < b.ceres_cost);
                }
            };
            std::sort(relocStat.begin(), relocStat.end(), compareRes());
            double cost_thres = relocStat[(int)(N*pose_percentile_thres)].ceres_cost;

            // Find the inliear relocalized poses
            deque<RelocStat> relocStatInlier;
            Vector3d p_inlr(0, 0, 0);
            Vector3d r_inlr(0, 0, 0);
            int Ninlr = 0;
            for(RelocStat &stat : relocStat)
            {
                if (stat.ceres_cost < cost_thres)
                {
                    Eigen::AngleAxis<double>e(stat.tf.rot);

                    p_inlr += stat.tf.pos;
                    r_inlr += e.angle()*e.axis();
                    Ninlr  += 1;

                    relocStatInlier.push_back(stat);
                }
            }
            p_inlr /= Ninlr;
            r_inlr /= Ninlr;
            Quaternd q_inlr(Eigen::AngleAxis<double>(r_inlr.norm(), r_inlr/r_inlr.norm()));
            
            mytf tf_Lprior_L0_opt = OptimizePoseWithCeres(p_inlr, q_inlr, relocStatInlier);
            
            // Display the stat
            int idx = -1;
            for(RelocStat &stat : relocStat)
            {
                idx++;
                printf("KF #%2d. ID %03d. TF_Lprior_L0_est: %9.3f, %9.3f, %9.3f | %9.3f, %9.3f, %9.3f. "
                    //    "PDM: %9.3f. RDM: %9.3f. "
                       "%s. CCost: %9.3f / %9.3f\n",
                        idx,
                        stat.keyframe_id,
                        stat.tf.pos(0), stat.tf.pos(1),  stat.tf.pos(2),
                        stat.tf.yaw(),  stat.tf.pitch(), stat.tf.roll(),
                        // stat.PDM, stat.RDM,
                       (stat.ceres_cost < cost_thres) ? KGRN "INLIER " RESET : KRED "OUTLIER" RESET,
                        stat.ceres_cost, cost_thres
                      );
            }

            printf("Mean inlier pose: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f.\n",
                    tf_Lprior_L0_opt.pos.x(), tf_Lprior_L0_opt.pos.y(), tf_Lprior_L0_opt.pos.z(),
                    tf_Lprior_L0_opt.yaw(),   tf_Lprior_L0_opt.pitch(), tf_Lprior_L0_opt.roll());

            // Publish the pose            
            geometry_msgs::PoseStamped poseMsg;
            poseMsg.header.stamp = ros::Time(relocStat.back().t);
            poseMsg.header.frame_id = "priormap";
            poseMsg.pose.position.x = tf_Lprior_L0_opt.pos(0);
            poseMsg.pose.position.y = tf_Lprior_L0_opt.pos(1);
            poseMsg.pose.position.z = tf_Lprior_L0_opt.pos(2);
            poseMsg.pose.orientation.x = tf_Lprior_L0_opt.rot.x();
            poseMsg.pose.orientation.y = tf_Lprior_L0_opt.rot.y();
            poseMsg.pose.orientation.z = tf_Lprior_L0_opt.rot.z();
            poseMsg.pose.orientation.w = tf_Lprior_L0_opt.rot.w();
            relocPub.publish(poseMsg);

            // Go to sleep until there is request to redo reloc
            go_to_sleep = true;
        }
    }

    mytf OptimizePoseWithCeres(Vector3d p_init, Quaternd q_init, deque<RelocStat> &relocStat)
    {
        // Create and solve the Ceres Problem
        ceres::Problem problem;
        ceres::Solver::Options options;

        // Set up the options
        // options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
        options.max_num_iterations           = 40;
        options.max_solver_time_in_seconds   = 0.5;
        options.num_threads                  = MAX_THREADS;
        options.minimizer_progress_to_stdout = false;

        // Create optimization params
        double* PARAM_POSE = new double[7];
        
        PARAM_POSE[0] = p_init(0);
        PARAM_POSE[1] = p_init(1);
        PARAM_POSE[2] = p_init(2);

        PARAM_POSE[3] = q_init.x();
        PARAM_POSE[4] = q_init.y();
        PARAM_POSE[5] = q_init.z();
        PARAM_POSE[6] = q_init.w();

        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(PARAM_POSE, 7, local_parameterization);

        vector<ceres::internal::ResidualBlock *> res_ids_pose;
        double cost_pose_init = -1, cost_pose_final = -1;
        for(RelocStat &stat : relocStat)
        {
            ceres::LossFunction *loss_function = new ceres::ArctanLoss(1.0);
            PoseFactorAnalytic *f = new PoseFactorAnalytic(stat.tf.pos, stat.tf.rot, 1.0, 1.0);
            ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(f, loss_function, PARAM_POSE);
            res_ids_pose.push_back(res_id);
        }

        Util::ComputeCeresCost(res_ids_pose, cost_pose_init, problem);

        TicToc tt_solve;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        tt_solve.Toc();

        Util::ComputeCeresCost(res_ids_pose, cost_pose_final, problem);

        Vector3d pos_est(PARAM_POSE[0], PARAM_POSE[1], PARAM_POSE[2]);
        Quaternd rot_est(PARAM_POSE[6], PARAM_POSE[3], PARAM_POSE[4], PARAM_POSE[5]);
        myTf tf_est(rot_est, pos_est);
        myTf tf_ini(q_init, p_init);

        printf("Tslv: %.3f. Iter: %d. J0: %6.3f -> JK: %6.3f. \n"
                "PoseEstStart: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f.\n"
                "PoseEstFinal: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f.\n",
                tt_solve.Toc(), summary.iterations.size(),
                cost_pose_init, cost_pose_final,
                tf_ini.pos(0), tf_ini.pos(1), tf_ini.pos(2), tf_ini.yaw(), tf_ini.pitch(), tf_ini.roll(),
                tf_est.pos(0), tf_est.pos(1), tf_est.pos(2), tf_est.yaw(), tf_est.pitch(), tf_est.roll());

        for(int i = 0; i < relocStat.size(); i++)
        {
            vector<ceres::internal::ResidualBlock *> res(1, res_ids_pose[i]);
            double cost;
            Util::ComputeCeresCost(res, cost, problem);
            relocStat[i].ceres_cost = cost;
        }

        return tf_est;
    }

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "relocalization");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Relocalization Started." RESET);

    Relocalization relocalization(nh_ptr);

    thread relocalize(&Relocalization::Relocalize, &relocalization);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    
    return 0;
}
