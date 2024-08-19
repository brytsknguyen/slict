/**
 * This file is part of slict.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot
 * sg>, School of EEE Nanyang Technological Univertsity, Singapore
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

#include <filesystem>
// #include <boost/format.hpp>
#include <condition_variable>
#include <deque>
#include <thread>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ceres/ceres.h>
#include <cv_bridge/cv_bridge.h>

/* All needed for kdtree of custom point type----------*/
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
/* All needed for kdtree of custom point type----------*/

/* All needed for filter of custom point type----------*/
#include <pcl/pcl_base.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/filters/filter.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/impl/uniform_sampling.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/impl/crop_box.hpp>
/* All needed for filter of custom point type----------*/

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// ROS
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/Imu.h"
#include <tf2_ros/static_transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include "tf/transform_broadcaster.h"
#include "slict/globalMapsPublish.h"

// UFO
#include <ufo/map/code/code_unordered_map.h>
#include <ufo/map/point_cloud.h>
#include <ufo/map/surfel_map.h>
#include <ufomap_msgs/UFOMapStamped.h>
#include <ufomap_msgs/conversions.h>
#include <ufomap_ros/conversions.h>

// Factor
#include "PoseLocalParameterization.h"
// #include "PreintBase.h"
// #include "factor/PreintFactor.h"
// #include "factor/PointToPlaneDisFactorCT.h"
#include "factor/RelOdomFactor.h"
// #include "factor/PoseFactorCT.h"
#include "factor/PoseAnalyticFactor.h"
#include "factor/PointToPlaneAnalyticFactor.hpp"
#include "factor/GyroAcceBiasAnalyticFactor.h"
#include "factor/VelocityAnalyticFactor.h"

// Custom for package
#include "utility.h"
#include "slict/FeatureCloud.h"
#include "slict/OptStat.h"
#include "slict/TimeLog.h"
#include "CloudMatcher.hpp"

// Add ikdtree
#include <ikdTree/ikd_Tree.h>

// myGN solver
#include "mySolver.h"
#include "factor/GyroAcceBiasFactorTMN.hpp"
#include "factor/Point2PlaneFactorTMN.hpp"

#include "PointToMapAssoc.h"

// #include "MapManager.hpp"

// #define SPLINE_N 4

using namespace std;
using namespace Eigen;
using namespace pcl;
using namespace basalt;

// Shorthands for ufomap
namespace ufopred     = ufo::map::predicate;
using ufoSurfelMap    = ufo::map::SurfelMap;
using ufoSurfelMapPtr = boost::shared_ptr<ufoSurfelMap>;
using ufoNode         = ufo::map::NodeBV;
using ufoSphere       = ufo::geometry::Sphere;
using ufoPoint3       = ufo::map::Point3;
//Create a prototype for predicates
auto PredProto = ufopred::HasSurfel()
              && ufopred::DepthMin(1)
              && ufopred::DepthMax(1)
              && ufopred::NumSurfelPointsMin(1)
              && ufopred::SurfelPlanarityMin(0.2);
// Declare the type of the predicate as a new type
typedef decltype(PredProto) PredType;

typedef Sophus::SE3d SE3d;

class Estimator
{

private:

    // Node handler
    ros::NodeHandlePtr nh_ptr;
    ros::Time program_start_time;
    
    // The coordinate frame at the initial position of the slam
    string slam_ref_frame = "world";
    // The coordinate frame that states on the sliding window is using
    string current_ref_frame = slam_ref_frame;

    bool autoexit = false;
    bool show_report = true;

    // Subscribers
    ros::Subscriber data_sub;

    // Service
    ros::ServiceServer global_maps_srv;       // For requesting the global map to be published

    // Synchronized data buffer
    mutex packet_buf_mtx;
    deque<slict::FeatureCloud::ConstPtr> packet_buf;

    bool ALL_INITED  = false;
    int  WINDOW_SIZE = 4;
    int  N_SUB_SEG   = 4;

    Vector3d GRAV = Vector3d(0, 0, 9.82);

    // Spline representing the trajectory
    // using PoseSpline = basalt::Se3Spline<SPLINE_N>;
    using PoseSplinePtr = std::shared_ptr<PoseSplineX>;
    PoseSplinePtr GlobalTraj = nullptr;
    
    int    SPLINE_N       = 4;
    double deltaT         = 0.05;
    double start_fix_span = 0.05;
    double final_fix_span = 0.05;

    int reassociate_steps = 0;
    int reassoc_rate = 3;
    vector<int> deskew_method = {0, 0};

    bool use_ceres = true;

    // Sliding window data (prefixed "Sw" )
    struct TimeSegment
    {
        TimeSegment(double start_time_, double final_time_)
            : start_time(start_time_), final_time(final_time_)
        {};

        double dt()
        {
            return (final_time - start_time);
        }

        double start_time;
        double final_time;
    };
    deque<deque<TimeSegment>> SwTimeStep;
    deque<CloudXYZITPtr>      SwCloud;
    deque<CloudXYZIPtr>       SwCloudDsk;
    deque<CloudXYZIPtr>       SwCloudDskDS;
    deque<vector<LidarCoef>>  SwLidarCoef;
    deque<map<int, int>>      SwDepVsAssoc;
    deque<deque<ImuSequence>> SwImuBundle;      // ImuSample defined in utility.h
    deque<deque<ImuProp>>     SwPropState;

    // Check list for adjusting the computation
    map<int, int> DVA;
    int total_lidar_coef;

    // States at the segments' start and final times; ss: segment's start time, sf: segment's final time
    deque<deque<Quaternd>> ssQua, sfQua;
    deque<deque<Vector3d>> ssPos, sfPos;
    deque<deque<Vector3d>> ssVel, sfVel;
    deque<deque<Vector3d>> ssBia, sfBia;
    deque<deque<Vector3d>> ssBig, sfBig;

    // IMU weight
    double GYR_N = 10;
    double GYR_W = 10;
    double ACC_N = 0.5;
    double ACC_W = 10;

    double ACC_SCALE = 1.0;

    // Velocity weight
    double POSE_N = 5;

    // Velocity weight
    double VEL_N = 10;

    // Lidar weight
    double lidar_weight = 10;

    Vector3d BG_BOUND = Vector3d(0.1, 0.1, 0.1);
    Vector3d BA_BOUND = Vector3d(0.1, 0.1, 0.2);

    int last_fixed_knot = 0;
    int first_fixed_knot = 0;

    double leaf_size = 0.1;
    double assoc_spacing = 0.2;
    int surfel_map_depth = 5;
    int surfel_min_point = 5;
    int surfel_min_depth = 0;
    int surfel_query_depth = 3;
    double surfel_intsect_rad = 0.5;
    double surfel_min_plnrty = 0.8;

    PredType *commonPred;

    // Size of k-nearest neighbourhood for the knn search
    double dis_to_surfel_max = 0.05;
    double score_min = 0.1;
    
    // Lidar downsample rate
    int lidar_ds_rate = 1;
    int sweep_len = 1;

    // Optimization parameters
    double lidar_loss_thres = 1.0;
    double imu_loss_thres = -1.0;

    // Keeping track of preparation before solving
    TicToc tt_preopt;
    TicToc tt_fitspline;
    double t_slv_budget;

    // Solver config
    ceres::LinearSolverType linSolver;
    ceres::TrustRegionStrategyType trustRegType;     // LEVENBERG_MARQUARDT, DOGLEG
    ceres::DenseLinearAlgebraLibraryType linAlgbLib; // EIGEN, LAPACK, CUDA
    double max_solve_time = 0.5;
    int max_iterations = 200;
    bool ensure_real_time = true;
    bool find_factor_cost = false;
    bool fit_spline = false;

    // Sensors used
    bool fuse_lidar      = true;
    bool fuse_imu        = true;
    bool fuse_poseprop   = true;
    bool fuse_velprop    = true;
    
    bool snap_to_0180    = false;
    bool regularize_imu  = true;
    bool lite_redeskew   = false;
    int  fix_mode        = 1;
    double imu_init_time = 0.1;
    int max_outer_iters  = 1;
    double dj_thres      = 0.1;
    int max_lidar_factor = 5000;

    // Map
    CloudPosePtr        KfCloudPose;
    deque<CloudXYZIPtr> KfCloudinB;
    deque<CloudXYZIPtr> KfCloudinW;

    bool refine_kf = false;

    int    ufomap_version = 0;
    mutex  global_map_mtx;
    CloudXYZIPtr globalMap;

    TicToc tt_margcloud;
    TicToc tt_ufoupdate;

    mutex map_mtx;
    ufoSurfelMapPtr activeSurfelMap;
    ikdtreePtr activeikdtMap;

    mutex mapqueue_mtx;
    deque<CloudXYZIPtr> mapqueue;
    thread thread_update_map;

    ufoSurfelMapPtr priorSurfelMapPtr;
    ufoSurfelMap priorSurfelMap;

    ikdtreePtr priorikdtMapPtr;
    // ikdtree priorikdtMap;

    enum RelocStat {NOT_RELOCALIZED, RELOCALIZING, RELOCALIZED};

    RelocStat reloc_stat = NOT_RELOCALIZED;
    mutex relocBufMtx;
    deque<myTf<double>> relocBuf;
    ros::Subscriber relocSub;
    mytf tf_Lprior_L0;
    // mytf tf_Lprior_L0_init;         // For debugging and development only
    bool refine_reloc_tf = false;
    int  ioa_max_iter = 20;
    bool marginalize_new_points = false;

    thread reloc_init;

    // Loop closure
    bool loop_en = true;
    int loop_kf_nbr = 5;            // Number of neighbours to check for loop closure
    int loop_time_mindiff = 10;     // Only check for loop when keyframes have this much difference
    struct LoopPrior
    {
        LoopPrior(int prevPoseId_, int currPoseId_, double JKavr_, double IcpFn_, mytf tf_Bp_Bc_)
            : prevPoseId(prevPoseId_), currPoseId(currPoseId_), JKavr(JKavr_), IcpFn(IcpFn_), tf_Bp_Bc(tf_Bp_Bc_) {};

        int prevPoseId = -1;
        int currPoseId = -1;
        double JKavr = -1;
        double IcpFn = -1;
        mytf tf_Bp_Bc;
    };
    deque<LoopPrior> loopPairs;     // Array to store loop priors

    int icpMaxIter = 20;            // Maximum iterations for ICP
    double icpFitnessThres = 0.3;   // Fitness threshold for ICP check
    double histDis = 15.0;          // Maximum correspondence distance for icp
    double lastICPFn = -1;

    int rib_edge = 5;
    double odom_q_noise = 0.1;
    double odom_p_noise = 0.1;
    double loop_weight  = 0.02;

    TicToc tt_loopBA;               // Timer to check the loop and BA time
    struct BAReport
    {
        int turn = -1;
        double pgopt_time = 0;
        int pgopt_iter = 0;
        int factor_relpose = 0;
        int factor_loop = 0;
        double J0 = 0;
        double JK = 0;
        double J0_relpose = 0;
        double JK_relpose = 0;
        double J0_loop = 0;
        double JK_loop = 0;
        double rebuildmap_time = 0;
    };
    BAReport baReport;

    struct KeyframeCand
    {
        KeyframeCand(double start_time_, double end_time_, CloudXYZIPtr kfCloud_)
            : start_time(start_time_), end_time(end_time_), kfCloud(kfCloud_) {};
        double start_time;
        double end_time;
        CloudXYZIPtr kfCloud;
    };
    
    // Publisher for global map.
    ros::Publisher global_map_pub;
    bool publish_map = false;

    // Keyframe params
    double kf_min_dis = 0.5;
    double kf_min_angle = 10;
    double margPerc = 0;
    
    // Publisher for latest keyframe
    ros::Publisher kfcloud_pub;
    ros::Publisher kfcloud_std_pub;
    ros::Publisher kfpose_pub;

    // Log
    string log_dir = "/home/tmn";
    string log_dir_kf;
    std::ofstream loop_log_file;

    // PriorMap
    bool use_prior_map = false;

    std::thread initPriorMapThread;

    // For visualization
    CloudXYZIPtr  pmSurfGlobal;
    CloudXYZIPtr  pmEdgeGlobal;

    CloudXYZIPtr  priorMap;
    
    deque<CloudXYZIPtr> pmFull;
    CloudPosePtr pmPose;

    bool pmLoaded = false;

    // KdTreeFLANN<PointXYZI>::Ptr kdTreePriorMap;

    ros::Publisher priorMapPub;
    ros::Timer     pmVizTimer;

    double priormap_viz_res = 0.2;
    
    // ufoSurfelMap surfelMapSurf;
    // ufoSurfelMap surfelMapEdge;

    // Use ufomap or ikdtree;
    bool use_ufm = false;
    
public:
    // Destructor
    ~Estimator() {}

    Estimator(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {   
        // Normal Processes
        Initialize();

        // Inialize Prior map
        InitializePriorMap();

    }

    void Initialize()
    {

        program_start_time = ros::Time::now();

        autoexit = GetBoolParam("/autoexit", false);

        // Disable report
        show_report = GetBoolParam("/show_report", true);

        // Get the coordinate frame of choice
        nh_ptr->getParam("/slam_ref_frame", slam_ref_frame);
        current_ref_frame = slam_ref_frame;

        // Maximum number of threads
        printf("Maximum number of threads: %d\n", MAX_THREADS);

        // Window size length
        if (nh_ptr->getParam("/WINDOW_SIZE", WINDOW_SIZE))
            printf("WINDOW_SIZE declared: %d\n", WINDOW_SIZE);
        else
        {
            printf("WINDOW_SIZE not found. Exiting\n");
            exit(-1);
        }

        if (nh_ptr->getParam("/N_SUB_SEG", N_SUB_SEG))
            printf("N_SUB_SEG declared: %d\n", N_SUB_SEG);
        else
        {
            printf("N_SUB_SEG not found. Exiting\n");
            exit(-1);
        }

        nh_ptr->param("/SPLINE_N", SPLINE_N, 4);
        nh_ptr->param("/deltaT", deltaT, 0.05);
        nh_ptr->param("/start_fix_span", start_fix_span, 0.05);
        nh_ptr->param("/final_fix_span", final_fix_span, 0.05);
        nh_ptr->param("/reassociate_steps", reassociate_steps, 0);
        nh_ptr->param("/deskew_method", deskew_method, deskew_method);
        nh_ptr->param("/reassoc_rate", reassoc_rate, 3);

        use_ceres = GetBoolParam("/use_ceres", false);

        // Initialize the states in the sliding window
        ssQua = sfQua = deque<deque<Quaternd>>(WINDOW_SIZE, deque<Quaternd>(N_SUB_SEG, Quaternd::Identity()));
        ssPos = sfPos = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));
        ssVel = sfVel = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));
        ssBia = sfBia = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));
        ssBig = sfBig = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));

        // Gravity constant
        double GRAV_ = 9.82;
        nh_ptr->param("/GRAV", GRAV_, 9.82);
        GRAV = Vector3d(0, 0, GRAV_);
        printf("GRAV constant: %f\n", GRAV_);

        nh_ptr->getParam("/GYR_N", GYR_N);
        nh_ptr->getParam("/GYR_W", GYR_W);
        nh_ptr->getParam("/ACC_N", ACC_N);
        nh_ptr->getParam("/ACC_W", ACC_W);

        printf("Gyro variance: %f\n", GYR_N);
        printf("Bgyr variance: %f\n", GYR_W);
        printf("Acce variance: %f\n", ACC_N);
        printf("Bacc variance: %f\n", ACC_W);

        nh_ptr->getParam("/POSE_N", POSE_N);
        printf("Position prior variance: %f\n", POSE_N);

        nh_ptr->getParam("/VEL_N", VEL_N);
        printf("Velocity variance: %f\n", VEL_N);

        // Sensor weightage
        nh_ptr->getParam("/lidar_weight", lidar_weight);

        // Bias bounds
        vector<double> BG_BOUND_ = {0.1, 0.1, 0.1};
        vector<double> BA_BOUND_ = {0.1, 0.1, 0.2};
        nh_ptr->getParam("/BG_BOUND", BG_BOUND_);
        nh_ptr->getParam("/BA_BOUND", BA_BOUND_);
        BG_BOUND = Vector3d(BG_BOUND_[0], BG_BOUND_[1], BG_BOUND_[2]);
        BA_BOUND = Vector3d(BA_BOUND_[0], BA_BOUND_[1], BA_BOUND_[2]);

        // If use ufm for incremental map
        use_ufm = GetBoolParam("/use_ufm", false);

        // Downsample size
        nh_ptr->getParam("/leaf_size",          leaf_size);
        nh_ptr->getParam("/assoc_spacing",      assoc_spacing);
        nh_ptr->getParam("/surfel_map_depth",   surfel_map_depth);
        nh_ptr->getParam("/surfel_min_point",   surfel_min_point);
        nh_ptr->getParam("/surfel_min_depth",   surfel_min_depth);
        nh_ptr->getParam("/surfel_query_depth", surfel_query_depth);
        nh_ptr->getParam("/surfel_intsect_rad", surfel_intsect_rad);
        nh_ptr->getParam("/surfel_min_plnrty",  surfel_min_plnrty);

        printf("leaf_size:          %f\n", leaf_size);
        printf("assoc_spacing:      %f\n", assoc_spacing);
        printf("surfel_map_depth:   %d\n", surfel_map_depth);
        printf("surfel_min_point:   %d\n", surfel_min_point);
        printf("surfel_min_depth:   %d\n", surfel_min_depth);
        printf("surfel_query_depth: %d\n", surfel_query_depth);
        printf("surfel_intsect_rad: %f\n", surfel_intsect_rad);
        printf("surfel_min_plnrty:  %f\n", surfel_min_plnrty);

        commonPred = new PredType(ufopred::HasSurfel()
                               && ufopred::DepthMin(surfel_min_depth)
                               && ufopred::DepthMax(surfel_query_depth - 1)
                               && ufopred::NumSurfelPointsMin(surfel_min_point)
                               && ufopred::SurfelPlanarityMin(surfel_min_plnrty));

        // Number of neigbours to check for in association
        nh_ptr->getParam("/dis_to_surfel_max", dis_to_surfel_max);
        nh_ptr->getParam("/score_min", score_min);
        // Lidar feature downsample rate
        // nh_ptr->getParam("/ds_rate", ds_rate);
        // Lidar sweep len by number of scans merged
        nh_ptr->getParam("/sweep_len", sweep_len);

        // Keyframe params
        nh_ptr->getParam("/kf_min_dis", kf_min_dis);
        nh_ptr->getParam("/kf_min_angle", kf_min_angle);
        // Refine keyframe with ICP
        refine_kf = GetBoolParam("/refine_kf", false);

        // Optimization parameters
        nh_ptr->getParam("/lidar_loss_thres", lidar_loss_thres);

        // Solver
        string linSolver_;
        nh_ptr->param("/linSolver", linSolver_, string("dqr"));
        if (linSolver_ == "dqr")
            linSolver = ceres::DENSE_QR;
        else if( linSolver_ == "dnc")
            linSolver = ceres::DENSE_NORMAL_CHOLESKY;
        else if( linSolver_ == "snc")
            linSolver = ceres::SPARSE_NORMAL_CHOLESKY;
        else if( linSolver_ == "cgnr")
            linSolver = ceres::CGNR;
        else if( linSolver_ == "dschur")
            linSolver = ceres::DENSE_SCHUR;
        else if( linSolver_ == "sschur")
            linSolver = ceres::SPARSE_SCHUR;
        else if( linSolver_ == "ischur")
            linSolver = ceres::ITERATIVE_SCHUR;
        else
            linSolver = ceres::SPARSE_NORMAL_CHOLESKY;
        printf(KYEL "/linSolver: %d. %s\n" RESET, linSolver, linSolver_.c_str());

        string trustRegType_;
        nh_ptr->param("/trustRegType", trustRegType_, string("lm"));
        if (trustRegType_ == "lm")
            trustRegType = ceres::LEVENBERG_MARQUARDT;
        else if( trustRegType_ == "dogleg")
            trustRegType = ceres::DOGLEG;
        else
            trustRegType = ceres::LEVENBERG_MARQUARDT;
        printf(KYEL "/trustRegType: %d. %s\n" RESET, trustRegType, trustRegType_.c_str());

        string linAlgbLib_;
        nh_ptr->param("/linAlgbLib", linAlgbLib_, string("cuda"));
        if (linAlgbLib_ == "eigen")
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::EIGEN;
        else if(linAlgbLib_ == "lapack")
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::LAPACK;
        else if(linAlgbLib_ == "cuda")
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::CUDA;
        else
            linAlgbLib = ceres::DenseLinearAlgebraLibraryType::EIGEN;
        printf(KYEL "/linAlgbLib: %d. %s\n" RESET, linAlgbLib, linAlgbLib_.c_str());

        nh_ptr->param("/max_solve_time", max_solve_time,  0.5);
        nh_ptr->param("/max_iterations", max_iterations,  200);
        
        ensure_real_time = GetBoolParam("/ensure_real_time", true);
        find_factor_cost = GetBoolParam("/find_factor_cost", true);
        fit_spline       = GetBoolParam("/fit_spline", true);

        // Fusion option
        fuse_lidar     = GetBoolParam("/fuse_lidar",     true);
        fuse_imu       = GetBoolParam("/fuse_imu",       true);
        fuse_poseprop  = GetBoolParam("/fuse_poseprop",  true);
        fuse_velprop   = GetBoolParam("/fuse_velprop",   true);

        snap_to_0180   = GetBoolParam("/snap_to_0180",   false);
        regularize_imu = GetBoolParam("/regularize_imu", true);
        lite_redeskew  = GetBoolParam("/lite_redeskew",  false);

        nh_ptr->param("/fix_mode",         fix_mode,         1);
        nh_ptr->param("/imu_init_time",    imu_init_time,    0.1);
        nh_ptr->param("/max_outer_iters",  max_outer_iters,  1);
        nh_ptr->param("/max_lidar_factor", max_lidar_factor, 4000);
        nh_ptr->param("/dj_thres",         dj_thres,         0.1);

        printf("max_outer_iters: %d.\n"
               "dj_thres:        %f.\n" 
               "fix_mode:        %d.\n"
               "max_iterations:  %d.\n"
               "imu_init_time:   %f\n",
                max_outer_iters, dj_thres, fix_mode, max_iterations, imu_init_time);
        
        // Loop parameters
        loop_en = GetBoolParam("/loop_en", true);
        nh_ptr->param("/loop_kf_nbr", loop_kf_nbr, 5);
        nh_ptr->param("/loop_time_mindiff", loop_time_mindiff, 10);

        nh_ptr->param("/icpMaxIter", icpMaxIter, 20);
        nh_ptr->param("/icpFitnessThres", icpFitnessThres, 0.3);
        nh_ptr->param("/histDis", histDis, 15.0);

        nh_ptr->param("/rib_edge", rib_edge, 5);
        nh_ptr->param("/odom_q_noise", odom_q_noise, 0.1);
        nh_ptr->param("/odom_p_noise", odom_p_noise, 0.1);
        nh_ptr->param("/loop_weight", loop_weight, 0.1);
        
        // Map inertialization
        KfCloudPose   = CloudPosePtr(new CloudPose());

        // Create a handle to the global map
        activeSurfelMap = ufoSurfelMapPtr(new ufoSurfelMap(leaf_size, surfel_map_depth));
        // Create an ikdtree
        activeikdtMap = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));

        // For visualization
        globalMap = CloudXYZIPtr(new CloudXYZI());

        // Advertise the global map
        global_map_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/global_map", 10);

        publish_map = GetBoolParam("/publish_map", true);

        // Subscribe to the lidar-imu package
        data_sub = nh_ptr->subscribe("/sensors_sync", 100, &Estimator::DataHandler, this);

        // Advertise the outputs
        kfcloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/kfcloud", 10);
        kfpose_pub  = nh_ptr->advertise<sensor_msgs::PointCloud2>("/kfpose", 10);

        kfcloud_std_pub = nh_ptr->advertise<slict::FeatureCloud>("/kfcloud_std", 10);

        // Advertise the service
        global_maps_srv = nh_ptr->advertiseService("/global_maps_publish", &Estimator::PublishGlobalMaps, this);

        // Log file
        log_dir = nh_ptr->param("/log_dir", log_dir);
        log_dir_kf = log_dir + "/KFCloud/";
        std::filesystem::create_directories(log_dir);
        std::filesystem::create_directories(log_dir_kf);

        loop_log_file.open(log_dir + "/loop_log.csv");
        loop_log_file.precision(std::numeric_limits<double>::digits10 + 1);
        // loop_log_file.close();

        // Create a prior map manager
        // mapManager = new MapManager(nh_ptr);

        // Create a thread to update the map
        thread_update_map = thread(&Estimator::UpdateMap, this); ;
    }

    void InitializePriorMap()
    {
        TicToc tt_initprior;

        tf_Lprior_L0 = myTf(Quaternd(1, 0, 0, 0), Vector3d(0, 0, 0));

        use_prior_map = GetBoolParam("/use_prior_map", false);
        printf("use_prior_map:   %d\n", use_prior_map);

        if (!use_prior_map)
            return;

        // Check if initial pose is available
        if(ros::param::has("/tf_Lprior_L0_init"))
        {
            vector<double> tf_Lprior_L0_init_;
            nh_ptr->param("/tf_Lprior_L0_init", tf_Lprior_L0_init_, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
            printf(KYEL "tf_Lprior_L0_init: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n" RESET,
                         tf_Lprior_L0_init_[0], tf_Lprior_L0_init_[1], tf_Lprior_L0_init_[2],
                         tf_Lprior_L0_init_[3], tf_Lprior_L0_init_[4], tf_Lprior_L0_init_[5]);
            
            myTf tf_Lprior_L0_init(Util::YPR2Quat(tf_Lprior_L0_init_[3], tf_Lprior_L0_init_[4], tf_Lprior_L0_init_[5]),
                                   Vector3d(tf_Lprior_L0_init_[0], tf_Lprior_L0_init_[1], tf_Lprior_L0_init_[2]));
            
            // lock_guard<mutex>lg(relocBufMtx);
            // relocBuf.push_back(tf_Lprior_L0_init);

            geometry_msgs::PoseStamped reloc_pose;
            reloc_pose.header.stamp = ros::Time::now();
            reloc_pose.header.frame_id = "priormap";
            reloc_pose.pose.position.x = tf_Lprior_L0_init.pos(0);
            reloc_pose.pose.position.y = tf_Lprior_L0_init.pos(1);
            reloc_pose.pose.position.z = tf_Lprior_L0_init.pos(2);
            reloc_pose.pose.orientation.x = tf_Lprior_L0_init.rot.x();
            reloc_pose.pose.orientation.y = tf_Lprior_L0_init.rot.y();
            reloc_pose.pose.orientation.z = tf_Lprior_L0_init.rot.z();
            reloc_pose.pose.orientation.w = tf_Lprior_L0_init.rot.w();

            reloc_init = std::thread(&Estimator::PublishManualReloc, this, reloc_pose);
        }

        // The name of the keyframe cloud
        string priormap_kfprefix = "cloud";
        nh_ptr->param("/priormap_kfprefix", priormap_kfprefix, string("cloud"));

        // Downsampling rate for visualizing the priormap
        nh_ptr->param("/priormap_viz_res", priormap_viz_res, 0.2);

        // Refine the relocalization transform
        refine_reloc_tf = GetBoolParam("/relocalization/refine_reloc_tf", false);
        marginalize_new_points = GetBoolParam("/relocalization/marginalize_new_points", false);

        // Get the maximum
        nh_ptr->param("/relocalization/ioa_max_iter", ioa_max_iter, 20);
        printf("ioa_max_iter: %d\n", ioa_max_iter);

        // Subscribe to the relocalization
        relocSub = nh_ptr->subscribe("/reloc_pose", 100, &Estimator::RelocCallback, this);

        // pmSurfGlobal = CloudXYZIPtr(new CloudXYZI());
        // pmEdgeGlobal = CloudXYZIPtr(new CloudXYZI());
        priorMap = CloudXYZIPtr(new CloudXYZI());

        // Initializing priormap
        priorMapPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/priormap", 10);

        string prior_map_dir = "";
        nh_ptr->param("/prior_map_dir", prior_map_dir, string(""));

        // Read the pose log of priormap
        string pmPose_ = prior_map_dir + "/kfpose6d.pcd";
        pmPose = CloudPosePtr(new CloudPose());
        pcl::io::loadPCDFile<PointPose>(pmPose_, *pmPose);

        int PM_KF_COUNT = pmPose->size();
        printf(KGRN "Prior map path %s. Num scans: %d. Begin loading ...\n" RESET, pmPose_.c_str(), pmPose->size());
        
        pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

        if (!std::filesystem::exists(prior_map_dir + "/ufo_surf_map.um"))
        {
            printf("Prebuilt UFO surf map not found, creating one.\n");

            // Reading the surf feature from log
            pmFull = deque<CloudXYZIPtr>(PM_KF_COUNT);
            #pragma omp parallel for num_threads(MAX_THREADS)
            for (int i = 0; i < PM_KF_COUNT; i++)
            {
                pmFull[i] = CloudXYZIPtr(new CloudXYZI());
                string pmFull_ = prior_map_dir + "/pointclouds/" + priormap_kfprefix + "_" + zeroPaddedString(i, PM_KF_COUNT) + ".pcd";
                pcl::io::loadPCDFile<PointXYZI>(pmFull_, *pmFull[i]);

                printf("Reading scan %s.\n", zeroPaddedString(i, PM_KF_COUNT).c_str());
            }

            pmLoaded = true;
            
            printf("Merging the scans:\n");

            // Merge the scans
            for (int i = 0; i < PM_KF_COUNT; i++)
            {
                *priorMap += *pmFull[i];
                printf("Scan %s merged\n", zeroPaddedString(i, PM_KF_COUNT).c_str());
            }

            printf("Surfelizing the prior map.\n");

            priorSurfelMapPtr = ufoSurfelMapPtr(new ufoSurfelMap(leaf_size, surfel_map_depth));
            insertCloudToSurfelMap(*priorSurfelMapPtr, *priorMap);

            // Downsample the prior map for visualization in another thread
            auto pmVizFunctor = [this](const CloudXYZIPtr& priorMap_)->void
            {
                // CloudXYZI priorMapDS;
                pcl::UniformSampling<PointXYZI> downsampler;
                downsampler.setRadiusSearch(priormap_viz_res);
                downsampler.setInputCloud(priorMap);
                downsampler.filter(*priorMap);

                Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");

                if (!refine_reloc_tf)
                {
                    for(auto &cloud : pmFull)
                        cloud->clear();
                }
                
                return;
            };
            initPriorMapThread = std::thread(pmVizFunctor, std::ref(priorMap));

           printf("Save the prior map...\n");

           // Save the ufomap object
           priorSurfelMapPtr->write(prior_map_dir + "/ufo_surf_map.um");
        }
        else
        {
            printf("Prebuilt UFO surf map found, loading...\n");
            priorSurfelMapPtr = ufoSurfelMapPtr(new ufoSurfelMap(prior_map_dir + "/ufo_surf_map.um"));

            // Merge and downsample the prior map for visualization in another thread
            auto pmVizFunctor = [this, priormap_kfprefix](string prior_map_dir, int PM_KF_COUNT, CloudXYZIPtr& priorMap_)->void
            {
                // Reading the surf feature from log
                pmFull = deque<CloudXYZIPtr>(PM_KF_COUNT);
                #pragma omp parallel for num_threads(MAX_THREADS)
                for (int i = 0; i < PM_KF_COUNT; i++)
                {
                    pmFull[i] = CloudXYZIPtr(new CloudXYZI());
                    string pmFull_ = prior_map_dir + "/pointclouds/" + priormap_kfprefix + "_" + zeroPaddedString(i, PM_KF_COUNT) + ".pcd";
                    pcl::io::loadPCDFile<PointXYZI>(pmFull_, *pmFull[i]);

                    printf("Reading scan %s.\n", zeroPaddedString(i, PM_KF_COUNT).c_str());
                }

                pmLoaded = true;
                
                // Merge the scans
                for (int i = 0; i < pmFull.size(); i++)
                {
                    *priorMap += *pmFull[i];
                    // printf("Map size: %d\n", priorMap->size());
                }

                // CloudXYZI priorMapDS;
                pcl::UniformSampling<PointXYZI> downsampler;
                downsampler.setRadiusSearch(priormap_viz_res);
                downsampler.setInputCloud(priorMap);
                downsampler.filter(*priorMap);

                Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");

                if (!refine_reloc_tf)
                {
                    for(auto &cloud : pmFull)
                        cloud->clear();
                }
                
                return;
            };
            initPriorMapThread = std::thread(pmVizFunctor, prior_map_dir, PM_KF_COUNT, std::ref(priorMap));
        }

        // printf(KYEL "Surfelizing the scans.\n" RESET);

        // // Surfelize the surf map
        // surfelMapSurf = ufoSurfelMap(leaf_size);
        // insertCloudToSurfelMap(surfelMapSurf, *pmSurfGlobal);

        // // Surfelize the edge map
        // surfelMapEdge = ufoSurfelMap(leaf_size);
        // insertCloudToSurfelMap(surfelMapEdge, *pmEdgeGlobal);

        printf(KGRN "Done. Time: %f\n" RESET, tt_initprior.Toc());

        // pmVizTimer = nh_ptr->createTimer(ros::Duration(5.0), &Estimator::PublishPriorMap, this);
    }

    void PublishManualReloc(geometry_msgs::PoseStamped relocPose_)
    {
        ros::Publisher relocPub = nh_ptr->advertise<geometry_msgs::PoseStamped>("/reloc_pose", 100);
        geometry_msgs::PoseStamped relocPose = relocPose_;
        while(true)
        {
            if(reloc_stat != RELOCALIZED)
            {
                relocPub.publish(relocPose);
                printf("Manual reloc pose published.\n");
            }
            else
                break;

            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }

    void RelocCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        static bool one_shot = true;
        if (!one_shot)
            return;
        one_shot = false;

        if(reloc_stat == RELOCALIZED)
            return;

        myTf<double> tf_Lprior_L0(*msg);
        
        printf(KYEL "Received Reloc Pose: %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f\n" RESET,
                     tf_Lprior_L0.pos(0), tf_Lprior_L0.pos(1), tf_Lprior_L0.pos(2),
                     tf_Lprior_L0.yaw(), tf_Lprior_L0.pitch(), tf_Lprior_L0.roll());

        if(refine_reloc_tf)
        {
            reloc_stat = RELOCALIZING;

            while (!pmLoaded)
            {
                printf(KYEL "Waiting for prior map to be done\n" RESET);
                this_thread::sleep_for(chrono::milliseconds(100));
            }

            while(KfCloudPose->size() == 0)
            {
                printf(KYEL "Waiting for first keyframe to be made\n" RESET);
                this_thread::sleep_for(chrono::milliseconds(100));
            }

            // Search for closeby prior keyframes
            pcl::KdTreeFLANN<PointPose> kdTreePmPose;
            kdTreePmPose.setInputCloud(pmPose);

            int knn_nbrkf = min(5, (int)pmPose->size());
            vector<int> knn_idx(knn_nbrkf); vector<float> knn_sq_dis(knn_nbrkf);
            kdTreePmPose.nearestKSearch((tf_Lprior_L0*myTf(KfCloudPose->back())).Pose6D(), knn_nbrkf, knn_idx, knn_sq_dis);

            // Create a local map
            CloudXYZIPtr localMap(new CloudXYZI());
            for(int i = 0; i < knn_idx.size(); i++)
                *localMap += *pmFull[knn_idx[i]];

            // Create a cloud matcher
            CloudMatcher cm(0.1, 0.1);

            // Run ICP to find the relative pose
            // Matrix4f tfm_Lprior_L0;
            // double icpFitness = 0;
            // double icpTime = 0;
            // cm.CheckICP(localMap, KfCloudinW.back(), tf_Lprior_L0.cast<float>().tfMat(), tfm_Lprior_L0,
            //             10, 10, 1.0, icpFitness, icpTime);
            
            // tf_Lprior_L0 = myTf(tfm_Lprior_L0).cast<double>();

            // printf(KGRN "Refine the transform: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f. Fitness: %f. Time: %f\n" RESET,
            //         tf_Lprior_L0.pos.x(), tf_Lprior_L0.pos.y(), tf_Lprior_L0.pos.z(),
            //         tf_Lprior_L0.yaw(), tf_Lprior_L0.pitch(), tf_Lprior_L0.roll(),
            //         icpFitness, icpTime);

            IOAOptions ioaOpt;
            ioaOpt.init_tf = tf_Lprior_L0;
            ioaOpt.max_iterations = ioa_max_iter;
            ioaOpt.show_report = true;
            ioaOpt.text = "T_Lprior_L0_refined_" + std::to_string(ioa_max_iter);
            // ioaOpt.fix_rot = fix_rot;
            // ioaOpt.fix_trans = fix_trans;
            IOASummary ioaSum;
            cm.IterateAssociateOptimize(ioaOpt, ioaSum, localMap, KfCloudinW.back());

            tf_Lprior_L0 = ioaSum.final_tf;

            printf(KGRN "Refined the transform: %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f. Time: %f\n" RESET,
                    tf_Lprior_L0.pos.x(), tf_Lprior_L0.pos.y(), tf_Lprior_L0.pos.z(),
                    tf_Lprior_L0.yaw(), tf_Lprior_L0.pitch(), tf_Lprior_L0.roll(), ioaSum.process_time);
        }
        
        // Create an auto exitting scope
        {
            lock_guard<mutex>lg(relocBufMtx);
            relocBuf.push_back(tf_Lprior_L0);
        }
    }

    void PublishPriorMap(const ros::TimerEvent& event)
    {
        Util::publishCloud(priorMapPub, *priorMap, ros::Time::now(), "map");
    }

    bool GetBoolParam(string param, bool default_value)
    {
        int param_;
        nh_ptr->param(param, param_, default_value == true ? 1 : 0);
        return (param_ == 0 ? false : true);
    }

    bool TimeIsValid(PoseSplineX &traj, double time, double tolerance = 0)
    {
        return traj.minTime() + tolerance < time && time < traj.maxTime() - tolerance;
    }

    void DataHandler(const slict::FeatureCloud::ConstPtr &msg)
    {
        lock_guard<mutex> lock(packet_buf_mtx);
        packet_buf.push_back(msg);
    }

    bool ProcessData()
    {
        while (ros::ok())
        {
            slict::TimeLog tlog; TicToc tt_whileloop;

            // Check for time out to exit the program
            static TicToc tt_time_out;

            static double data_time_out = -1;
            if ( (data_time_out != -1) && (tt_time_out.Toc()/1000.0 - data_time_out) > 20 && (packet_buf.size() == 0) && autoexit)
            {
                printf(KYEL "Data timeout, Buf: %d. exit!\n" RESET, packet_buf.size());
                SaveTrajLog();
                exit(0);
            }

            /* #region STEP 0: Loop if there is no new data ---------------------------------------------------------*/
            
            TicToc tt_loop;

            if (packet_buf.empty())
            {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }

            /* #endregion STEP 0: Loop if there is no new data ------------------------------------------------------*/

            /* #region STEP 1: Extract the data packet --------------------------------------------------------------*/
            
            tt_preopt.Tic();

            TicToc tt_extract;

            slict::FeatureCloud::ConstPtr packet;
            {
                lock_guard<mutex> lock(packet_buf_mtx);
                packet = packet_buf.front();
                packet_buf.pop_front();
            }

            // Reset the time out clock
            data_time_out = tt_time_out.Toc()/1000.0;

            tt_extract.Toc();

            /* #endregion STEP 1: Extract the data packet -----------------------------------------------------------*/

            /* #region STEP 2: Initialize orientation and Map -------------------------------------------------------*/
            
            TicToc tt_init;

            if (!ALL_INITED)
            {
                InitSensorData(packet);
                if (!ALL_INITED)
                    continue;
            }

            tt_init.Toc();

            /* #endregion STEP 2: Initialize orientation and Map ----------------------------------------------------*/

            /* #region STEP 3: Insert the data to the buffers -------------------------------------------------------*/
            
            TicToc tt_insert;

            // Extend the time steps
            AddNewTimeStep(SwTimeStep, packet);

            // Copy the pointcloud
            SwCloud.push_back(CloudXYZITPtr(new CloudXYZIT()));
            pcl::fromROSMsg(packet->extracted_cloud, *SwCloud.back());

            // Downsample the scan
            if(leaf_size > 0)
            {
                pcl::UniformSampling<PointXYZIT> downsampler;
                downsampler.setRadiusSearch(leaf_size);
                downsampler.setInputCloud(SwCloud.back());
                downsampler.filter(*SwCloud.back());
            }

            // Create the container for the latest pointcloud
            SwCloudDsk.push_back(CloudXYZIPtr(new CloudXYZI()));

            // Create the container for the downsampled deskewed pointcloud
            SwCloudDskDS.push_back(CloudXYZIPtr(new CloudXYZI()));

            // Buffer to store the coefficients of the lidar factors
            SwLidarCoef.push_back(vector<LidarCoef>());

            // Buffer to count the number of associations per voxel
            SwDepVsAssoc.push_back(map<int, int>());

            // Add buffer the IMU samples at the last state
            AddImuToBuff(SwTimeStep, SwImuBundle, packet, regularize_imu);

            // Imu propagated states
            SwPropState.push_back(deque<ImuProp>(N_SUB_SEG));

            // Extend the spline
            if (GlobalTraj == nullptr)
            {
                GlobalTraj = PoseSplinePtr(new PoseSplineX(SPLINE_N, deltaT));
                GlobalTraj->setStartTime(SwTimeStep.front().front().start_time);
                printf("Creating spline of order %d, dt %f s. Time: %f\n", SPLINE_N, deltaT, SwTimeStep.front().front().start_time);
            }
            GlobalTraj->extendKnotsTo(SwTimeStep.back().back().final_time, SE3d());
            
            tlog.t_insert = tt_insert.Toc();

            /* #endregion STEP 3: Insert the data to the buffers ----------------------------------------------------*/

            /* #region STEP 4: IMU Propagation on the last segments -------------------------------------------------*/

            TicToc tt_imuprop;

            for(int i = 0; i < SwImuBundle.back().size(); i++)
            {
                auto &imuSubSeq = SwImuBundle.back()[i];
                auto &subSegment = SwTimeStep.back()[i];

                // ASSUMPTION: IMU data is not interrupted (any sub-segment should have IMU data), can be relaxed ?
                // printf("Step: %2d / %2d. imuSubSeq.size(): %d\n", i, SwImuBundle.back().size(), imuSubSeq.size());
                ROS_ASSERT(!imuSubSeq.empty());

                // ASSUMPTION: an IMU sample is interpolated at each segment's start and end point
                ROS_ASSERT_MSG(imuSubSeq.front().t == subSegment.start_time && imuSubSeq.back().t == subSegment.final_time,
                               "IMU Time: %f, %f. Seg. Time: %f, %f\n",
                               imuSubSeq.front().t, imuSubSeq.back().t, subSegment.start_time, subSegment.final_time);

                SwPropState.back()[i] = ImuProp(ssQua.back()[i], ssPos.back()[i], ssVel.back()[i],
                                                ssBig.back()[i], ssBia.back()[i], GRAV, imuSubSeq);

                sfQua.back()[i] = SwPropState.back()[i].Q.back();
                sfPos.back()[i] = SwPropState.back()[i].P.back();
                sfVel.back()[i] = SwPropState.back()[i].V.back();

                // Initialize start state of next segment with the final propogated state in the previous segment
                if (i <= SwImuBundle.back().size() - 2)
                {
                    ssQua.back()[i+1] = sfQua.back()[i];
                    ssPos.back()[i+1] = sfPos.back()[i];
                    ssVel.back()[i+1] = sfVel.back()[i];
                }
            }

            tlog.t_prop.push_back(tt_imuprop.Toc());

            /* #endregion STEP 4: IMU Propagation on the last segments ----------------------------------------------*/

            /* #region STEP 5: Intialize the extended part of the spline --------------------------------------------*/

            TicToc tt_extspline;
            
            static int last_updated_knot = -1;

            // Initialize the spline knots by the propagation value
            int baseKnot = GlobalTraj->computeTIndex(SwPropState.back().front().t[0]).second + 1;
            for(int knot_idx = baseKnot; knot_idx < GlobalTraj->numKnots(); knot_idx++)
            {
                // Initialize by linear interpolation
                double knot_time = GlobalTraj->getKnotTime(knot_idx);
                // Find the propagated pose
                for(int seg_idx = 0; seg_idx < SwPropState.size(); seg_idx++)
                {
                    for(int subseg_idx = 0; subseg_idx < SwPropState.back().size(); subseg_idx++)
                    {
                        if(SwPropState.back()[subseg_idx].t.back() < knot_time)
                            continue;

                        GlobalTraj->setKnot(SwPropState.back()[subseg_idx].getTf(knot_time).getSE3(), knot_idx);
                        break;
                    }
                }

                // Initialize by copying the previous control point
                if (GlobalTraj->getKnotTime(knot_idx) >= GlobalTraj->maxTime()
                    || GlobalTraj->getKnotTime(knot_idx) >= SwPropState.back().back().t.back())
                {
                    // GlobalTraj->setKnot(SwPropState.back().back().getTf(knot_time, false).getSE3(), knot_idx);
                    GlobalTraj->setKnot(GlobalTraj->getKnot(knot_idx-1), knot_idx);
                    continue;
                }
            }
            
            // Fit the spline at the ending segment to avoid high cost
            std::thread threadFitSpline;
            if (fit_spline)
            {
                static bool fit_spline_enabled = false;
                if (SwTimeStep.size() >= WINDOW_SIZE && !fit_spline_enabled)
                    fit_spline_enabled = true;
                else if(fit_spline_enabled)
                    threadFitSpline = std::thread(&Estimator::FitSpline, this);
            }

            tt_extspline.Toc();

            /* #endregion STEP 5: Intialize the extended part of the spline -----------------------------------------*/

            // Loop if sliding window has not reached required length
            if (SwTimeStep.size() < WINDOW_SIZE)
            {
                printf(KGRN "Buffer size %02d / %02d\n" RESET, SwTimeStep.size(), WINDOW_SIZE);
                continue;
            }
            else
            {
                static bool first_shot = true;
                if (first_shot)
                {
                    first_shot = false;
                    printf(KGRN "Buffer size %02d / %02d. WINDOW SIZE reached.\n" RESET, SwTimeStep.size(), WINDOW_SIZE);
                }
            }

            // Reset the association if ufomap has been updated
            static int last_ufomap_version = ufomap_version;
            static bool first_round = true;
            if (ufomap_version != last_ufomap_version)
            {
                first_round   = true;
                last_ufomap_version = ufomap_version;

                printf(KYEL "UFOMAP RESET.\n" RESET);
            }

            /* #region STEP 6: DESKEW the pointcloud ----------------------------------------------------------------*/
            
            string t_deskew;
            double tt_deskew = 0;

            for (int i = first_round ? 0 : WINDOW_SIZE - 1; i < WINDOW_SIZE; i++)
            {
                TicToc tt;

                switch (deskew_method[0])
                {
                    case 0:
                        DeskewByImu(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);
                        break;
                    case 1:
                        DeskewBySpline(*GlobalTraj, SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);
                        break;
                    default:
                        break;
                }

                // Check the timing
                tt.Toc();
                t_deskew  += myprintf("#%d: %3.1f, ", i, tt.GetLastStop());
                tt_deskew += tt.GetLastStop();
            }

            t_deskew = myprintf("deskew: %3.1f, ", tt_deskew) + t_deskew;

            tlog.t_desk.push_back(tt_deskew);

            /* #endregion STEP 6: DESKEW the pointcloud -------------------------------------------------------------*/

            /* #region STEP 7: Associate scan with map --------------------------------------------------------------*/
            
            string t_assoc;
            double tt_assoc = 0;

            for (int i = first_round ? 0 : WINDOW_SIZE - 1; i < WINDOW_SIZE; i++)
            {
                TicToc tt;

                lock_guard<mutex> lg(map_mtx);
                SwDepVsAssoc[i].clear(); SwLidarCoef[i].clear();
                AssociateCloudWithMap(*activeSurfelMap, activeikdtMap, mytf(sfQua[i].back(), sfPos[i].back()),
                                       SwCloud[i], SwCloudDskDS[i], SwLidarCoef[i], SwDepVsAssoc[i]);

                // Check the timing
                tt.Toc();
                t_assoc  += myprintf("#%d: %3.1f, ", i, tt.GetLastStop());
                tt_assoc += tt.GetLastStop();
            }
            
            t_assoc = myprintf("assoc: %3.1f, ", tt_assoc) + t_assoc;

            tlog.t_assoc.push_back(tt_assoc);

            // find_new_node = false;

            /* #endregion STEP 7: Associate scan with map -----------------------------------------------------------*/

            /* #region STEP 8: LIO optimizaton ----------------------------------------------------------------------*/

            tt_preopt.Toc();

            static int optNum = 0; optNum++;
            vector<slict::OptStat> optreport(max_outer_iters);
            for (auto &report : optreport)
                report.OptNum = optNum;

            // Update the time check
            tlog.OptNum = optNum;
            tlog.header.stamp = ros::Time(SwTimeStep.back().back().final_time);

            string printout, lioop_times_report = "", DVAReport;

            int outer_iter = max_outer_iters;
            while(true)
            {
                // Decrement outer interation counter
                outer_iter--;

                // Prepare a report
                slict::OptStat &report = optreport[outer_iter];

                // Calculate the downsampling rate at each depth            
                makeDVAReport(SwDepVsAssoc, DVA, total_lidar_coef, DVAReport);
                lidar_ds_rate = (max_lidar_factor == -1 ? 1 : max(1, (int)std::floor( (double)total_lidar_coef/max_lidar_factor) ));

                // if(threadFitSpline.joinable())
                //     threadFitSpline.join();

                // Create a local spline to store the new knots, isolating the poses from the global trajectory
                PoseSplineX LocalTraj(SPLINE_N, deltaT);
                int swBaseKnot = GlobalTraj->computeTIndex(SwImuBundle[0].front().front().t).second;
                int swNextBase = GlobalTraj->computeTIndex(SwImuBundle[1].front().front().t).second;

                static map<int, int> prev_knot_x;
                static map<int, int> curr_knot_x;

                double swStartTime = GlobalTraj->getKnotTime(swBaseKnot);
                double swFinalTime = SwTimeStep.back().back().final_time - 1e-3; // Add a small offset to avoid localtraj having extra knots due to rounding error

                LocalTraj.setStartTime(swStartTime);
                LocalTraj.extendKnotsTo(swFinalTime, SE3d());

                // Copy the knots value from global to local traj
                for(int knot_idx = swBaseKnot; knot_idx < GlobalTraj->numKnots(); knot_idx++)
                {
                    if ((knot_idx - swBaseKnot) > LocalTraj.numKnots() - 1)
                        continue;
                    LocalTraj.setKnot(GlobalTraj->getKnot(knot_idx), knot_idx - swBaseKnot);
                }

                // Check insantity
                ROS_ASSERT_MSG(LocalTraj.numKnots() <= GlobalTraj->numKnots() - swBaseKnot,
                               "Knot count not matching %d, %d, %d\n",
                               LocalTraj.numKnots(), GlobalTraj->numKnots() - swBaseKnot, swBaseKnot);

                // Accounting for the knot idx for marginalization
                curr_knot_x.clear();
                for(int knot_idx = 0; knot_idx < LocalTraj.numKnots(); knot_idx++)
                    curr_knot_x[knot_idx + swBaseKnot] = knot_idx;

                TicToc tt_feaSel;
                // Select the features
                vector<ImuIdx> imuSelected;
                vector<lidarFeaIdx> featureSelected;
                FactorSelection(LocalTraj, imuSelected, featureSelected);
                // Visualize the selection
                PublishAssocCloud(featureSelected, SwLidarCoef);
                tt_feaSel.Toc();

                tlog.t_feasel.push_back(tt_feaSel.GetLastStop());

                // Optimization
                lioop_times_report = "";
                LIOOptimization(report, lioop_times_report, LocalTraj,
                                prev_knot_x, curr_knot_x, swNextBase, outer_iter,
                                imuSelected, featureSelected, tlog);

                // Load the knot values back to the global traj
                for(int knot_idx = 0; knot_idx < LocalTraj.numKnots(); knot_idx++)
                {
                    GlobalTraj->setKnot(LocalTraj.getKnot(knot_idx), knot_idx + swBaseKnot);
                    last_updated_knot = knot_idx + swBaseKnot;
                }

                /* #region Post optimization ------------------------------------------------------------------------*/

                TicToc tt_posproc;

                string pstop_times_report = "pp: ";

                // Break the loop early if the optimization finishes quickly.
                bool redo_optimization = true;
                if (outer_iter <= 0
                    || (outer_iter <= max_outer_iters - 2
                        && report.JK < report.J0
                        && (report.J0 - report.JK)/report.J0 < dj_thres )
                   )
                    redo_optimization = false;

                // int PROP_THREADS = std::min(WINDOW_SIZE, MAX_THREADS);
                // Redo propagation

                TicToc tt_prop_;

                #pragma omp parallel for num_threads(WINDOW_SIZE)
                for (int i = 0; i < WINDOW_SIZE; i++)
                {
                    for(int j = 0; j < SwTimeStep[i].size(); j++)
                    {
                        SwPropState[i][j] = ImuProp(sfQua[i][j], sfPos[i][j], sfVel[i][j],
                                                    sfBig[i][j], sfBia[i][j], GRAV, SwImuBundle[i][j], -1);

                        if (i == 0 && j == 0)
                        {
                            ssQua[i][j] = SwPropState[i][j].Q.front();
                            ssPos[i][j] = SwPropState[i][j].P.front();
                            ssVel[i][j] = SwPropState[i][j].V.front();
                        }
                    }
                }

                tlog.t_prop.push_back(tt_prop_.Toc());
                pstop_times_report += myprintf("prop: %.1f, ", tlog.t_prop.back());

                TicToc tt_deskew_;

                // Redo the deskew
                if(lite_redeskew)
                    if (redo_optimization)  // If optimization is gonna be done again, deskew with light method
                        for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                            Redeskew(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDskDS[i]);
                    else                    // If optimization is not gonna be done again, deskew with heavy method
                        for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                            DeskewByImu(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);
                else
                    for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                        DeskewByImu(SwPropState[i], SwTimeStep[i], SwCloud[i], SwCloudDsk[i], SwCloudDskDS[i], assoc_spacing);

                tlog.t_desk.push_back(tt_deskew_.Toc());
                pstop_times_report += myprintf("dsk: %.1f, ", tlog.t_desk.back());

                TicToc tt_assoc_;
    
                // Redo the map association
                for (int i = first_round ? 0 : max(0, WINDOW_SIZE - reassociate_steps); i < WINDOW_SIZE; i++)
                {
                    // TicToc tt_assoc;
                    
                    lock_guard<mutex> lg(map_mtx);
                    SwDepVsAssoc[i].clear(); SwLidarCoef[i].clear();
                    AssociateCloudWithMap(*activeSurfelMap, activeikdtMap, mytf(sfQua[i].back(), sfPos[i].back()),
                                           SwCloud[i], SwCloudDskDS[i], SwLidarCoef[i], SwDepVsAssoc[i]);

                    // printf("Assoc Time: %f\n", tt_assoc.Toc());
                }

                tlog.t_assoc.push_back(tt_assoc_.Toc());
                pstop_times_report += myprintf("assoc: %.1f, ", tlog.t_assoc.back());

                tt_posproc.Toc();

                /* #endregion Post optimization ---------------------------------------------------------------------*/

                /* #region Write the report -------------------------------------------------------------------------*/

                // Update the report
                report.header.stamp   = ros::Time(SwTimeStep.back().back().final_time);
                report.OptNumSub      = outer_iter + 1;
                report.keyfrm         = KfCloudPose->size();
                report.margPerc       = margPerc;
                report.fixed_knot_min = first_fixed_knot;
                report.fixed_knot_max = last_fixed_knot;

                report.tpreopt        = tt_preopt.GetLastStop();
                report.tpostopt       = tt_posproc.GetLastStop();
                report.tlp            = tt_loop.Toc();
                
                static double last_tmapping = -1;
                if (last_tmapping != tt_margcloud.GetLastStop())
                {
                    report.tmapimg = tt_margcloud.GetLastStop();
                    last_tmapping  = tt_margcloud.GetLastStop();
                }
                else
                    report.tmapimg = -1;

                Vector3d eul_est = Util::Quat2YPR(Quaternd(report.Qest.w, report.Qest.x, report.Qest.y, report.Qest.z));
                Vector3d eul_imu = Util::Quat2YPR(Quaternd(report.Qest.w, report.Qest.x, report.Qest.y, report.Qest.z));
                Vector3d Vest = Vector3d(report.Vest.x, report.Vest.y, report.Vest.z);
                Vector3d Vimu = Vector3d(report.Vimu.x, report.Vimu.y, report.Vimu.z);

                /* #region */
                printout +=
                    show_report ?
                    myprintf("Op#.Oi#: %04d. %2d /%2d. Itr: %2d / %2d. trun: %.3f. %s. RL: %d\n"
                             "tpo: %4.0f. tfs: %4.0f. tbc: %4.0f. tslv: %4.0f / %4.0f. tpp: %4.0f. tlp: %4.0f. tufm: %4.0f. tlpBa: %4.0f.\n"
                             "Ftr: Ldr: %5d / %5d / %5d. IMU: %5d. Prop: %5d. Vel: %2d. Buf: %2d. Kfr: %d. Marg%%: %6.3f. Kfca: %d. "
                             "Fixed: %d -> %d. "
                             "Map: %d\n"
                             "J0:  %15.3f, Ldr: %9.3f. IMU: %9.3f. Prp: %9.3f. Vel: %9.3f.\n"
                             "JK:  %15.3f, Ldr: %9.3f. IMU: %9.3f. Prp: %9.3f. Vel: %9.3f.\n"
                            //  "BiaG: %7.2f, %7.2f, %7.2f. BiaA: %7.2f, %7.2f, %7.2f. (%7.2f, %7.2f, %7.2f), (%7.2f, %7.2f, %7.2f)\n"
                            //  "Eimu: %7.2f, %7.2f, %7.2f. Pimu: %7.2f, %7.2f, %7.2f. Vimu: %7.2f, %7.2f, %7.2f.\n"
                            //  "Eest: %7.2f, %7.2f, %7.2f. Pest: %7.2f, %7.2f, %7.2f. Vest: %7.2f, %7.2f, %7.2f. Spd: %.3f. Dif: %.3f.\n"
                             "DVA:  %s\n",
                             // Time and iterations
                             report.OptNum, report.OptNumSub, max_outer_iters,
                             report.iters, max_iterations,
                             report.trun,
                             reloc_stat == RELOCALIZED ? KYEL "RELOCALIZED!" RESET : "",
                             relocBuf.size(),
                             report.tpreopt,           // time preparing before lio optimization
                             tt_fitspline.GetLastStop(), // time to fit the spline
                             report.tbuildceres,       // time building the ceres problem before solving
                             report.tslv,              // time solving ceres problem
                             t_slv_budget,             // time left to solve the problem
                             report.tpostopt,          // time for post processing
                             report.tlp,               // time packet was extracted up to now
                             report.tmapimg,           // time of last insertion of data to ufomap
                             tt_loopBA.GetLastStop(),  // time checking loop closure
                             // Sliding window stats
                             report.surfFactors, max_lidar_factor, total_lidar_coef,
                             report.imuFactors, report.propFactors, report.velFactors,
                             report.mfcBuf = packet_buf.size(), report.keyfrm, report.margPerc*100, report.kfcand,
                            //  active_knots.begin()->first, active_knots.rbegin()->first,
                             report.fixed_knot_min, report.fixed_knot_max,
                             use_ufm ? activeSurfelMap->size() : activeikdtMap->size(),
                             // Optimization initial costs
                             report.J0, report.J0Surf, report.J0Imu, report.J0Prop, report.J0Vel,
                             // Optimization final costs
                             report.JK, report.JKSurf, report.JKImu, report.JKProp, report.JKVel,
                             // Bias Estimate
                            //  ssBig.back().back().x(), ssBig.back().back().y(), ssBig.back().back().z(),
                            //  ssBia.back().back().x(), ssBia.back().back().y(), ssBia.back().back().z(),
                            //  BG_BOUND(0), BG_BOUND(1), BG_BOUND(2), BA_BOUND(0), BA_BOUND(1), BA_BOUND(2),
                             // Pose Estimate from propogation
                            //  eul_imu.x(), eul_imu.y(), eul_imu.z(),
                            //  report.Pimu.x, report.Pimu.y, report.Pimu.z,
                            //  report.Vimu.x, report.Vimu.y, report.Vimu.z,
                             // Pose Estimate from Optimization
                            //  eul_est.x(), eul_est.y(), eul_est.z(),
                            //  report.Pest.x, report.Pest.y, report.Pest.z,
                            //  report.Vest.x, report.Vest.y, report.Vest.z,
                            //  Vest.norm(), (Vest - Vimu).norm(),
                             // Report on the assocations at different scales
                             DVAReport.c_str())
                    : "\n";
                /* #endregion */

                // Attach the report from loop closure
                /* #region */
                printout +=
                    myprintf("%sBA# %4d. LoopEn: %d. LastFn: %6.3f. Itr: %3d. tslv: %4.0f. trbm: %4.0f. Ftr: RP: %4d. Lp: %4d.\n"
                             "J:  %6.3f -> %6.3f. rP: %6.3f -> %6.3f. Lp: %6.3f -> %6.3f\n" RESET,
                             // Stats
                             baReport.turn % 2 == 0 ? KBLU : KGRN, baReport.turn, loop_en, lastICPFn,
                             baReport.pgopt_iter, baReport.pgopt_time, baReport.rebuildmap_time,
                             baReport.factor_relpose, baReport.factor_loop,
                             // Costs
                             baReport.J0, baReport.JK,
                             baReport.J0_relpose, baReport.JK_relpose,
                             baReport.J0_loop, baReport.JK_loop);

                // Show the preop times
                string preop_times_report = "";
                if (GetBoolParam("/show_preop_times", false))
                {
                    preop_times_report += "Preop: ";
                    preop_times_report += myprintf("insert:  %3.1f, ", tt_insert.GetLastStop());
                    preop_times_report += myprintf("imuprop: %3.1f, ", tt_imuprop.GetLastStop());
                    preop_times_report += myprintf("extspln: %3.1f, ", tt_extspline.GetLastStop());
                    preop_times_report += t_deskew; 
                    preop_times_report += t_assoc;
                    preop_times_report += myprintf("feaSel: %3.1f, ", tt_feaSel.GetLastStop());
                }
                printout += preop_times_report + pstop_times_report + "\n";
                printout += lioop_times_report + "\n";
                /* #endregion */

                // Publish the optimization results
                static ros::Publisher opt_stat_pub = nh_ptr->advertise<slict::OptStat>("/opt_stat", 1);
                opt_stat_pub.publish(report);

                /* #endregion Write the report ----------------------------------------------------------------------*/

                if(!redo_optimization)
                {
                    prev_knot_x = curr_knot_x;
                    break;
                }
            }

            first_round = false;

            /* #endregion STEP 8: LIO optimizaton -------------------------------------------------------------------*/

            /* #region STEP 9: Recruit Keyframe ---------------------------------------------------------------------*/
            
            NominateKeyframe();

            /* #endregion STEP 9: Recruit Keyframe ------------------------------------------------------------------*/

            /* #region STEP 9: Loop Closure and BA ------------------------------------------------------------------*/

            tt_loopBA.Tic();

            if (loop_en)
            {
                DetectLoop();
                BundleAdjustment(baReport);
            }

            tt_loopBA.Toc();

            /* #endregion STEP 9: Loop Closure and BA ---------------------------------------------------------------*/

            /* #region STEP 11: Report and Vizualize ----------------------------------------------------------------*/ 
            
            // Export the summaries
            if (show_report)
                cout << printout;

            // std::thread vizSwTrajThread(&Estimator::VisualizeSwTraj, this);            
            // std::thread vizSwLoopThread(&Estimator::VisualizeLoop, this);
            
            // vizSwTrajThread.join();
            // vizSwLoopThread.join();

            VisualizeSwTraj();
            VisualizeLoop();

            /* #endregion STEP 11: Report and Vizualize -------------------------------------------------------------*/ 

            /* #region STEP 12: Slide window forward ----------------------------------------------------------------*/

            // Slide the window forward
            SlideWindowForward();

            /* #endregion STEP 12: Slide window forward -------------------------------------------------------------*/

            /* #region STEP 13: Transform everything to prior map frame ---------------------------------------------*/

            // Simulated behaviours
            if (use_prior_map && reloc_stat != RELOCALIZED && relocBuf.size() != 0)
            {
                // Extract the relocalization pose
                {
                    lock_guard<mutex>lg(relocBufMtx);
                    tf_Lprior_L0 = relocBuf.back();
                }

                // Move all states to the new coordinates
                Quaternd q_Lprior_L0 = tf_Lprior_L0.rot;
                Vector3d p_Lprior_L0 = tf_Lprior_L0.pos;

                // Transform the traj to the prior map
                for(int knot_idx = 0; knot_idx < GlobalTraj->numKnots(); knot_idx++)
                    GlobalTraj->setKnot(tf_Lprior_L0.getSE3()*GlobalTraj->getKnot(knot_idx), knot_idx);

                // Convert all the IMU poses to the prior map
                for(int i = 0; i < SwPropState.size(); i++)
                {
                    for(int j = 0; j < SwPropState[i].size(); j++)
                    {
                        for (int k = 0; k < SwPropState[i][j].size(); k++)
                        {
                            SwPropState[i][j].Q[k] = q_Lprior_L0*SwPropState[i][j].Q[k];
                            SwPropState[i][j].P[k] = q_Lprior_L0*SwPropState[i][j].P[k] + p_Lprior_L0;
                            SwPropState[i][j].V[k] = q_Lprior_L0*SwPropState[i][j].V[k];
                        }
                    }
                }

                for(int i = 0; i < ssQua.size(); i++)
                {
                    for(int j = 0; j < ssQua[i].size(); j++)
                    {
                        ssQua[i][j] = q_Lprior_L0*ssQua[i][j];
                        ssPos[i][j] = q_Lprior_L0*ssPos[i][j] + p_Lprior_L0;
                        ssVel[i][j] = q_Lprior_L0*ssVel[i][j];

                        sfQua[i][j] = q_Lprior_L0*sfQua[i][j];
                        sfPos[i][j] = q_Lprior_L0*sfPos[i][j] + p_Lprior_L0;
                        sfVel[i][j] = q_Lprior_L0*sfVel[i][j];
                    }
                }

                // Keyframe poses
                pcl::transformPointCloud(*KfCloudPose, *KfCloudPose, tf_Lprior_L0.cast<float>().tfMat());
                
                // Clear the coef of previous steps
                for(int i = 0; i < SwLidarCoef.size(); i++)
                {
                    SwLidarCoef[i].clear();
                    SwDepVsAssoc[i].clear();
                }
                
                // Change the map
                {
                    lock_guard<mutex> lg(map_mtx);
                    
                    if(use_ufm)
                        activeSurfelMap = priorSurfelMapPtr;
                    else
                        activeikdtMap = priorikdtMapPtr;

                    ufomap_version++;
                }

                // Clear the previous keyframe
                #pragma omp parallel for num_threads(MAX_THREADS)
                for(int i = 0; i < KfCloudPose->size(); i++)
                    pcl::transformPointCloud(*KfCloudinW[i], *KfCloudinW[i], tf_Lprior_L0.cast<float>().tfMat());

                // Add keyframe pointcloud to global map
                {
                    lock_guard<mutex> lock(global_map_mtx);
                    globalMap->clear();
                }

                // Log down the relocalization information
                Matrix4d tfMat_L0_Lprior = tf_Lprior_L0.inverse().tfMat();
                Matrix4d tfMat_Lprior_L0 = tf_Lprior_L0.tfMat();

                ofstream prior_fwtf_file;
                prior_fwtf_file.open((log_dir + string("/tf_L0_Lprior.txt")).c_str());
                prior_fwtf_file << std::fixed << std::setprecision(9);
                prior_fwtf_file << tfMat_L0_Lprior;
                prior_fwtf_file.close();

                ofstream prior_rvtf_file;
                prior_rvtf_file.open((log_dir + string("/tf_Lprior_L0.txt")).c_str());
                prior_rvtf_file << std::fixed << std::setprecision(9);
                prior_rvtf_file << tfMat_Lprior_L0;
                prior_rvtf_file.close();

                ofstream reloc_info_file;
                reloc_info_file.open((log_dir + string("/reloc_info.csv")).c_str());
                reloc_info_file << std::fixed << std::setprecision(9);
                reloc_info_file << "Time, "
                                << "TF_Lprior_L0.x, TF_Lprior_L0.y, TF_Lprior_L0.z, "
                                << "TF_Lprior_L0.qx, TF_Lprior_L0.qy, TF_Lprior_L0.qz, TF_Lprior_L0.qw, "
                                << "TF_Lprior_L0.yaw, TF_Lprior_L0.pitch, TF_Lprior_L0.roll"  << endl;
                reloc_info_file << SwTimeStep.back().back().final_time << ", "
                                << tf_Lprior_L0.pos.x() << ", " << tf_Lprior_L0.pos.y() << ", " << tf_Lprior_L0.pos.z() << ", "
                                << tf_Lprior_L0.rot.x() << ", " << tf_Lprior_L0.rot.y() << ", " << tf_Lprior_L0.rot.z() << ", " << tf_Lprior_L0.rot.w() << ", "
                                << tf_Lprior_L0.yaw()   << ", " << tf_Lprior_L0.pitch() << ", " << tf_Lprior_L0.roll()  << endl;
                reloc_info_file.close();

                // Change the frame of reference
                current_ref_frame = "map";

                reloc_stat = RELOCALIZED;
            }

            /* #endregion STEP 13: Transform everything to prior map frame ------------------------------------------*/

            // Publish the loop time
            tlog.t_loop = tt_whileloop.Toc();
            static ros::Publisher tlog_pub = nh_ptr->advertise<slict::TimeLog>("/time_log", 100);
            tlog_pub.publish(tlog);
        }
    }

    void PublishAssocCloud(vector<lidarFeaIdx> &featureSelected, deque<vector<LidarCoef>> &SwLidarCoef)
    {
        static CloudXYZIPtr assocCloud(new CloudXYZI());
        assocCloud->resize(featureSelected.size());

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < featureSelected.size(); i++)
        {
            LidarCoef &coef = SwLidarCoef[featureSelected[i].wdidx][featureSelected[i].pointidx];
            assocCloud->points[i].x = coef.finW(0);
            assocCloud->points[i].y = coef.finW(1);
            assocCloud->points[i].z = coef.finW(2);
            assocCloud->points[i].intensity = featureSelected[i].wdidx;
        }

        static ros::Publisher assoc_cloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/assoc_cloud", 100);
        Util::publishCloud(assoc_cloud_pub, *assocCloud, ros::Time(SwTimeStep.back().back().final_time), current_ref_frame);
    }

    void InitSensorData(slict::FeatureCloud::ConstPtr &packet)
    {
        static bool IMU_INITED = false;
        static bool LIDAR_INITED = false;

        if (!IMU_INITED)
        {
            const vector<sensor_msgs::Imu> &imu_bundle = packet->imu_msgs;

            static vector<Vector3d> gyr_buf;
            static vector<Vector3d> acc_buf;
            static double first_imu_time = imu_bundle.front().header.stamp.toSec();

            // Push the sample into buffer;
            for (auto imu_sample : imu_bundle)
            {
                if (imu_sample.header.seq == 0)
                {
                    gyr_buf.push_back(Vector3d(imu_sample.angular_velocity.x,
                                               imu_sample.angular_velocity.y,
                                               imu_sample.angular_velocity.z));
                    acc_buf.push_back(Vector3d(imu_sample.linear_acceleration.x,
                                               imu_sample.linear_acceleration.y,
                                               imu_sample.linear_acceleration.z));
                }
            }

            // Average the IMU measurements and initialize the states
            if (!gyr_buf.empty() &&
                fabs(imu_bundle.front().header.stamp.toSec() - first_imu_time) > imu_init_time)
            {
                // Calculate the gyro bias
                Vector3d gyr_avr(0, 0, 0);
                for (auto gyr_sample : gyr_buf)
                    gyr_avr += gyr_sample;

                gyr_avr /= gyr_buf.size();

                // Calculate the original orientation
                Vector3d acc_avr(0, 0, 0);
                for (auto acc_sample : acc_buf)
                    acc_avr += acc_sample;

                acc_avr /= acc_buf.size();

                ACC_SCALE = GRAV.norm()/acc_avr.norm();
                
                Quaternd q_init(Util::grav2Rot(acc_avr));
                Vector3d ypr = Util::Quat2YPR(q_init);

                printf("Gyro Bias: %.3f, %.3f, %.3f. Samples: %d. %d\n",
                        gyr_avr(0), gyr_avr(1), gyr_avr(2), gyr_buf.size(), acc_buf.size());
                printf("Init YPR:  %.3f, %.3f, %.3f.\n", ypr(0), ypr(1), ypr(2));

                // Initialize the original quaternion state
                ssQua = sfQua = deque<deque<Quaternd>>(WINDOW_SIZE, deque<Quaternd>(N_SUB_SEG, q_init));
                ssBig = sfBig = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, gyr_avr));
                ssBia = sfBia = deque<deque<Vector3d>>(WINDOW_SIZE, deque<Vector3d>(N_SUB_SEG, Vector3d(0, 0, 0)));

                IMU_INITED = true;
            }
        }
    
        if (!LIDAR_INITED)
        {
            static CloudXYZITPtr kfCloud0_(new CloudXYZIT());
            pcl::fromROSMsg(packet->extracted_cloud, *kfCloud0_);

            if(IMU_INITED)
            {   
                // Downsample the cached kf data
                pcl::UniformSampling<PointXYZIT> downsampler;
                downsampler.setRadiusSearch(leaf_size);
                downsampler.setInputCloud(kfCloud0_);
                downsampler.filter(*kfCloud0_);

                // Admitting only latest 0.09s part of the pointcloud
                printf("PointTime: %f -> %f\n", kfCloud0_->points.front().t, kfCloud0_->points.back().t);
                CloudXYZIPtr kfCloud0(new CloudXYZI());
                for(PointXYZIT &p : kfCloud0_->points)
                {
                    if(p.t > kfCloud0_->points.back().t - 0.09)
                    {
                        PointXYZI pnew;
                        pnew.x = p.x; pnew.y = p.y; pnew.z = p.z; pnew.intensity = p.intensity;
                        kfCloud0->push_back(pnew);   
                    }
                }

                // Key frame cloud in world
                CloudXYZIPtr kfCloud0InW(new CloudXYZI());
                pcl::transformPointCloud(*kfCloud0, *kfCloud0InW, Vector3d(0, 0, 0), sfQua[0].back());

                // Admit the pointcloud to buffer
                AdmitKeyframe(packet->header.stamp.toSec(), sfQua[0].back(), Vector3d(0, 0, 0), kfCloud0, kfCloud0InW);

                // Write the file for quick visualization
                PCDWriter writer; writer.writeASCII<PointPose>(log_dir + "/KfCloudPose.pcd", *KfCloudPose, 18);

                LIDAR_INITED = true;
            }
        }

        if (IMU_INITED && LIDAR_INITED)
            ALL_INITED = true;
    }

    void AddNewTimeStep(deque<deque<TimeSegment>> &timeStepDeque, slict::FeatureCloud::ConstPtr &packet)
    {
        // Add new sequence of sub time step
        timeStepDeque.push_back(deque<TimeSegment>());

        // Calculate the sub time steps
        double start_time, final_time, sub_timestep;
        if (timeStepDeque.size() == 1)
        {
            start_time = packet->scanStartTime;
            final_time = packet->scanEndTime;
            sub_timestep = (final_time - start_time)/N_SUB_SEG;
        }
        else
        {
            start_time = timeStepDeque.rbegin()[1].back().final_time;
            final_time = packet->scanEndTime;
            sub_timestep = (final_time - start_time)/N_SUB_SEG;
        }

        for(int i = 0; i < N_SUB_SEG; i++)
            timeStepDeque.back().push_back(TimeSegment(start_time + i*sub_timestep,
                                                       start_time + (i+1)*sub_timestep));
    }

    void AddImuToBuff(deque<deque<TimeSegment>> &timeStepDeque, deque<deque<ImuSequence>> &imuBundleDeque,
                      slict::FeatureCloud::ConstPtr &packet, bool regularize_imu)
    {
        // Extend the imu deque
        imuBundleDeque.push_back(deque<ImuSequence>(N_SUB_SEG));

        // Extract and regularize imu data at on the latest buffer
        ImuSequence newImuSequence;
        ExtractImuData(newImuSequence, packet, regularize_imu); // Only select the primary IMU at this stage

        // Extract subsequence of imu sample data, interpolate at sub steps
        if(timeStepDeque.size() == 1)
        {
            // Duplicate the first sample for continuity
            newImuSequence.push_front(newImuSequence.front());
            newImuSequence.front().t = timeStepDeque.front().front().start_time;
        }
        else
            // Borrow the last sample time in the previous interval for continuity
            newImuSequence.push_front(imuBundleDeque.rbegin()[1].back().back());

        // Extract the samples in each sub interval          
        for(int i = 0; i < timeStepDeque.back().size(); i++)
        {
            double start_time = timeStepDeque.back()[i].start_time;
            double final_time = timeStepDeque.back()[i].final_time;
            double dt = final_time - start_time;
            
            imuBundleDeque.back()[i] = newImuSequence.subsequence(start_time, final_time);
            for(int j = 0; j < imuBundleDeque.back()[i].size(); j++)
            {
                imuBundleDeque.back()[i][j].u = start_time;
                imuBundleDeque.back()[i][j].s = (imuBundleDeque.back()[i][j].t - start_time)/dt;
            }
        }
    }

    void ExtractImuData(ImuSequence &imu_sequence, slict::FeatureCloud::ConstPtr &packet, bool regularize_timestamp)
    {
        // Copy the messages to the deque
        for(auto &imu : packet->imu_msgs)
        {
            imu_sequence.push_back(ImuSample(imu.header.stamp.toSec(),
                                             Vector3d(imu.angular_velocity.x,
                                                      imu.angular_velocity.y,
                                                      imu.angular_velocity.z),
                                             Vector3d(imu.linear_acceleration.x,
                                                      imu.linear_acceleration.y,
                                                      imu.linear_acceleration.z)
                                                      *ACC_SCALE
                                            ));
        }

        if (regularize_timestamp)
        {
            if (imu_sequence.size() <= 2)
                return;

            double t0 = imu_sequence.front().t;
            double tK = imu_sequence.back().t;

            double dt = (tK - t0)/(imu_sequence.size() - 1);
            
            for(int i = 0; i < imu_sequence.size(); i++)
                imu_sequence[i].t = t0 + dt*i;
        }
    }

    // Complete deskew with downsample
    void DeskewByImu(const deque<ImuProp> &imuProp, const deque<TimeSegment> timeSeg,
                     const CloudXYZITPtr &inCloud, CloudXYZIPtr &outCloud, CloudXYZIPtr &outCloudDS,
                     double ds_radius)
    {
        if (!fuse_imu)
        {
            *outCloud = toCloudXYZI(*inCloud);
            
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(ds_radius);
            downsampler.setInputCloud(outCloud);
            downsampler.filter(*outCloudDS);

            return;
        }

        int cloud_size = inCloud->size();
        outCloud->resize(cloud_size);
        outCloudDS->resize(cloud_size);

        const double &start_time = timeSeg.front().start_time;
        const double &final_time = timeSeg.back().final_time;
        
        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < cloud_size; i++)
        {
            auto &inPoint = inCloud->points[i];

            double ts = inPoint.t;
            
            // Find the corresponding subsegment
            int seg_idx = -1;
            for(int j = 0; j < timeSeg.size(); j++)
            {
                if(timeSeg[j].start_time <= ts && ts <= timeSeg[j].final_time)
                {
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].start_time - 1.0e-6 <= ts && ts < timeSeg[j].start_time)
                {
                    ts = timeSeg[j].start_time;
                    inPoint.t = start_time;
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].final_time < ts && ts <= timeSeg[j].final_time - 1.0e-6)
                {
                    ts = timeSeg[j].final_time;
                    inPoint.t = final_time;
                    seg_idx = j;
                    break;
                }
            }

            if(seg_idx == -1)
            {
                printf(KYEL "Point time %f not in segment: [%f, %f]. Discarding\n" RESET, ts, start_time, final_time);
                outCloud->points[i].x = 0; outCloud->points[i].y = 0; outCloud->points[i].z = 0;
                outCloud->points[i].intensity = 0;
                // outCloud->points[i].t = -1; // Mark this point as invalid
                continue;
            }

            // Transform all points to the end of the scan
            myTf T_Bk_Bs = imuProp.back().getBackTf().inverse()*imuProp[seg_idx].getTf(ts);

            Vector3d point_at_end_time = T_Bk_Bs.rot * Vector3d(inPoint.x, inPoint.y, inPoint.z) + T_Bk_Bs.pos;

            outCloud->points[i].x = point_at_end_time.x();
            outCloud->points[i].y = point_at_end_time.y();
            outCloud->points[i].z = point_at_end_time.z();
            outCloud->points[i].intensity = inPoint.intensity;
            // outCloud->points[i].t = inPoint.t;

            outCloudDS->points[i] = outCloud->points[i];
            outCloudDS->points[i].intensity = i;
        }

        // Downsample the pointcloud
        static int step_time;
        static int step_scale;
        // Reset the scale if the time has elapsed
        if (step_time == -1 || timeSeg.front().start_time - step_time > 5.0)
        {
            step_time = timeSeg.front().start_time;
            step_scale = 0;
        }

        if (ds_radius > 0.0)
        {
            int ds_scale = step_scale;
            CloudXYZIPtr tempDSCloud(new CloudXYZI);
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setInputCloud(outCloudDS);
            
            while(true)
            {
                double ds_effective_radius = ds_radius/(std::pow(2, ds_scale));

                downsampler.setRadiusSearch(ds_effective_radius);
                downsampler.setInputCloud(outCloudDS);
                downsampler.filter(*tempDSCloud);

                // If downsampled pointcloud has too few points, relax the ds_radius
                if(tempDSCloud->size() >= 2*max_lidar_factor/WINDOW_SIZE
                    || tempDSCloud->size() == outCloudDS->size()
                    || ds_effective_radius < leaf_size)
                {
                    outCloudDS = tempDSCloud;
                    break;
                }
                else
                {
                    printf(KYEL "Effective assoc_spacing: %f. Points: %d -> %d. Too few points. Relaxing assoc_spacing...\n" RESET,
                                 ds_effective_radius, outCloudDS->size(), tempDSCloud->size());
                    ds_scale++;
                    continue;
                }
            }
            
            if (ds_scale != step_scale)
            {
                step_scale = ds_scale;
                step_time = timeSeg.front().start_time;
            }
        }
    }

    // Only deskew the associated set
    void Redeskew(const deque<ImuProp> &imuProp, const deque<TimeSegment> timeSeg,
                  const CloudXYZITPtr  &inCloud, CloudXYZIPtr &outCloudDS)
    {
        const double &start_time = timeSeg.front().start_time;
        const double &final_time = timeSeg.back().final_time;
      
        int cloud_size = outCloudDS->size();
        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < cloud_size; i++)
        {
            int point_idx = (int)(outCloudDS->points[i].intensity);

            PointXYZIT &pointRaw = inCloud->points[point_idx];
            double ts = pointRaw.t;
            if (ts < 0)
                continue;
            
            // Find the corresponding subsegment
            int seg_idx = -1;
            for(int j = 0; j < timeSeg.size(); j++)
            {
                if(timeSeg[j].start_time <= ts && ts <= timeSeg[j].final_time)
                {
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].start_time - 1.0e-6 <= ts && ts < timeSeg[j].start_time)
                {
                    ts = timeSeg[j].start_time;
                    // coef.t_ = start_time;
                    seg_idx = j;
                    break;
                }
                else if (timeSeg[j].final_time < ts && ts <= timeSeg[j].final_time - 1.0e-6)
                {
                    ts = timeSeg[j].final_time;
                    // coef.t_ = final_time;
                    seg_idx = j;
                    break;
                }
            }

            if(seg_idx == -1)
            {
                printf(KYEL "Point time %f not in segment: [%f, %f]. Discarding\n" RESET, ts, start_time, final_time);
                outCloudDS->points[i].x = 0; outCloudDS->points[i].y = 0; outCloudDS->points[i].z = 0;
                outCloudDS->points[i].intensity = 0;
                // outCloud->points[i].t = -1; // Mark this point as invalid
                continue;
            }

            // Transform all points to the end of the scan
            myTf T_Bk_Bs = imuProp.back().getBackTf().inverse()*imuProp[seg_idx].getTf(ts);

            Vector3d point_at_end_time = T_Bk_Bs.rot * Vector3d(pointRaw.x, pointRaw.y, pointRaw.z) + T_Bk_Bs.pos;

            outCloudDS->points[i].x = point_at_end_time.x();
            outCloudDS->points[i].y = point_at_end_time.y();
            outCloudDS->points[i].z = point_at_end_time.z();
            // outCloud->points[i].intensity = inPoint.intensity;
            // outCloud->points[i].t = inPoint.t;

            // outCloudDS->points[i] = outCloud->points[i];
            // outCloudDS->points[i].intensity = i;
        }
    }

    void DeskewBySpline(PoseSplineX &traj, const deque<TimeSegment> timeSeg,
                        const CloudXYZITPtr &inCloud, CloudXYZIPtr &outCloud, CloudXYZIPtr &outCloudDS,
                        double ds_radius)
    {
        if (!fuse_imu)
        {
            *outCloud = toCloudXYZI(*inCloud);
            
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(ds_radius);
            downsampler.setInputCloud(outCloud);
            downsampler.filter(*outCloudDS);

            return;
        }

        int cloud_size = inCloud->size();
        outCloud->resize(cloud_size);
        outCloudDS->resize(cloud_size);

        const double &start_time = timeSeg.front().start_time;
        const double &final_time = timeSeg.back().final_time;

        #pragma omp parallel for num_threads(MAX_THREADS)
        for (int i = 0; i < cloud_size; i++)
        {
            auto &inPoint = inCloud->points[i];

            double ts = inPoint.t;

            if(!TimeIsValid(traj, ts, 1e-6))
            {
                printf(KYEL "Point time %f not in segment: [%f, %f]. Discarding\n" RESET, ts, start_time, final_time);
                outCloud->points[i].x = 0; outCloud->points[i].y = 0; outCloud->points[i].z = 0;
                outCloud->points[i].intensity = 0;
                // outCloud->points[i].t = -1; // Mark this point as invalid
                continue;
            }

            // Transform all points to the end of the scan
            SE3d pose_Bk_Bs = traj.pose(final_time).inverse()*traj.pose(ts);
            myTf T_Bk_Bs(pose_Bk_Bs.so3().unit_quaternion(), pose_Bk_Bs.translation());

            Vector3d point_at_end_time = T_Bk_Bs.rot * Vector3d(inPoint.x, inPoint.y, inPoint.z) + T_Bk_Bs.pos;

            outCloud->points[i].x = point_at_end_time.x();
            outCloud->points[i].y = point_at_end_time.y();
            outCloud->points[i].z = point_at_end_time.z();
            outCloud->points[i].intensity = inPoint.intensity;
            // outCloud->points[i].t = inPoint.t;

            outCloudDS->points[i] = outCloud->points[i];
            // outCloudDS->points[i].intensity = i;
        }

        if (ds_radius > 0.0)
        {
            int ds_scale = 0;
            CloudXYZIPtr tempDSCloud(new CloudXYZI);
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setInputCloud(outCloudDS);
            
            while(true)
            {
                double ds_effective_radius = ds_radius/(std::pow(2, ds_scale));

                downsampler.setRadiusSearch(ds_effective_radius);
                downsampler.setInputCloud(outCloudDS);
                downsampler.filter(*tempDSCloud);

                // If downsampled pointcloud has too few points, relax the ds_radius
                if(tempDSCloud->size() >= 2*max_lidar_factor/WINDOW_SIZE
                    || tempDSCloud->size() == outCloudDS->size()
                    || ds_effective_radius < leaf_size)
                {
                    outCloudDS = tempDSCloud;
                    break;
                }
                else
                {
                    printf(KYEL "Effective assoc_spacing: %f. Points: %d -> %d. Too few points. Relaxing assoc_spacing...\n" RESET,
                                 ds_effective_radius, tempDSCloud->size(), outCloudDS->size());
                    ds_scale++;
                    continue;
                }
            }
        }
    }

    void FitSpline()
    {
        tt_fitspline.Tic();

        // Create a local spline to store the new knots, isolating the poses from the global trajectory
        PoseSplineX SwTraj(SPLINE_N, deltaT);
        int swBaseKnot = GlobalTraj->computeTIndex(SwTimeStep.front().front().start_time).second;

        double swStartTime = GlobalTraj->getKnotTime(swBaseKnot);
        double swFinalTime = SwTimeStep.back().back().final_time;

        SwTraj.setStartTime(swStartTime);
        SwTraj.extendKnotsTo(swFinalTime, SE3d());

        // Copy the knots value
        for(int knot_idx = swBaseKnot; knot_idx < GlobalTraj->numKnots(); knot_idx++)
            SwTraj.setKnot(GlobalTraj->getKnot(knot_idx), knot_idx - swBaseKnot);

        PoseSplineX &traj = SwTraj;

        // Create and solve the Ceres Problem
        ceres::Problem problem;
        ceres::Solver::Options options;

        // Set up the options
        options.linear_solver_type                = linSolver;
        options.trust_region_strategy_type        = trustRegType;
        options.dense_linear_algebra_library_type = linAlgbLib;
        options.max_num_iterations                = max_iterations;
        options.max_solver_time_in_seconds        = max_solve_time;
        options.num_threads                       = MAX_THREADS;
        options.minimizer_progress_to_stdout      = false;

        ceres::LocalParameterization *local_parameterization = new LieAnalyticLocalParameterization<SO3d>();

        // Number of knots of the spline
        int KNOTS = traj.numKnots();

        // Add the parameter blocks for rotational knots
        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            problem.AddParameterBlock(traj.getKnotSO3(knot_idx).data(), 4, local_parameterization);

        // Add the parameter blocks for positional knots
        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            problem.AddParameterBlock(traj.getKnotPos(knot_idx).data(), 3);

        // Add the parameters for imu biases
        double *BIAS_G = new double[3];
        double *BIAS_A = new double[3];

        BIAS_G[0] = sfBig.back().back().x(); BIAS_A[0] = sfBia.back().back().x();
        BIAS_G[1] = sfBig.back().back().y(); BIAS_A[1] = sfBia.back().back().y();
        BIAS_G[2] = sfBig.back().back().z(); BIAS_A[2] = sfBia.back().back().z();

        problem.AddParameterBlock(BIAS_G, 3);
        problem.AddParameterBlock(BIAS_A, 3);

        for(int i = 0; i < 3; i++)
        {
            if(BG_BOUND(i) > 0)
            {
                problem.SetParameterLowerBound(BIAS_G, i, -BG_BOUND(i));
                problem.SetParameterUpperBound(BIAS_G, i,  BG_BOUND(i));
            }

            if(BA_BOUND(i) > 0)
            {
                problem.SetParameterLowerBound(BIAS_A, i, -BA_BOUND(i));
                problem.SetParameterUpperBound(BIAS_A, i,  BA_BOUND(i));
            }
        }

        for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
        {
            if (traj.getKnotTime(knot_idx) <= SwTimeStep.rbegin()[1].back().final_time)
            {
                problem.SetParameterBlockConstant(traj.getKnotSO3(knot_idx).data());
                problem.SetParameterBlockConstant(traj.getKnotPos(knot_idx).data());
            }
        }

        // Fit the spline with pose and IMU measurements
        vector<ceres::internal::ResidualBlock *> res_ids_poseprop;
        for(int i = WINDOW_SIZE-1; i < WINDOW_SIZE; i++)
        {
            for(int j = 0; j < SwPropState[i].size(); j++)
            {
                for (int k = 0; k < SwPropState[i][j].size()-1; k++)
                {
                    double sample_time = SwPropState[i][j].t[k];

                    // Continue if sample is out of the window
                    if (!traj.TimeIsValid(sample_time, 1e-6))
                        continue;

                    auto   us = traj.computeTIndex(sample_time);
                    double u  = us.first;
                    int    s  = us.second;

                    // Pose
                    ceres::CostFunction *cost_function
                        = new PoseAnalyticFactor
                                (myTf(SwPropState[i][j].Q[k], SwPropState[i][j].P[k]).getSE3(), POSE_N, POSE_N, SPLINE_N, traj.getDt(), u);

                    // Find the coupled poses
                    vector<double *> factor_param_blocks;
                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                    auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
                    res_ids_poseprop.push_back(res_block);
                }
            }
        }

        // Add the IMU
        vector<ceres::internal::ResidualBlock *> res_ids_pimu;
        for(int i = WINDOW_SIZE-1; i < WINDOW_SIZE; i++)
        {
            for(int j = 0; j < N_SUB_SEG; j++)
            {
                for(int k = 1; k < SwImuBundle[i][j].size(); k++)
                {
                    double sample_time = SwImuBundle[i][j][k].t;

                    // Skip if sample time exceeds the bound
                    if (!traj.TimeIsValid(sample_time, 1e-6))
                        continue;

                    auto imuBias = ImuBias(Vector3d(BIAS_G[0], BIAS_G[1], BIAS_G[2]),
                                           Vector3d(BIAS_A[0], BIAS_A[1], BIAS_A[2]));

                    auto   us = traj.computeTIndex(sample_time);
                    double u  = us.first;
                    int    s  = us.second;

                    double gyro_weight = GYR_N;
                    double acce_weight = ACC_N;
                    double bgyr_weight = GYR_W;
                    double bacc_weight = ACC_W;

                    ceres::CostFunction *cost_function =
                        new GyroAcceBiasAnalyticFactor
                            (SwImuBundle[i][j][k], imuBias, GRAV, gyro_weight, acce_weight, bgyr_weight, bacc_weight, SPLINE_N, traj.getDt(), u);

                    // Find the coupled poses
                    vector<double *> factor_param_blocks;
                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                    for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                    // gyro bias
                    factor_param_blocks.emplace_back(BIAS_G);

                    // acce bias
                    factor_param_blocks.emplace_back(BIAS_A);

                    // printf("Creating functor: u: %f, s: %d. sample: %d / %d\n", u, s, sample_idx, pose_gt.size());

                    // cost_function->SetNumResiduals(12);
                    ceres::LossFunction* loss_function = imu_loss_thres < 0 ? NULL : new ceres::CauchyLoss(imu_loss_thres);
                    auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
                    res_ids_pimu.push_back(res_block);    
                }
            }
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Copy the new knots back to the global trajectory
        for(int knot_idx = 0; knot_idx < SwTraj.numKnots(); knot_idx++)
            GlobalTraj->setKnot(SwTraj.getKnot(knot_idx), knot_idx + swBaseKnot);
        
        delete BIAS_G; delete BIAS_A;

        tt_fitspline.Toc();    
    }
    
    void LIOOptimization(slict::OptStat &report, string &lioop_times_report, PoseSplineX &traj,
                         map<int, int> &prev_knot_x, map<int, int> &curr_knot_x, int swNextBase, int iter,
                         vector<ImuIdx> &imuSelected,vector<lidarFeaIdx> &featureSelected, slict::TimeLog &tlog)
    {

        // Create the states for the bias
        Vector3d XBIG(sfBig.back().back());
        Vector3d XBIA(sfBia.back().back());

        // Create a solver
        static mySolver ms(nh_ptr);
        string iekf_report = "";
        bool ms_success = false;

        // Solve the least square problem
        if (!use_ceres)
            ms_success = ms.Solve(traj, XBIG, XBIA, prev_knot_x, curr_knot_x, swNextBase, iter,
                                  SwImuBundle, SwCloudDskDS, SwLidarCoef,
                                  imuSelected, featureSelected, iekf_report, report, tlog);

        if (ms_success)
        {
            struct Loader
            {
                void CopyParamToState(double t, PoseSplineX &traj, double *ba, double *bg, Vector3d &BAMAX, Vector3d &BGMAX,
                                      Vector3d &p_, Quaternd &q_, Vector3d &v_, Vector3d &ba_, Vector3d &bg_)
                {

                    if (t < traj.minTime() + 1e-06)
                    {
                        // printf("State time is earlier than SW time: %f < %f\n", t, traj.minTime());
                        t = traj.minTime() + 1e-06;
                    }

                    if (t > traj.maxTime() - 1e-06)
                    {
                        // printf("State time is later than SW time: %f > %f\n", t, traj.maxTime());
                        t = traj.maxTime() - 1e-06;
                    }

                    SE3d pose = traj.pose(t);

                    p_ = pose.translation();
                    q_ = pose.so3().unit_quaternion();

                    v_ = traj.transVelWorld(t);

                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(ba[i]) > BAMAX[i])
                        {
                            ba_(i) = ba[i]/fabs(ba[i])*BAMAX[i];
                            break;
                        }
                        else
                            ba_(i) = ba[i];
                    }

                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(bg[i]) > BGMAX[i])
                        {
                            bg_(i) = bg[i]/fabs(bg[i])*BGMAX[i];
                            break;
                        }
                        else
                            bg_(i) = bg[i];
                    }

                    // printf("Bg: %f, %f, %f -> %f, %f, %f\n", bg[0], bg[1], bg[2], bg_.x(), bg_.y(), bg_.z());
                    // printf("Ba: %f, %f, %f -> %f, %f, %f\n", ba[0], ba[1], ba[2], ba_.x(), ba_.y(), ba_.z());
                }

            } loader;

            // Load values from params to state
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                for(int j = 0; j < SwTimeStep[i].size(); j++)
                {
                    // Load the state at the start time of each segment
                    double ss_time = SwTimeStep[i][j].start_time;
                    loader.CopyParamToState(ss_time, traj, XBIA.data(), XBIG.data(), BA_BOUND, BG_BOUND,
                                            ssPos[i][j], ssQua[i][j], ssVel[i][j], ssBia[i][j], ssBig[i][j]);    

                    // Load the state at the final time of each segment
                    double sf_time = SwTimeStep[i][j].final_time;
                    loader.CopyParamToState(sf_time, traj, XBIA.data(), XBIG.data(), BA_BOUND, BG_BOUND,
                                            sfPos[i][j], sfQua[i][j], sfVel[i][j], sfBia[i][j], sfBig[i][j]);

                    // printf("Vel %f: %.2f, %.2f, %.2f\n", sf_time, sfVel[i][j].x(), sfVel[i][j].y(), sfVel[i][j].z());
                }
            }
        }

        if(!ms_success)
        {
/* #region */ TicToc tt_buildceres;

/* #region */ TicToc tt_create;

            // Create and solve the Ceres Problem
            ceres::Problem problem;
            ceres::Solver::Options options;

            // Set up the options
            options.minimizer_type                    = ceres::TRUST_REGION;
            options.linear_solver_type                = linSolver;
            options.trust_region_strategy_type        = trustRegType;
            options.dense_linear_algebra_library_type = linAlgbLib;
            options.max_num_iterations                = max_iterations;
            options.max_solver_time_in_seconds        = max_solve_time;
            options.num_threads                       = MAX_THREADS;
            options.minimizer_progress_to_stdout      = false;
            options.use_nonmonotonic_steps            = true;

            ceres::LocalParameterization *local_parameterization = new LieAnalyticLocalParameterization<SO3d>();

            // Number of knots of the spline
            int KNOTS = traj.numKnots();

            // Add the parameter blocks for rotational knots
            for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
                problem.AddParameterBlock(traj.getKnotSO3(knot_idx).data(), 4, local_parameterization);

            // Add the parameter blocks for positional knots
            for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
                problem.AddParameterBlock(traj.getKnotPos(knot_idx).data(), 3);

            // Add the parameters for imu biases
            double *BIAS_G = XBIG.data();
            double *BIAS_A = XBIA.data();

            // BIAS_G[0] = sfBig.back().back().x(); BIAS_A[0] = sfBia.back().back().x();
            // BIAS_G[1] = sfBig.back().back().y(); BIAS_A[1] = sfBia.back().back().y();
            // BIAS_G[2] = sfBig.back().back().z(); BIAS_A[2] = sfBia.back().back().z();

            problem.AddParameterBlock(BIAS_G, 3);
            problem.AddParameterBlock(BIAS_A, 3);

            // for(int i = 0; i < 3; i++)
            // {
            //     if(BG_BOUND[i] > 0)
            //     {
            //         problem.SetParameterLowerBound(BIAS_G, i, -BG_BOUND[i]);
            //         problem.SetParameterUpperBound(BIAS_G, i,  BG_BOUND[i]);
            //     }

            //     if(BA_BOUND[i] > 0)
            //     {
            //         problem.SetParameterLowerBound(BIAS_A, i, -BA_BOUND[i]);
            //         problem.SetParameterUpperBound(BIAS_A, i,  BA_BOUND[i]);
            //     }
            // }

            // Fix the fist and the last N knots
            first_fixed_knot = -1;
            last_fixed_knot = -1;
            for (int knot_idx = 0; knot_idx < KNOTS; knot_idx++)
            {
                if (
                    traj.getKnotTime(knot_idx) <= traj.minTime() + start_fix_span
                    // || traj.getKnotTime(knot_idx) > traj.getKnotTime(KNOTS-1) - final_fix_span
                )
                {
                    if(first_fixed_knot == -1)
                        first_fixed_knot = knot_idx;

                    last_fixed_knot = knot_idx;
                    problem.SetParameterBlockConstant(traj.getKnotSO3(knot_idx).data());
                    problem.SetParameterBlockConstant(traj.getKnotPos(knot_idx).data());
                }
            }

/* #endregion */ tt_create.Toc();

/* #region */ TicToc tt_addlidar;

            // Cloud to show points being associated
            CloudXYZIPtr assocCloud(new CloudXYZI());

            // Add the lidar factors
            vector<ceres::internal::ResidualBlock *> res_ids_surf;
            double cost_surf_init = -1, cost_surf_final = -1;
            if(fuse_lidar)
            {
                // Shared loss function
                ceres::LossFunction *lidar_loss_function = lidar_loss_thres == -1 ? NULL : new ceres::CauchyLoss(lidar_loss_thres);
                int factor_idx = 0;

                // Find and mark the used factors
                // for (int i = 0; i < WINDOW_SIZE; i++)
                // {
                // #pragma omp parallel for num_threads(MAX_THREADS)
                for (int j = 0; j < featureSelected.size(); j++)
                {   
                    int  i = featureSelected[j].wdidx;
                    int  k = featureSelected[j].pointidx;
                    int  depth = featureSelected[j].depth;

                    auto &point = SwCloudDskDS[i]->points[k];
                    int  point_idx = (int)(point.intensity);
                    int  coeff_idx = k;

                    const LidarCoef &coef = SwLidarCoef[i][coeff_idx];
                    // ROS_ASSERT_MSG(coef.t >= 0, "i = %d, k = %d, t = %f", i, k, coef.t);
                    double sample_time = coef.t;
                    // ROS_ASSERT(traj.TimeIsValid(sample_time, 1e-6));
                    auto   us = traj.computeTIndex(sample_time);
                    double u  = us.first;
                    int    s  = us.second;
                    int base_knot = s;
                    vector<double*> factor_param_blocks;
                    // Add the parameter blocks for rotation
                    for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                        factor_param_blocks.push_back(traj.getKnotSO3(knot_idx).data());
                    // Add the parameter blocks for position
                    for (int knot_idx = base_knot; knot_idx < base_knot + SPLINE_N; knot_idx++)
                        factor_param_blocks.push_back(traj.getKnotPos(knot_idx).data());
                    // Shared associate settings
                    factor_idx++;
                    assocSettings settings(use_ufm, reassoc_rate > 0, reassoc_rate,
                                           surfel_min_point, surfel_min_plnrty, surfel_intsect_rad, dis_to_surfel_max,
                                           lidar_weight, i*100000 + factor_idx);
                    // Add the residual
                    typedef PointToPlaneAnalyticFactor<PredType> p2pFactor;
                    auto res = problem.AddResidualBlock(
                                new p2pFactor(coef.finW, coef.f, coef.n, coef.plnrty*lidar_weight,
                                              SPLINE_N, traj.getDt(), u, activeSurfelMap, *commonPred, activeikdtMap, settings),
                                              lidar_loss_function, factor_param_blocks);
                    res_ids_surf.push_back(res);    
                    // Add point to visualization
                    PointXYZI pointInW; pointInW.x = coef.finW.x(); pointInW.y = coef.finW.y(); pointInW.z = coef.finW.z();
                    assocCloud->push_back(pointInW);
                    assocCloud->points.back().intensity = i;
                }
                // }
            }

/* #endregion */ tt_addlidar.Toc();
        
/* #region */ TicToc tt_addimu;

            // Create and add the new preintegration factors
            vector<ceres::internal::ResidualBlock *> res_ids_pimu;
            double cost_pimu_init = -1, cost_pimu_final = -1;
            // deque<deque<PreintBase *>> local_preints(WINDOW_SIZE, deque<PreintBase *>(N_SUB_SEG));
            if(fuse_imu)
            {
                ceres::LossFunction* loss_function = imu_loss_thres < 0 ? NULL : new ceres::CauchyLoss(imu_loss_thres);

                for(int i = 0; i < WINDOW_SIZE; i++)
                {
                    for(int j = 0; j < N_SUB_SEG; j++)
                    {
                        for(int k = 1; k < SwImuBundle[i][j].size(); k++)
                        {
                            double sample_time = SwImuBundle[i][j][k].t;
                            
                            // Skip if sample time exceeds the bound
                            if (!traj.TimeIsValid(sample_time, 1e-6))
                                continue;

                            auto imuBias = ImuBias(Vector3d(BIAS_G[0], BIAS_G[1], BIAS_G[2]),
                                                Vector3d(BIAS_A[0], BIAS_A[1], BIAS_A[2]));
        
                            auto   us = traj.computeTIndex(sample_time);
                            double u  = us.first;
                            int    s  = us.second;

                            double gyro_weight = GYR_N;
                            double acce_weight = ACC_N;
                            double bgyr_weight = GYR_W;
                            double bacc_weight = ACC_W;

                            ceres::CostFunction *cost_function =
                                new GyroAcceBiasAnalyticFactor
                                    (SwImuBundle[i][j][k], imuBias, GRAV, gyro_weight, acce_weight, bgyr_weight, bacc_weight, SPLINE_N, traj.getDt(), u);

                            // Find the coupled poses
                            vector<double *> factor_param_blocks;
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                            // gyro bias
                            factor_param_blocks.emplace_back(BIAS_G);

                            // acce bias
                            factor_param_blocks.emplace_back(BIAS_A);

                            // cost_function->SetNumResiduals(12);
                            auto res_block = problem.AddResidualBlock(cost_function, loss_function, factor_param_blocks);
                            res_ids_pimu.push_back(res_block);    
                        }
                    }
                }
            }

/* #endregion */ tt_addimu.Toc();

/* #region */ TicToc tt_addpp;

            vector<ceres::internal::ResidualBlock *> res_ids_poseprop;
            double cost_poseprop_init = -1, cost_poseprop_final = -1;
            if(fuse_poseprop)
            {
                // Add the poses
                for(int i = 0; i < WINDOW_SIZE - 1 - reassociate_steps; i++)
                {
                    for(int j = 0; j < SwPropState[i].size(); j++)
                    {
                        for (int k = 0; k < SwPropState[i][j].size()-1; k++)
                        {
                            double sample_time = SwPropState[i][j].t[k];

                            // Continue if sample is in the window
                            if (!traj.TimeIsValid(sample_time, 1e-6) || sample_time > traj.minTime() + 0.1)
                                continue;

                            auto   us = traj.computeTIndex(sample_time);
                            double u  = us.first;
                            int    s  = us.second;

                            // Pose
                            ceres::CostFunction *cost_function
                                = new PoseAnalyticFactor
                                        (myTf(SwPropState[i][j].Q[k], SwPropState[i][j].P[k]).getSE3(), POSE_N, POSE_N, SPLINE_N, traj.getDt(), u);

                            // Find the coupled poses
                            vector<double *> factor_param_blocks;
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotSO3(knot_idx).data());

                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                            auto res_block = problem.AddResidualBlock(cost_function, NULL, factor_param_blocks);
                            res_ids_poseprop.push_back(res_block);
                        }
                    }
                }
            }

            vector<ceres::internal::ResidualBlock *> res_ids_velprop;
            double cost_velprop_init = -1, cost_velprop_final = -1;
            if(fuse_velprop)
            {
                // Add the velocity
                for(int i = 0; i < 1; i++)
                {
                    for(int j = 0; j < SwPropState[i].size(); j++)
                    {
                        for (int k = 0; k < SwPropState[i][j].size()-1; k++)
                        {
                            double sample_time = SwPropState[i][j].t[k];

                            // Continue if sample is in the window
                            if (sample_time < traj.minTime() + 1.0e-6 || sample_time > traj.maxTime() - 1.0e-6)
                                continue;

                            auto   us = traj.computeTIndex(sample_time);
                            double u  = us.first;
                            int    s  = us.second;

                            double &vel_weight = VEL_N;

                            // Velocity
                            ceres::CostFunction *vel_cost_function
                                = new VelocityAnalyticFactor(SwPropState[i][j].V[k], vel_weight, SPLINE_N, traj.getDt(), u);

                            // Find the coupled poses
                            vector<double *> factor_param_blocks;
                            for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                                factor_param_blocks.emplace_back(traj.getKnotPos(knot_idx).data());

                            auto res_block = problem.AddResidualBlock(vel_cost_function, NULL, factor_param_blocks);
                            res_ids_velprop.push_back(res_block);
                        }
                    }
                }
            }

/* #endregion */ tt_addpp.Toc();
            
/* #region */ TicToc tt_init_cost;

            if(find_factor_cost)
            {
                Util::ComputeCeresCost(res_ids_surf, cost_surf_init, problem);
                Util::ComputeCeresCost(res_ids_pimu, cost_pimu_init, problem);
                Util::ComputeCeresCost(res_ids_poseprop, cost_poseprop_init, problem);
                Util::ComputeCeresCost(res_ids_velprop, cost_velprop_init, problem);
            }

/* #endregion */ tt_init_cost.Toc();
            
/* #endregion */ tt_buildceres.Toc(); tlog.t_prep.push_back(tt_buildceres.GetLastStop());

/* #region */ TicToc tt_solve;

            if (ensure_real_time)
            {
                t_slv_budget = max(50.0, sweep_len * 95 - (tt_preopt.GetLastStop() + tt_buildceres.GetLastStop()));
                if (packet_buf.size() > 0)
                    t_slv_budget = 50.0;
                options.max_solver_time_in_seconds = t_slv_budget/1000.0;
            }
            else
                t_slv_budget = options.max_solver_time_in_seconds*1000;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

/* #endregion */ tt_solve.Toc(); tlog.t_compute.push_back(tt_solve.GetLastStop());

/* #region */ TicToc tt_aftsolve;

/* #region */ TicToc tt_final_cost;
            if(find_factor_cost)
            {
                Util::ComputeCeresCost(res_ids_surf, cost_surf_final, problem);
                Util::ComputeCeresCost(res_ids_pimu, cost_pimu_final, problem);
                Util::ComputeCeresCost(res_ids_poseprop, cost_poseprop_final, problem);
                Util::ComputeCeresCost(res_ids_velprop, cost_velprop_final, problem);
            }
/* #endregion */ tt_final_cost.Toc();

/* #region  */ TicToc tt_load;

            struct Loader
            {
                // void CopyStateToParam(Vector3d &p_, Quaternd &q_, Vector3d &v_,
                //                       Vector3d &ba, Vector3d &bg,
                //                       double *&pose, double *&velo, double *&bias)
                // {
                //     pose[0] = p_.x(); pose[1] = p_.y(); pose[2] = p_.z();
                //     pose[3] = q_.x(); pose[4] = q_.y(); pose[5] = q_.z(); pose[6] = q_.w();

                //     velo[0] = v_.x(); velo[1] = v_.y(); velo[2] = v_.z();
                    
                //     bias[0] = ba.x(); bias[1] = ba.y(); bias[2] = ba.z();
                //     bias[3] = bg.x(); bias[4] = bg.y(); bias[5] = bg.z();
                // }

                void CopyParamToState(double t, PoseSplineX &traj, double *&ba, double *&bg, Vector3d &BAMAX, Vector3d &BGMAX,
                                    Vector3d &p_, Quaternd &q_, Vector3d &v_, Vector3d &ba_, Vector3d &bg_)
                {

                    if (t < traj.minTime() + 1e-06)
                    {
                        // printf("State time is earlier than SW time: %f < %f\n", t, traj.minTime());
                        t = traj.minTime() + 1e-06;
                    }

                    if (t > traj.maxTime() - 1e-06)
                    {
                        // printf("State time is later than SW time: %f > %f\n", t, traj.maxTime());
                        t = traj.maxTime() - 1e-06;
                    }

                    SE3d pose = traj.pose(t);

                    p_ = pose.translation();
                    q_ = pose.so3().unit_quaternion();

                    v_ = traj.transVelWorld(t);

                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(ba[i]) > BAMAX[i])
                        {
                            ba_ = Vector3d(0, 0, 0);
                            break;
                        }
                        else
                            ba_(i) = ba[i];
                    }

                    for (int i = 0; i < 3; i++)
                    {
                        if (fabs(bg[i]) > BGMAX[i])
                        {
                            bg_ = Vector3d(0, 0, 0);
                            break;
                        }
                        else
                            bg_(i) = bg[i];
                    }

                    // printf("Bg: %f, %f, %f -> %f, %f, %f\n", bg[0], bg[1], bg[2], bg_.x(), bg_.y(), bg_.z());
                    // printf("Ba: %f, %f, %f -> %f, %f, %f\n", ba[0], ba[1], ba[2], ba_.x(), ba_.y(), ba_.z());
                }

            } loader;

            // Load values from params to state
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                for(int j = 0; j < SwTimeStep[i].size(); j++)
                {
                    // Load the state at the start time of each segment
                    double ss_time = SwTimeStep[i][j].start_time;
                    loader.CopyParamToState(ss_time, traj, BIAS_A, BIAS_G, BA_BOUND, BG_BOUND,
                                            ssPos[i][j], ssQua[i][j], ssVel[i][j], ssBia[i][j], ssBig[i][j]);    

                    // Load the state at the final time of each segment
                    double sf_time = SwTimeStep[i][j].final_time;
                    loader.CopyParamToState(sf_time, traj, BIAS_A, BIAS_G, BA_BOUND, BG_BOUND,
                                            sfPos[i][j], sfQua[i][j], sfVel[i][j], sfBia[i][j], sfBig[i][j]);

                    // printf("Vel %f: %.2f, %.2f, %.2f\n", sf_time, sfVel[i][j].x(), sfVel[i][j].y(), sfVel[i][j].z());
                }
            }
            
            // delete BIAS_G; delete BIAS_A;

/* #endregion */ tt_load.Toc();
            
/* #region Load data to the report */ TicToc tt_report;

            tlog.ceres_iter = summary.iterations.size();

            report.surfFactors = res_ids_surf.size();
            report.J0Surf = cost_surf_init;
            report.JKSurf = cost_surf_final;
            
            report.imuFactors = res_ids_pimu.size();
            report.J0Imu = cost_pimu_init;
            report.JKImu = cost_pimu_final;

            report.propFactors = res_ids_poseprop.size();
            report.J0Prop = cost_poseprop_init;
            report.JKProp = cost_poseprop_final;

            report.velFactors = res_ids_velprop.size();
            report.J0Vel = cost_velprop_init;      //cost_vel_init;
            report.JKVel = cost_velprop_final;     //cost_vel_final;

            report.J0 = summary.initial_cost;
            report.JK = summary.final_cost;
            
            report.Qest.x = sfQua.back().back().x();
            report.Qest.y = sfQua.back().back().y();
            report.Qest.z = sfQua.back().back().z();
            report.Qest.w = sfQua.back().back().w();

            report.Pest.x = sfPos.back().back().x();
            report.Pest.y = sfPos.back().back().y();
            report.Pest.z = sfPos.back().back().z();

            report.Vest.x = sfVel.back().back().x();
            report.Vest.y = sfVel.back().back().y();
            report.Vest.z = sfVel.back().back().z();

            report.Qimu.x = SwPropState.back().back().Q.back().x();
            report.Qimu.y = SwPropState.back().back().Q.back().y();
            report.Qimu.z = SwPropState.back().back().Q.back().z();
            report.Qimu.w = SwPropState.back().back().Q.back().w();

            report.Pimu.x = SwPropState.back().back().P.back().x();
            report.Pimu.y = SwPropState.back().back().P.back().y();
            report.Pimu.z = SwPropState.back().back().P.back().z();

            report.Vimu.x = SwPropState.back().back().V.back().x();
            report.Vimu.y = SwPropState.back().back().V.back().y();
            report.Vimu.z = SwPropState.back().back().V.back().z();

            // Calculate the relative pose to the last keyframe
            PointPose lastKf = KfCloudPose->back();
            myTf tf_W_Blast(lastKf);

            report.lastKfId = (int)(lastKf.intensity);
            myTf tf_Blast_Bcurr = tf_W_Blast.inverse()*myTf(sfQua.back().back(), sfPos.back().back());

            report.Qref.x = tf_Blast_Bcurr.rot.x();
            report.Qref.y = tf_Blast_Bcurr.rot.y();
            report.Qref.z = tf_Blast_Bcurr.rot.z();
            report.Qref.w = tf_Blast_Bcurr.rot.w();
            
            report.Pref.x = tf_Blast_Bcurr.pos.x();
            report.Pref.y = tf_Blast_Bcurr.pos.y();
            report.Pref.z = tf_Blast_Bcurr.pos.z();
            
            report.iters = summary.iterations.size();
            report.tbuildceres = tt_buildceres.GetLastStop();
            report.tslv  = tt_solve.GetLastStop();
            report.trun  = (ros::Time::now() - program_start_time).toSec();

            report.BANum            = baReport.turn;
            report.BAItr            = baReport.pgopt_iter;
            report.BALoopTime       = tt_loopBA.GetLastStop();
            report.BASolveTime      = baReport.pgopt_time;
            report.BARelPoseFactors = baReport.factor_relpose;
            report.BALoopFactors    = baReport.factor_loop;
            report.BAJ0             = baReport.J0;
            report.BAJK             = baReport.JK;
            report.BAJ0RelPose      = baReport.J0_relpose;
            report.BAJKRelPose      = baReport.JK_relpose;
            report.BAJ0Loop         = baReport.J0_loop;
            report.BAJKLoop         = baReport.JK_loop;

/* #endregion Load data to the report */ tt_report.Toc();

// /* #region */ TicToc tt_showassoc;

//             // Publish the assoc cloud
//             static ros::Publisher assoc_cloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/assoc_cloud", 100);
//             Util::publishCloud(assoc_cloud_pub, *assocCloud, ros::Time(SwTimeStep.back().back().final_time), current_ref_frame);

// /* #endregion */ tt_showassoc.Toc();

/* #endregion */ tt_aftsolve.Toc(); tlog.t_update.push_back(tt_aftsolve.GetLastStop());

            lioop_times_report = ""; 
            if(GetBoolParam("/show_lioop_times", false))
            {
                lioop_times_report += "lioop: ";
                lioop_times_report += "crt: "     + myprintf("%3.1f. ", tt_create.GetLastStop());
                lioop_times_report += "addldr: "  + myprintf("%4.1f. ", tt_addlidar.GetLastStop());
                lioop_times_report += "addimu: "  + myprintf("%3.1f. ", tt_addimu.GetLastStop());
                lioop_times_report += "addpp: "   + myprintf("%3.1f. ", tt_addpp.GetLastStop());
                lioop_times_report += "J0: "      + myprintf("%4.1f. ", tt_init_cost.GetLastStop());
                lioop_times_report += "bc: "      + myprintf("%4.1f. ", tt_buildceres.GetLastStop());
                lioop_times_report += "slv: "     + myprintf("%4.1f. ", tt_solve.GetLastStop());
                lioop_times_report += "JK: "      + myprintf("%4.1f. ", tt_final_cost.GetLastStop());
                lioop_times_report += "load: "    + myprintf("%3.1f. ", tt_load.GetLastStop());
                lioop_times_report += "rep: "     + myprintf("%3.1f. ", tt_load.GetLastStop());
                // lioop_times_report += "showasc: " + myprintf("%3.1f. ", tt_showassoc.GetLastStop());
                lioop_times_report += "aftslv: "  + myprintf("%3.1f, ", tt_aftsolve.GetLastStop());
                lioop_times_report += "\n";
            }
        }

        lioop_times_report += iekf_report;
    }

    void NominateKeyframe()
    {
        tt_margcloud.Tic();

        int mid_step = 0;//int(std::floor(WINDOW_SIZE/2.0));

        static double last_kf_time = SwTimeStep[mid_step].back().final_time;

        double kf_cand_time = SwTimeStep[mid_step].back().final_time;

        CloudPosePtr kfTempPose(new CloudPose());
        *kfTempPose = *KfCloudPose;

        static KdTreeFLANN<PointPose> kdTreeKeyFrames;
        kdTreeKeyFrames.setInputCloud(kfTempPose);

        myTf tf_W_Bcand(sfQua[mid_step].back(), sfPos[mid_step].back());
        PointPose kf_cand = tf_W_Bcand.Pose6D(kf_cand_time);

        int knn_nbrkf = min(10, (int)kfTempPose->size());
        vector<int> knn_idx(knn_nbrkf); vector<float> knn_sq_dis(knn_nbrkf);
        kdTreeKeyFrames.nearestKSearch(kf_cand, knn_nbrkf, knn_idx, knn_sq_dis);
        
        bool far_distance = knn_sq_dis.front() > kf_min_dis*kf_min_dis;
        bool far_angle = true;
        for(int i = 0; i < knn_idx.size(); i++)
        {
            int kf_idx = knn_idx[i];

            // Collect the angle difference
            Quaternionf Qa(kfTempPose->points[kf_idx].qw,
                           kfTempPose->points[kf_idx].qx,
                           kfTempPose->points[kf_idx].qy,
                           kfTempPose->points[kf_idx].qz);

            Quaternionf Qb(kf_cand.qw, kf_cand.qx, kf_cand.qy, kf_cand.qz);

            // If the angle is more than 10 degrees, add this to the key pose
            if (fabs(Util::angleDiff(Qa, Qb)) < kf_min_angle)
            {
                far_angle = false;
                break;
            }
        }
        bool kf_timeout = fabs(kf_cand_time - last_kf_time) > 2.0 && (knn_sq_dis.front() > 0.1*0.1);

        bool ikdtree_init = false;
        if(!use_ufm)
        {
            static int init_count = 20;
            if(init_count > 0)
            {
                init_count--;
                ikdtree_init = true;
            }
        }

        if (far_distance || far_angle || kf_timeout || ikdtree_init)
        {
            last_kf_time = kf_cand_time;

            static double leaf_sq = pow(leaf_size, 2);

            IOAOptions ioaOpt;
            IOASummary ioaSum;
            ioaSum.final_tf = tf_W_Bcand;
            CloudXYZIPtr marginalizedCloudInW(new CloudXYZI());

            if(refine_kf)
            {
                CloudXYZIPtr localMap(new CloudXYZI());
                // Merge the neighbour cloud
                for(int i = 0; i < knn_idx.size(); i++)
                {
                    int kf_idx = knn_idx[i];
                    *localMap += *KfCloudinW[kf_idx];
                }

                ioaOpt.init_tf = tf_W_Bcand;
                ioaOpt.max_iterations = ioa_max_iter;
                ioaOpt.show_report = false;
                ioaOpt.text = myprintf("Refine T_L_B(%d)_EST", KfCloudPose->size());

                CloudMatcher cm(0.1, 0.1);
                cm.IterateAssociateOptimize(ioaOpt, ioaSum, localMap, SwCloudDsk[mid_step]);

                printf("KF initial: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f.\n"
                       "KF refined: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f.\n",
                        ioaOpt.init_tf.pos.x(), ioaOpt.init_tf.pos.y(), ioaOpt.init_tf.pos.z(),
                        ioaOpt.init_tf.yaw(),   ioaOpt.init_tf.pitch(),
                        ioaOpt.init_tf.roll(),
                        ioaSum.final_tf.pos.x(),
                        ioaSum.final_tf.pos.y(),
                        ioaSum.final_tf.pos.z(),
                        ioaSum.final_tf.yaw(),
                        ioaSum.final_tf.pitch(),
                        ioaSum.final_tf.roll());

                tf_W_Bcand = ioaSum.final_tf;
            }

            if(reloc_stat != RELOCALIZED)
            {
                pcl::transformPointCloud(*SwCloudDsk[mid_step], *marginalizedCloudInW, tf_W_Bcand.cast<float>().tfMat());
            }
            else if(marginalize_new_points)
            {   
                // Skip a few keyframes
                static int count = 5;
                if (count > 0)
                    count--;
                else
                {
                    int new_nodes = 0;
                    int old_nodes = 0;

                    for(int i = 0; i < SwCloudDskDS[mid_step]->size(); i++)
                    {
                        int point_idx = (int)(SwCloudDskDS[mid_step]->points[i].intensity);
                        int coeff_idx = i;
                        if( SwLidarCoef[mid_step][coeff_idx].marginalized )
                        {
                            LidarCoef &coef = SwLidarCoef[mid_step][coeff_idx];
                            
                            ROS_ASSERT(point_idx == coef.ptIdx);

                            PointXYZI pointInB = SwCloudDsk[mid_step]->points[point_idx];
                            
                            PointXYZI pointInW = Util::transform_point(tf_W_Bcand, pointInB);
                            
                            marginalizedCloudInW->push_back(pointInW);
                        }
                    }
                }
            }

            // CloudXYZIPtr marginalizedCloud(new CloudXYZI());
            int margCount = marginalizedCloudInW->size();

            margPerc = double(margCount)/SwCloudDsk[mid_step]->size();
            AdmitKeyframe(SwTimeStep[mid_step].back().final_time, tf_W_Bcand.rot, tf_W_Bcand.pos,
                          SwCloudDsk[mid_step], marginalizedCloudInW);
        }

        tt_margcloud.Toc();
    }

    void AdmitKeyframe(double t, Quaternd q, Vector3d p, CloudXYZIPtr &cloud, CloudXYZIPtr &marginalizedCloudInW)
    {
        tt_ufoupdate.Tic();

        KfCloudinB.push_back(CloudXYZIPtr(new CloudXYZI()));
        KfCloudinW.push_back(CloudXYZIPtr(new CloudXYZI()));

        *KfCloudinB.back() = *cloud;
        pcl::transformPointCloud(*KfCloudinB.back(), *KfCloudinW.back(), p, q);

        KfCloudPose->push_back(myTf(q, p).Pose6D(t));
        KfCloudPose->points.back().intensity = KfCloudPose->size()-1;   // Use intensity to store keyframe id

        // for(int i = 0; i < KfCloudinB.size(); i++)
        //     printf("KF %d. Size: %d. %d\n",
        //             i, KfCloudinB[i]->size(), KfCloudinW[i]->size());

        // printf("Be4 add: GMap: %d.\n", globalMap->size(), KfCloudinW.back()->size());
        
        // Add keyframe pointcloud to global map
        {
            lock_guard<mutex> lock(global_map_mtx);
            *globalMap += *KfCloudinW.back();
        }

        // Add keyframe pointcloud to surfel map
        SendCloudToMapQueue(marginalizedCloudInW);

        // Filter global map
        if (KfCloudPose->size() > 1 && publish_map)
        {
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(leaf_size);
            downsampler.setInputCloud(globalMap);
            downsampler.filter(*globalMap);
        }

        static ros::Publisher margcloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/marginalized_cloud", 100);
        Util::publishCloud(margcloud_pub, *marginalizedCloudInW, ros::Time(t), current_ref_frame);

        sensor_msgs::PointCloud2 kfCloudROS
            = Util::publishCloud(kfcloud_pub, *KfCloudinW.back(), ros::Time(t), current_ref_frame);

        sensor_msgs::PointCloud2 kfPoseCloudROS
            = Util::publishCloud(kfpose_pub, *KfCloudPose, ros::Time(t), current_ref_frame);
        
        slict::FeatureCloud msg;
        msg.header.stamp = ros::Time(t);
        msg.pose.position.x = p.x();
        msg.pose.position.y = p.y();
        msg.pose.position.z = p.z();
        msg.pose.orientation.x = q.x();
        msg.pose.orientation.y = q.y();
        msg.pose.orientation.z = q.z();
        msg.pose.orientation.w = q.w();        
        msg.extracted_cloud = kfCloudROS;
        msg.edge_cloud = kfPoseCloudROS;
        msg.scanStartTime = t;
        msg.scanEndTime = t;
        kfcloud_std_pub.publish(msg);

        if (publish_map)
            Util::publishCloud(global_map_pub, *globalMap, ros::Time(t), current_ref_frame);

        tt_ufoupdate.Toc();    
    }

    void SendCloudToMapQueue(CloudXYZIPtr &cloud)
    {
        lock_guard<mutex> lg(mapqueue_mtx);
        mapqueue.push_back(cloud);
    }

    void UpdateMap()
    {
        while(ros::ok())
        {
            if (mapqueue.size() == 0)
            {
                this_thread::sleep_for(chrono::milliseconds(5));
                continue;
            }
            
            // Extract the cloud
            CloudXYZIPtr cloud;
            {
                lock_guard<mutex> lg(mapqueue_mtx);
                cloud = mapqueue.front();
                mapqueue.pop_front();
            }

            // Insert the cloud to the map
            {
                lock_guard<mutex> lg(map_mtx);
                if(use_ufm)
                    insertCloudToSurfelMap(*activeSurfelMap, *cloud);
                else
                {
                    if(activeikdtMap->Root_Node == nullptr)
                        activeikdtMap->Build(cloud->points);
                    else
                        activeikdtMap->Add_Points(cloud->points, true);
                }
            }
        }
    }

    void AssociateCloudWithMap(ufoSurfelMap &Map, ikdtreePtr &activeikdtMap, mytf tf_W_B,
                               CloudXYZITPtr const &CloudSkewed, CloudXYZIPtr &CloudDeskewedDS,
                               vector<LidarCoef> &CloudCoef, map<int, int> &stat)
    {
        int pointsCount = CloudDeskewedDS->points.size();
        if (CloudCoef.size() != pointsCount)
        {
            // Initialize the coefficent buffer
            CloudCoef.reserve(pointsCount);
        }

        // Create a static associator
        static PointToMapAssoc pma(nh_ptr);

        int featureTotal = CloudDeskewedDS->size();
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int k = 0; k < featureTotal; k++)
        {
            auto &point = CloudDeskewedDS->points[k];
            int  point_idx = (int)(point.intensity);
            int  coeff_idx = k;

            // Reset the coefficient
            CloudCoef[coeff_idx].t = -1;
            CloudCoef[coeff_idx].t_ = CloudSkewed->points[point_idx].t;
            CloudCoef[coeff_idx].d2P = -1;
            CloudCoef[coeff_idx].ptIdx = point_idx;
            CloudCoef[coeff_idx].marginalized = false;

            // Set the default value
            PointXYZIT pointRaw = CloudSkewed->points[point_idx];
            PointXYZI  pointInB = point;
            PointXYZI  pointInW = Util::transform_point(tf_W_B, pointInB);

            // Check if the point is valid
            if(!Util::PointIsValid(pointInB) || pointRaw.t < 0)
            {
                // printf(KRED "Invalid surf point!: %f, %f, %f\n" RESET, pointInB.x, pointInB.y, pointInB.z);
                pointInB.x = 0; pointInB.y = 0; pointInB.z = 0; pointInB.intensity = 0;
                continue;
            }

            // // Check if containing node can marginalize the points
            // auto containPred = ufopred::DepthE(surfel_min_depth)
            //                 && ufopred::Contains(ufoPoint3(pointInW.x, pointInW.y, pointInW.z));

            // // Find the list of containing node
            // deque<ufoNode> containingNode;
            // for (const ufoNode &node : Map.queryBV(containPred))
            //     containingNode.push_back(node);

            // // If point has no containing node, consider it a marginalizable points
            // if (containingNode.size() == 0)
            // {
            //     CloudCoef[coeff_idx + surfel_min_depth].marginalized = true;
            //     CloudCoef[coeff_idx + surfel_min_depth].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
            // }
            // else
            // {
            //     for (const ufoNode &node : containingNode)
            //     {
            //         if (Map.getSurfel(containingNode.front()).getNumPoints() < surfel_min_point)
            //         {
            //             ROS_ASSERT( node.depth() == surfel_min_depth );
            //             CloudCoef[coeff_idx + surfel_min_depth].marginalized = true;
            //             CloudCoef[coeff_idx + surfel_min_depth].finW = Vector3d(pointInW.x, pointInW.y, pointInW.z);
            //         }
            //     }
            // }

            if(use_ufm)
            {
                Vector3d finB(pointInB.x, pointInB.y, pointInB.z);
                Vector3d finW(pointInW.x, pointInW.y, pointInW.z);
                pma.AssociatePointWithMap(pointRaw, finB, finW, Map, CloudCoef[coeff_idx]);
            }
            else
            {
                int numNbr = surfel_min_point;
                ikdtPointVec nbrPoints;
                vector<float> knnSqDis;
                activeikdtMap->Nearest_Search(pointInW, numNbr, nbrPoints, knnSqDis);

                if (nbrPoints.size() < numNbr)
                    continue;
                else if (knnSqDis[numNbr - 1] > 5.0)
                    continue;
                else
                {
                    Vector4d pabcd;
                    double rho;
                    if(Util::fitPlane(nbrPoints, surfel_min_plnrty, dis_to_surfel_max, pabcd, rho))
                    {
                        float d2p = pabcd(0) * pointInW.x + pabcd(1) * pointInW.y + pabcd(2) * pointInW.z + pabcd(3);
                        float score = (1 - 0.9 * fabs(d2p) / Util::pointDistance(pointInB))*rho;
                        // float score = 1 - 0.9 * fabs(d2p) / (1 + pow(Util::pointDistance(pointInB), 4));
                        // float score = 1;

                        if (score > score_min)
                        {
                            // Add to coeff

                            LidarCoef &coef = CloudCoef[coeff_idx];

                            coef.t      = pointRaw.t;
                            coef.ptIdx  = point_idx;
                            coef.n      = pabcd;
                            coef.scale  = surfel_min_depth;
                            coef.surfNp = numNbr;
                            coef.plnrty = score;
                            coef.d2P    = d2p;
                            coef.f      = Vector3d(pointRaw.x, pointRaw.y, pointRaw.z);
                            coef.fdsk   = Vector3d(pointInB.x, pointInB.y, pointInB.z);
                            coef.finW   = Vector3d(pointInW.x, pointInW.y, pointInW.z);
                        }
                    }
                }
            }
        }

        // Find the statistics of the associations
        for(int i = 0; i < featureTotal; i++)
        {
            int point_idx = (int)CloudDeskewedDS->points[i].intensity;
            int coeff_idx = i;

            auto &coef = CloudCoef[coeff_idx];
            if (coef.t >= 0)
            {
                // CloudCoef.push_back(coef);
                stat[coef.scale] += 1;
                // break;
            }
        }
    }
    
    void makeDVAReport(deque<map<int, int>> &stats, map<int, int> &DVA, int &total, string &DVAReport)
    {
        DVA.clear();
        total = 0;
        DVAReport = "";

        for(auto &stepdva : stats)
        {
            for(auto &dva : stepdva)
            {
                total += dva.second;
                DVA[dva.first] += dva.second;
            }
        }
        
        // Calculate the mean and variance of associations at each scale
        // double N = DVA.size();
        // double mean = total / N;
        // double variance = 0;
        // for(auto &dva : DVA)
        //     variance += std::pow(dva.second - mean, 2);

        // variance = sqrt(variance/N);

        // // Find the depths with association count within 2 variance
        // map<int, bool> inlier;
        // for(auto &dva : DVA)
        //     inlier[dva.first] = fabs(dva.second - mean) < variance;

        int max_depth = -1;
        int max_assoc = 0;
        for(auto &dva : DVA)
        {
            if (dva.second > max_assoc)
            {
                max_depth = dva.first;
                max_assoc = dva.second;
            }
        }            

        // Create a report with color code
        for(auto &dva : DVA)
            DVAReport += myprintf("%s[%2d, %5d]"RESET, dva.second > max_assoc/3 ? KYEL : KWHT, dva.first, dva.second);

        DVAReport += myprintf(". DM: %2d. MaxA: %d", max_depth, max_assoc);
    }
    
    void FactorSelection(PoseSplineX &traj, vector<ImuIdx> &imuSelected, vector<lidarFeaIdx> &featureSelected)
    {
        // Counters for coupling on each knot
        // vector<int> knot_count_imu(traj.numKnots(), 0);
        // vector<int> knot_count_ldr(traj.numKnots(), 0);

        // Selecting the imu factors
        for(int i = 0; i < WINDOW_SIZE; i++)
        {
            for(int j = 0; j < N_SUB_SEG; j++)
            {
                for(int k = 1; k < SwImuBundle[i][j].size(); k++)
                {
                    double sample_time = SwImuBundle[i][j][k].t;
                    
                    // Skip if sample time exceeds the bound
                    if (!traj.TimeIsValid(sample_time, 1e-6))
                        continue;

                    auto us = traj.computeTIndex(sample_time);
                    int knot_idx = us.second;

                    // knot_count_imu[knot_idx] += 1;
                    imuSelected.push_back(ImuIdx(i, j, k));
                }
            }
        }

        // A temporary container of selected features by steps in the sliding window for further downsampling
        vector<vector<lidarFeaIdx>> featureBySwStep(WINDOW_SIZE);

        // Selecting the lidar factor
        int total_selected = 0;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            for (int k = 0; k < SwCloudDskDS[i]->size(); k++)
            {
                // A lot of factors are calculated but only a subset are used for optimization (time constraint).
                // By adding a counter we can shuffle the factors so all factors have the chance to be used.
                if ((k + i) % lidar_ds_rate != 0)
                    continue;

                auto &point = SwCloudDskDS[i]->points[k];
                int  point_idx = (int)(point.intensity);
                int  coeff_idx = k;

                LidarCoef &coef = SwLidarCoef[i][coeff_idx];

                if (coef.t < 0)
                    continue;

                double sample_time = coef.t;

                if (!traj.TimeIsValid(sample_time, 1e-6))
                    continue;

                auto us = traj.computeTIndex(sample_time);
                int knot_idx = us.second;

                total_selected++;
                // knot_count_ldr[knot_idx] += 1;
                featureBySwStep[i].push_back(lidarFeaIdx(i, k, coef.scale, total_selected));
            }
        }

        // If number of lidar feature remain large, randomly select a subset
        if (total_selected > max_lidar_factor)
        {
            // Define Fisher-Yates shuffle lambda function
            auto fisherYatesShuffle = [](std::vector<int>& array)
            {
                std::random_device rd;
                std::mt19937 gen(rd());

                for (int i = array.size() - 1; i > 0; --i)
                {
                    std::uniform_int_distribution<int> distribution(0, i);
                    int j = distribution(gen);
                    std::swap(array[i], array[j]);
                }
            };

            // How many features each swstep do we need?
            int maxFeaPerSwStep = ceil(double(max_lidar_factor) / WINDOW_SIZE);

            // Container for shuffled features
            // vector<vector<lidarFeaIdx>> featureBySwStepShuffled(WINDOW_SIZE);
            vector<vector<int>> shuffledIdx(WINDOW_SIZE);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int wid = 0; wid < WINDOW_SIZE; wid++)
            {
                shuffledIdx[wid] = vector<int>(featureBySwStep[wid].size());
                std::iota(shuffledIdx[wid].begin(), shuffledIdx[wid].end(), 0);
                
                // Shuffle the feature set a few times
                fisherYatesShuffle(shuffledIdx[wid]);
                fisherYatesShuffle(shuffledIdx[wid]);
                fisherYatesShuffle(shuffledIdx[wid]);
            }

            for(int wid = 0; wid < WINDOW_SIZE; wid++)
                for(int idx = 0; idx < min(maxFeaPerSwStep, (int)featureBySwStep[wid].size()); idx++)
                        featureSelected.push_back(featureBySwStep[wid][shuffledIdx[wid][idx]]);
        }
        else
        {
            for(int wid = 0; wid < WINDOW_SIZE; wid++)
                for(int idx = 0; idx < featureBySwStep[wid].size(); idx++)
                    featureSelected.push_back(featureBySwStep[wid][idx]);
        }
    }

    void DetectLoop()
    {
        // For visualization
        static ros::Publisher loop_kf_nbr_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_kf_nbr", 100);
        CloudPosePtr loopKfNbr(new CloudPose());

        static ros::Publisher loop_currkf_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_currkf", 100);
        CloudPosePtr loopCurrKf(new CloudPose());

        static ros::Publisher loop_prevkf_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_prevkf", 100);
        CloudPosePtr loopPrevKf(new CloudPose());

        static ros::Publisher loop_currCloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_curr_cloud", 100);
        static ros::Publisher loop_prevCloud_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_prev_cloud", 100);
        static ros::Publisher loop_currCloud_refined_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/loop_curr_refined_cloud", 100);

        // Extract the current pose
        int currPoseId = (int)(KfCloudPose->points.back().intensity);
        PointPose currPose = KfCloudPose->points[currPoseId];
        CloudXYZIPtr currCloudInB(new CloudXYZI()); CloudXYZIPtr currCloudInW(new CloudXYZI());
        *currCloudInB = *KfCloudinB[currPoseId];
        *currCloudInW = *KfCloudinW[currPoseId];

        // Search for the nearest neighbours
        vector<int> knn_idx(loop_kf_nbr); vector<float> knn_sq_dis(loop_kf_nbr);
        static KdTreeFLANN<PointPose> kdTreeKeyFrames;
        kdTreeKeyFrames.setInputCloud(KfCloudPose);
        kdTreeKeyFrames.nearestKSearch(currPose, loop_kf_nbr, knn_idx, knn_sq_dis);

        // Publish the current keyframe
        loopCurrKf->push_back(currPose);
        if (loopCurrKf->size() > 0)
            Util::publishCloud(loop_currkf_pub, *loopCurrKf, ros::Time(currPose.t), current_ref_frame);

        // Find the oldest index in the neigborhood
        int prevPoseId = -1;
        PointPose prevPose;
        CloudXYZIPtr prevCloudInW(new CloudXYZI());

        for (auto nbr_idx : knn_idx)
        {
            PointPose &kfPose = KfCloudPose->points[nbr_idx];
            loopKfNbr->push_back(kfPose);

            ROS_ASSERT(nbr_idx == (int)(kfPose.intensity));
            if (prevPoseId == -1 || nbr_idx < prevPoseId)
                prevPoseId = nbr_idx;
        }

        // Publish the nbr kf for visualization
        if (loopKfNbr->size() > 0)
            Util::publishCloud(loop_kf_nbr_pub, *loopKfNbr, ros::Time(currPose.t), current_ref_frame);

        static int LAST_KF_COUNT = KfCloudPose->size();

        // Only do the check every 5 keyframes
        int newKfCount = KfCloudPose->size();
        if (newKfCount - LAST_KF_COUNT < 5 || newKfCount <= loop_kf_nbr)
            return;
        LAST_KF_COUNT = newKfCount;

        // If new loop is too close to last loop in time, skip
        if (!loopPairs.empty())
        {
            double time_since_lastloop = fabs(KfCloudPose->points.back().t - KfCloudPose->points[loopPairs.back().currPoseId].t);
            // printf("Time since last loop: %f\n", time_since_lastloop);

            if (time_since_lastloop < loop_time_mindiff)
                return;
        }

        double time_nbr_diff = fabs(KfCloudPose->points[currPoseId].t - KfCloudPose->points[prevPoseId].t);
        // printf("Time nbr diff: %f\n", time_nbr_diff);

        // Return if no neighbour found, or the two poses are too close in time
        if (prevPoseId == -1 || time_nbr_diff < loop_time_mindiff || abs(currPoseId - prevPoseId) < loop_kf_nbr)
            return;
        else
            prevPose = KfCloudPose->points[prevPoseId];

        // Previous pose detected, build the previous local map

        // Find the range of keyframe Ids
        int bId = prevPoseId; int fId = prevPoseId; int span = fId - bId;
        while(span < loop_kf_nbr)
        {
            bId = max(0, bId - 1);
            fId = min(fId + 1, currPoseId - 1);

            int new_span = fId - bId;

            if ( new_span == span || new_span >= loop_kf_nbr )
                break;
            else
                span = new_span;
        }

        // Extract the keyframe pointcloud around the reference pose
        for(int kfId = bId; kfId < fId; kfId++)
        {
            loopPrevKf->push_back(KfCloudPose->points[kfId]);
            *prevCloudInW += *KfCloudinW[kfId];
        }

        // Publish previous keyframe for vizualization
        if (loopPrevKf->size() > 0)
            Util::publishCloud(loop_prevkf_pub, *loopPrevKf, ros::Time(currPose.t), current_ref_frame);

        // Downsample the pointclouds
        pcl::UniformSampling<PointXYZI> downsampler;
        double voxel_size = max(leaf_size, 0.4);
        downsampler.setRadiusSearch(voxel_size);

        downsampler.setInputCloud(prevCloudInW);
        downsampler.filter(*prevCloudInW);
        
        downsampler.setInputCloud(currCloudInB);
        downsampler.filter(*currCloudInB);

        // Publish the cloud for visualization
        Util::publishCloud(loop_prevCloud_pub, *prevCloudInW, ros::Time(currPose.t), current_ref_frame);
        Util::publishCloud(loop_currCloud_pub, *currCloudInW, ros::Time(currPose.t), current_ref_frame);

        // Check match by ICP
        myTf tf_W_Bcurr_start = myTf(currPose);
        myTf tf_W_Bcurr_final = tf_W_Bcurr_start; Matrix4f tfm_W_Bcurr_final;

        bool icp_passed = false; double icpFitnessRes = -1; double icpCheckTime = -1;
        icp_passed =    CheckICP(prevCloudInW, currCloudInB,
                                 tf_W_Bcurr_start.cast<float>().tfMat(), tfm_W_Bcurr_final,
                                 histDis, icpMaxIter, icpFitnessThres, icpFitnessRes, icpCheckTime);
        lastICPFn = icpFitnessRes;

        // Return if icp check fails
        if (!icp_passed)
            return;

        tf_W_Bcurr_final = myTf(tfm_W_Bcurr_final).cast<double>();

        printf("%sICP %s. T_W(%03d)_B(%03d). Fn: %.3f. icpTime: %.3f.\n"
               "Start: Pos: %f, %f, %f. YPR: %f, %f, %f\n"
               "Final: Pos: %f, %f, %f. YPR: %f, %f, %f\n"
                RESET,
                icp_passed ? KBLU : KRED, icp_passed ? "passed" : "failed", prevPoseId, currPoseId, icpFitnessRes, icpCheckTime,
                tf_W_Bcurr_start.pos.x(), tf_W_Bcurr_start.pos.y(), tf_W_Bcurr_start.pos.z(),
                tf_W_Bcurr_start.yaw(),   tf_W_Bcurr_start.pitch(), tf_W_Bcurr_start.roll(),
                tf_W_Bcurr_final.pos.x(), tf_W_Bcurr_final.pos.y(), tf_W_Bcurr_final.pos.z(),
                tf_W_Bcurr_final.yaw(),   tf_W_Bcurr_final.pitch(), tf_W_Bcurr_final.roll());

        // Add the loop to buffer
        loopPairs.push_back(LoopPrior(prevPoseId, currPoseId, 1e-3, icpFitnessRes,
                                      mytf(prevPose).inverse()*tf_W_Bcurr_final));

        // Publish the transform current cloud
        pcl::transformPointCloud(*currCloudInB, *currCloudInW, tf_W_Bcurr_final.pos, tf_W_Bcurr_final.rot);
        Util::publishCloud(loop_currCloud_refined_pub, *currCloudInW, ros::Time(currPose.t), current_ref_frame);
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

    void BundleAdjustment(BAReport &report)
    {
        static int LAST_LOOP_COUNT = loopPairs.size();
        int newLoopCount = loopPairs.size();

        // Return if no new loop detected
        if (newLoopCount - LAST_LOOP_COUNT < 1)
            return;
        LAST_LOOP_COUNT = newLoopCount;

        // Solve the pose graph optimization problem
        OptimizePoseGraph(KfCloudPose, loopPairs, report);

        TicToc tt_rebuildmap;

        // Recompute the keyframe pointclouds
        #pragma omp parallel for num_threads(MAX_THREADS)
        for(int i = 0; i < KfCloudPose->size(); i++)
        {
            myTf tf_W_B(KfCloudPose->points[i]);
            pcl::transformPointCloud(*KfCloudinB[i], *KfCloudinW[i], tf_W_B.pos, tf_W_B.rot);
        }

        // Recompute the globalmap and ufomap
        {
            lock_guard<mutex> lggm(global_map_mtx);
            globalMap->clear();

            for(int i = 0; i < KfCloudPose->size(); i++)
                *globalMap += *KfCloudinW[i];

            // Downsample the global map
            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(leaf_size);
            downsampler.setInputCloud(globalMap);
            downsampler.filter(*globalMap);

            Util::publishCloud(global_map_pub, *globalMap, ros::Time(KfCloudPose->points.back().t), current_ref_frame);

            // Clear the map queu            
            {
                lock_guard<mutex> lgmq(mapqueue_mtx);
                mapqueue.clear();                       // TODO: Should transform the remaining clouds to the new coordinates.
            }
            // Build the surfelmap
            {
                lock_guard<mutex> lgam(map_mtx);

                if(use_ufm)
                {
                    activeSurfelMap->clear();
                    insertCloudToSurfelMap(*activeSurfelMap, *globalMap);
                }
                else
                {   
                    activeikdtMap = ikdtreePtr(new ikdtree(0.5, 0.6, leaf_size));
                    activeikdtMap->Add_Points(globalMap->points, false);
                }
            }

            // Increment the ufomap version
            ufomap_version++;
        }

        tt_rebuildmap.Toc();

        report.rebuildmap_time = tt_rebuildmap.GetLastStop();
    }

    void OptimizePoseGraph(CloudPosePtr &kfCloud, const deque<LoopPrior> &loops, BAReport &report)
    {
        TicToc tt_pgopt;

        static int BA_NUM = -1;

        int KF_NUM = kfCloud->size();

        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = omp_get_max_threads();
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Create params and load data
        double **PARAM_POSE = new double *[KF_NUM];
        for(int i = 0; i < KF_NUM; i++)
        {
            PARAM_POSE[i] = new double[7];

            PARAM_POSE[i][0] = kfCloud->points[i].x;
            PARAM_POSE[i][1] = kfCloud->points[i].y;
            PARAM_POSE[i][2] = kfCloud->points[i].z;
            PARAM_POSE[i][3] = kfCloud->points[i].qx;
            PARAM_POSE[i][4] = kfCloud->points[i].qy;
            PARAM_POSE[i][5] = kfCloud->points[i].qz;
            PARAM_POSE[i][6] = kfCloud->points[i].qw;

            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(PARAM_POSE[i], 7, local_parameterization);

            // Fix the last pose
            if (i == KF_NUM - 1)
                problem.SetParameterBlockConstant(PARAM_POSE[i]);
        }

        // Add relative pose factors
        vector<ceres::internal::ResidualBlock *> res_ids_relpose;
        double cost_relpose_init = -1, cost_relpose_final = -1;
        for(int i = 1; i < KF_NUM; i++)
        {
            for (int j = 1; j < rib_edge; j++)
            {
                int jj = j;

                // Make an edge to the first pose for the poses with 5 steps
                if (i - j <= 0)
                    jj = i;

                myTf pose_i = myTf(kfCloud->points[i]);
                myTf pose_j = myTf(kfCloud->points[i-jj]);

                RelOdomFactor* relodomfactor = new RelOdomFactor(pose_i.pos, pose_j.pos, pose_i.rot, pose_j.rot,
                                                                 odom_q_noise, odom_p_noise);
                ceres::internal::ResidualBlock *res_id =  problem.AddResidualBlock(relodomfactor, NULL, PARAM_POSE[i], PARAM_POSE[i-jj]);
                res_ids_relpose.push_back(res_id);
            }
        }

        // Add loop factors
        vector<ceres::internal::ResidualBlock *> res_ids_loop;
        double cost_loop_init = -1, cost_loop_final = -1;
        for(auto &loop_edge : loopPairs)
        {
            // printf("Loop Factor: prev %d, curr: %d\n", loop_edge.prev_idx, loop_edge.curr_idx);
            
            int &curr_idx = loop_edge.currPoseId;
            int &prev_idx = loop_edge.prevPoseId;

            double &JKavr = loop_edge.JKavr;
            double &IcpFn = loop_edge.IcpFn;

            myTf pose_i = myTf(kfCloud->points[prev_idx]);
            myTf pose_j = myTf(kfCloud->points[prev_idx])*loop_edge.tf_Bp_Bc;

            RelOdomFactor* relodomfactor = new RelOdomFactor(pose_i.pos, pose_j.pos, pose_i.rot, pose_j.rot,
                                                             odom_q_noise*loop_weight, odom_p_noise*loop_weight);
            ceres::internal::ResidualBlock *res_id = problem.AddResidualBlock(relodomfactor, NULL, PARAM_POSE[prev_idx], PARAM_POSE[curr_idx]);
            res_ids_loop.push_back(res_id);
        }
        
        Util::ComputeCeresCost(res_ids_relpose, cost_relpose_init, problem);
        Util::ComputeCeresCost(res_ids_loop, cost_loop_init, problem);
        
        ceres::Solve(options, &problem, &summary);

        Util::ComputeCeresCost(res_ids_relpose, cost_relpose_final, problem);
        Util::ComputeCeresCost(res_ids_loop, cost_loop_final, problem);

        // Return the keyframe result
        for(int i = 0; i < KF_NUM; i++)
        {
            kfCloud->points[i].x  = PARAM_POSE[i][0];
            kfCloud->points[i].y  = PARAM_POSE[i][1];
            kfCloud->points[i].z  = PARAM_POSE[i][2];
            kfCloud->points[i].qx = PARAM_POSE[i][3];
            kfCloud->points[i].qy = PARAM_POSE[i][4];
            kfCloud->points[i].qz = PARAM_POSE[i][5];
            kfCloud->points[i].qw = PARAM_POSE[i][6];
        }
        
        baReport.turn           = (BA_NUM++);
        baReport.pgopt_time     = tt_pgopt.Toc();
        baReport.pgopt_iter     = summary.iterations.size();
        baReport.factor_relpose = res_ids_relpose.size();
        baReport.factor_loop    = res_ids_loop.size();
        baReport.J0             = summary.initial_cost;
        baReport.JK             = summary.final_cost;
        baReport.J0_relpose     = cost_relpose_init;
        baReport.JK_relpose     = cost_relpose_final;
        baReport.J0_loop        = cost_loop_init;
        baReport.JK_loop        = cost_loop_final;
    }

    void VisualizeLoop()
    {
        // Visualize the loop
        static visualization_msgs::Marker loop_marker; static bool loop_marker_inited = false;
        static ros::Publisher loop_marker_pub = nh_ptr->advertise<visualization_msgs::Marker>("/loop_marker", 100);
        static std_msgs::ColorRGBA color;

        if (!loop_marker_inited)
        {
            // Set up the loop marker
            loop_marker_inited = true;
            loop_marker.header.frame_id = current_ref_frame;
            loop_marker.ns       = "loop_marker";
            loop_marker.type     = visualization_msgs::Marker::LINE_LIST;
            loop_marker.action   = visualization_msgs::Marker::ADD;
            loop_marker.pose.orientation.w = 1.0;
            loop_marker.lifetime = ros::Duration(0);
            loop_marker.id       = 0;

            loop_marker.scale.x = 0.3; loop_marker.scale.y = 0.3; loop_marker.scale.z = 0.3;
            loop_marker.color.r = 0.0; loop_marker.color.g = 1.0; loop_marker.color.b = 1.0; loop_marker.color.a = 1.0;
            
            color.r = 0.0; color.g = 1.0; color.b = 1.0; color.a = 1.0;
        }

        loop_marker.points.clear();
        loop_marker.colors.clear();
        for(int i = 0; i < loopPairs.size(); i++)
        {
            int curr_idx = loopPairs[i].currPoseId;
            int prev_idx = loopPairs[i].prevPoseId;

            auto pose_curr = KfCloudPose->points[curr_idx];
            auto pose_prev = KfCloudPose->points[prev_idx];

            // Updating the line segments------------------------
            
            geometry_msgs::Point point;

            point.x = pose_curr.x;
            point.y = pose_curr.y;
            point.z = pose_curr.z;

            loop_marker.points.push_back(point);
            loop_marker.colors.push_back(color);

            point.x = pose_prev.x;
            point.y = pose_prev.y;
            point.z = pose_prev.z;

            loop_marker.points.push_back(point);
            loop_marker.colors.push_back(color);
        }
        // Publish the loop markers
        loop_marker_pub.publish(loop_marker);
    }

    void VisualizeSwTraj()
    {
        // Publish the sliding window trajectory and log the spline in the world frame
        {
            // Publish the traj
            static ros::Publisher swprop_viz_pub = nh_ptr->advertise<nav_msgs::Path>("/swprop_traj", 100);

            // Check if we have completed relocalization
            myTf tf_L0_Lprior(Quaternd(1, 0, 0, 0), Vector3d(0, 0, 0));
            if(reloc_stat == RELOCALIZED)
                tf_L0_Lprior = tf_Lprior_L0.inverse();

            // static ofstream swtraj_log;
            // static bool one_shot = true;
            // if (one_shot)
            // {
            //     swtraj_log.precision(std::numeric_limits<double>::digits10 + 1);
            //     swtraj_log.open((log_dir + "/swtraj.csv").c_str());
            //     swtraj_log.close(); // To reset the file
            //     one_shot = false;
            // }

            // // Append the data
            // swtraj_log.open((log_dir + "/swtraj.csv").c_str(), std::ios::app);

            double time_stamp = SwTimeStep.back().back().final_time;

            // Publish the propagated poses
            nav_msgs::Path prop_path;
            prop_path.header.frame_id = slam_ref_frame;
            prop_path.header.stamp = ros::Time(time_stamp);
            for(int i = 0; i < WINDOW_SIZE; i++)
            {
                for(int j = 0; j < SwPropState[i].size(); j++)
                {
                    for (int k = 0; k < SwPropState[i][j].size(); k++)
                    {
                        geometry_msgs::PoseStamped msg;
                        msg.header.frame_id = slam_ref_frame;
                        msg.header.stamp = ros::Time(SwPropState[i][j].t[k]);
                        
                        Vector3d pInL0 = tf_L0_Lprior*SwPropState[i][j].P[k];
                        msg.pose.position.x = pInL0.x();
                        msg.pose.position.y = pInL0.y();
                        msg.pose.position.z = pInL0.z();
                        
                        prop_path.poses.push_back(msg);

                        if (i == 0)
                        {
                            SE3d pose = GlobalTraj->pose(SwPropState[i][j].t[k]);
                            Vector3d pos = pose.translation();
                            Quaternd qua = pose.so3().unit_quaternion();
                            Vector3d vel = GlobalTraj->transVelWorld(SwPropState[i][j].t[k]);
                            Vector3d gyr = GlobalTraj->rotVelBody(SwPropState[i][j].t[k]) + sfBig[i][j];
                            Vector3d acc = qua.inverse()*(GlobalTraj->transAccelWorld(SwPropState[i][j].t[k]) + GRAV) + sfBia[i][j];

                            // swtraj_log << SwPropState[i][j].t[k]
                            //            << "," << SwPropState[i][j].P[k].x() << "," << SwPropState[i][j].P[k].y() << "," << SwPropState[i][j].P[k].z()
                            //            << "," << SwPropState[i][j].Q[k].x() << "," << SwPropState[i][j].Q[k].y() << "," << SwPropState[i][j].Q[k].z() << "," << SwPropState[i][j].Q[k].w()
                            //            << "," << SwPropState[i][j].V[k].x() << "," << SwPropState[i][j].V[k].y() << "," << SwPropState[i][j].V[k].z()
                            //            << "," << SwPropState[i][j].gyr[k].x() << "," << SwPropState[i][j].gyr[k].y() << "," << SwPropState[i][j].gyr[k].z()
                            //            << "," << SwPropState[i][j].acc[k].x() << "," << SwPropState[i][j].acc[k].y() << "," << SwPropState[i][j].acc[k].z()
                            //            << "," << pos.x() << "," << pos.y() << "," << pos.z()
                            //            << "," << qua.x() << "," << qua.y() << "," << qua.z() << "," << qua.w()
                            //            << "," << vel.x() << "," << vel.y() << "," << vel.z()
                            //            << "," << gyr.x() << "," << gyr.y() << "," << gyr.z()
                            //            << "," << acc.x() << "," << acc.y() << "," << acc.z()
                            //            << endl;
                        }
                    }
                }
            }
            // swtraj_log.close();
            swprop_viz_pub.publish(prop_path);
        }

        // Publish the control points
        {
            static ros::Publisher sw_ctr_pose_viz_pub = nh_ptr->advertise<nav_msgs::Path>("/sw_ctr_pose", 100);
            
            double SwTstart = SwTimeStep.front().front().start_time;
            double SwTfinal = SwTimeStep.front().front().final_time;

            double SwDur = SwTfinal - SwTstart;

            nav_msgs::Path path;
            path.header.frame_id = slam_ref_frame;
            path.header.stamp = ros::Time(SwTfinal);

            for(int knot_idx = GlobalTraj->numKnots() - 1; knot_idx >= 0; knot_idx--)
            {
                double tknot = GlobalTraj->getKnotTime(knot_idx);
                if (tknot < SwTstart - 2*SwDur )
                    break;

                Vector3d pos = GlobalTraj->getKnotPos(knot_idx);
                geometry_msgs::PoseStamped msg;
                msg.header.frame_id = slam_ref_frame;
                msg.header.stamp = ros::Time(tknot);
                
                msg.pose.position.x = pos.x();
                msg.pose.position.y = pos.y();
                msg.pose.position.z = pos.z();
                
                path.poses.push_back(msg);
            }

            sw_ctr_pose_viz_pub.publish(path);
        }

        // Publishing odometry stuff
        static myTf tf_Lprior_L0_init;
        {
            static bool one_shot = true;
            if (one_shot)
            {
                // Get the init transform
                vector<double> T_W_B_ = {1, 0, 0, 0,
                                         0, 1, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1};
                nh_ptr->getParam("/T_M_W_init", T_W_B_);
                Matrix4d T_B_V = Matrix<double, 4, 4, RowMajor>(&T_W_B_[0]);
                tf_Lprior_L0_init = myTf(T_B_V);
                
                one_shot = false;
            }
        }

        // Stuff in world frame
        static ros::Publisher opt_odom_pub           = nh_ptr->advertise<nav_msgs::Odometry>("/opt_odom", 100);
        static ros::Publisher opt_odom_high_freq_pub = nh_ptr->advertise<nav_msgs::Odometry>("/opt_odom_high_freq", 100);
        static ros::Publisher lastcloud_pub          = nh_ptr->advertise<sensor_msgs::PointCloud2>("/lastcloud", 100);
        // Stuff in map frame
        static ros::Publisher opt_odom_inM_pub       = nh_ptr->advertise<nav_msgs::Odometry>("/opt_odom_inM", 100);
        
        if (reloc_stat != RELOCALIZED)
        {
            // Publish the odom
            PublishOdom(opt_odom_pub, sfPos.back().back(), sfQua.back().back(),
                        sfVel.back().back(), SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);

            // Publish the odom at sub segment ends
            for(int i = 0; i < N_SUB_SEG; i++)
            {
                double time_stamp = SwTimeStep.front()[i].final_time;
                PublishOdom(opt_odom_high_freq_pub, sfPos.front()[i], sfQua.front()[i],
                            sfVel.front()[i], SwPropState.front()[i].gyr.back(), SwPropState.front()[i].acc.back(),
                            sfBig.front()[i], sfBia.front()[i], ros::Time(time_stamp), slam_ref_frame);
            }

            // Publish the latest cloud
            CloudXYZIPtr latestCloud(new CloudXYZI());
            pcl::transformPointCloud(*SwCloudDsk.back(), *latestCloud, sfPos.back().back(), sfQua.back().back());
            Util::publishCloud(lastcloud_pub, *latestCloud, ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);
            
            // Publish pose in map frame by the initial pose guess
            Vector3d posInM = tf_Lprior_L0_init*sfPos.back().back();
            Quaternd quaInM = tf_Lprior_L0_init.rot*sfQua.back().back();
            Vector3d velInM = tf_Lprior_L0_init.rot*sfVel.back().back();  
            PublishOdom(opt_odom_inM_pub, posInM, quaInM,
                        velInM, SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), "map");

            // Publish the transform between map and world at low rate
            {
                // static double update_time = -1;
                // if (update_time == -1 || ros::Time::now().toSec() - update_time > 1.0)
                static bool oneshot = true;
                if(oneshot)
                {
                    oneshot = false;
                    // update_time = ros::Time::now().toSec();
                    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
                    geometry_msgs::TransformStamped rostf_M_W;
                    rostf_M_W.header.stamp            = ros::Time::now();
                    rostf_M_W.header.frame_id         = "map";
                    rostf_M_W.child_frame_id          = slam_ref_frame;
                    rostf_M_W.transform.translation.x = tf_Lprior_L0_init.pos.x();
                    rostf_M_W.transform.translation.y = tf_Lprior_L0_init.pos.y();
                    rostf_M_W.transform.translation.z = tf_Lprior_L0_init.pos.z();
                    rostf_M_W.transform.rotation.x    = tf_Lprior_L0_init.rot.x();
                    rostf_M_W.transform.rotation.y    = tf_Lprior_L0_init.rot.y();
                    rostf_M_W.transform.rotation.z    = tf_Lprior_L0_init.rot.z();
                    rostf_M_W.transform.rotation.w    = tf_Lprior_L0_init.rot.w();
                    static_broadcaster.sendTransform(rostf_M_W);
                }
            }
        }
        else
        {
            static myTf tf_L0_Lprior = tf_Lprior_L0.inverse();

            // Publish the odom in the original slam reference frame
            Vector3d posInW = tf_L0_Lprior*sfPos.back().back();
            Quaternd quaInW = tf_L0_Lprior.rot*sfQua.back().back();
            Vector3d velInW = tf_L0_Lprior.rot*sfVel.back().back();        
            PublishOdom(opt_odom_pub, posInW, quaInW,
                        velInW, SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);

            // Publish the odom at sub segment ends in the original slam reference frame
            for(int i = 0; i < N_SUB_SEG; i++)
            {
                double time_stamp = SwTimeStep.front()[i].final_time;
                Vector3d posInW = tf_L0_Lprior*sfPos.front()[i];
                Quaternd quaInW = tf_L0_Lprior.rot*sfQua.front()[i];
                Vector3d velInW = tf_L0_Lprior.rot*sfVel.front()[i];        
                PublishOdom(opt_odom_high_freq_pub, posInW, quaInW,
                            velInW, SwPropState.front()[i].gyr.back(), SwPropState.front()[i].acc.back(),
                            sfBig.front()[i], sfBia.front()[i], ros::Time(time_stamp), slam_ref_frame);
            }

            // Publish the latest cloud in the original slam reference frame
            CloudXYZIPtr latestCloud(new CloudXYZI());
            pcl::transformPointCloud(*SwCloudDsk.back(), *latestCloud, posInW, quaInW);
            Util::publishCloud(lastcloud_pub, *latestCloud, ros::Time(SwTimeStep.back().back().final_time), slam_ref_frame);

            // Publish pose in map frame by the true transform
            PublishOdom(opt_odom_inM_pub, sfPos.back().back(), sfQua.back().back(),
                        sfVel.back().back(), SwPropState.back().back().gyr.back(), SwPropState.back().back().acc.back(),
                        sfBig.back().back(), sfBia.back().back(), ros::Time(SwTimeStep.back().back().final_time), current_ref_frame);

            // Publish the transform between map and world at low rate
            {
                // static double update_time = -1;
                // if (update_time == -1 || ros::Time::now().toSec() - update_time > 1.0)
                static bool oneshot = true;
                if(oneshot)
                {
                    oneshot = false;
                    // update_time = ros::Time::now().toSec();
                    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
                    geometry_msgs::TransformStamped rostf_M_W;
                    rostf_M_W.header.stamp            = ros::Time::now();
                    rostf_M_W.header.frame_id         = "map";
                    rostf_M_W.child_frame_id          = slam_ref_frame;
                    rostf_M_W.transform.translation.x = tf_Lprior_L0.pos.x();
                    rostf_M_W.transform.translation.y = tf_Lprior_L0.pos.y();
                    rostf_M_W.transform.translation.z = tf_Lprior_L0.pos.z();
                    rostf_M_W.transform.rotation.x    = tf_Lprior_L0.rot.x();
                    rostf_M_W.transform.rotation.y    = tf_Lprior_L0.rot.y();
                    rostf_M_W.transform.rotation.z    = tf_Lprior_L0.rot.z();
                    rostf_M_W.transform.rotation.w    = tf_Lprior_L0.rot.w();
                    static_broadcaster.sendTransform(rostf_M_W);
                }
            }
        }

        // Publish the relocalization status
        {
            static ros::Publisher reloc_stat_pub = nh_ptr->advertise<std_msgs::String>("/reloc_stat", 100);
            std_msgs::String msg;
            if (reloc_stat == NOT_RELOCALIZED)
                msg.data = "NOT_RELOCALIZED";
            else if (reloc_stat == RELOCALIZED)
                msg.data = "RELOCALIZED";
            else if (reloc_stat == RELOCALIZING)
                msg.data = "RELOCALIZING";
            reloc_stat_pub.publish(msg);
        }

        {
            static ros::Publisher reloc_pose_pub = nh_ptr->advertise<std_msgs::String>("/reloc_pose_str", 100);
            std_msgs::String msg;
            if (reloc_stat == NOT_RELOCALIZED)
                msg.data = "NOT_RELOCALIZED";
            else if (reloc_stat == RELOCALIZED)
                msg.data = myprintf("RELOCALIZED. [%7.2f, %7.2f, %7.2f, %7.2f, %7.2f, %7.2f]",
                                    tf_Lprior_L0.pos(0), tf_Lprior_L0.pos(1),  tf_Lprior_L0.pos(2),
                                    tf_Lprior_L0.yaw(),  tf_Lprior_L0.pitch(), tf_Lprior_L0.roll());
            else if (reloc_stat == RELOCALIZING)
                msg.data = "RELOCALIZING";
            reloc_pose_pub.publish(msg);
        }
    }

    void SlideWindowForward()
    {
        // Pop the states and replicate the final state
        ssQua.pop_front(); ssQua.push_back(deque<Quaternd>(N_SUB_SEG, sfQua.back().back()));
        ssPos.pop_front(); ssPos.push_back(deque<Vector3d>(N_SUB_SEG, sfPos.back().back()));
        ssVel.pop_front(); ssVel.push_back(deque<Vector3d>(N_SUB_SEG, sfVel.back().back()));
        ssBia.pop_front(); ssBia.push_back(deque<Vector3d>(N_SUB_SEG, sfBia.back().back()));
        ssBig.pop_front(); ssBig.push_back(deque<Vector3d>(N_SUB_SEG, sfBig.back().back()));

        sfQua.pop_front(); sfQua.push_back(ssQua.back());
        sfPos.pop_front(); sfPos.push_back(ssPos.back());
        sfVel.pop_front(); sfVel.push_back(ssVel.back());
        sfBia.pop_front(); sfBia.push_back(ssBia.back());
        sfBig.pop_front(); sfBig.push_back(ssBig.back());

        // Pop the buffers
        SwTimeStep.pop_front();
        SwCloud.pop_front();
        SwCloudDsk.pop_front();
        SwCloudDskDS.pop_front();
        SwLidarCoef.pop_front();
        SwDepVsAssoc.pop_front();
        SwImuBundle.pop_front();
        SwPropState.pop_front();
    }

    bool PublishGlobalMaps(slict::globalMapsPublish::Request &req, slict::globalMapsPublish::Response &res)
    {
        // Log and save the trajectory
        SaveTrajLog();

        // Publish the full map
        Util::publishCloud(global_map_pub, *globalMap, ros::Time(KfCloudPose->points.back().t), current_ref_frame);

        res.result = 1;
        return true;
    }

    void SaveTrajLog()
    {
        printf("Logging cloud pose: %s.\n", (log_dir + "/KfCloudPose.pcd").c_str());
        
        int save_attempts = 0;
        int save_attempts_max = 50;
        CloudPose cloudTemp;
        PCDWriter writer;

        save_attempts = 0;
        writer.write<PointPose>(log_dir + "/KfCloudPoseBin.pcd", *KfCloudPose, 18);
        while (true)
        {
            writer.write(log_dir + "/KfCloudPoseBin.pcd", *KfCloudPose, 18);
            save_attempts++;

            bool saving_succeeded = false;
            if(pcl::io::loadPCDFile<PointPose>(log_dir + "/KfCloudPoseBin.pcd", cloudTemp) == 0)
            {
                if (cloudTemp.size() == KfCloudPose->size())
                    saving_succeeded = true;
            }
            
            if (saving_succeeded)
                break;
            else if (save_attempts > save_attempts_max)
            {
                printf(KRED "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d!. Giving up \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
                break;

            }
            else
            {
                printf(KYEL "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
            }
        }

        save_attempts = 0;
        writer.writeASCII<PointPose>(log_dir + "/KfCloudPose.pcd", *KfCloudPose, 18);
        while (true)
        {
            writer.writeASCII<PointPose>(log_dir + "/KfCloudPose.pcd", *KfCloudPose, 18);
            save_attempts++;

            bool saving_succeeded = false;
            if(pcl::io::loadPCDFile<PointPose>(log_dir + "/KfCloudPose.pcd", cloudTemp) == 0)
            {
                if (cloudTemp.size() == KfCloudPose->size())
                    saving_succeeded = true;
            }
            
            if (saving_succeeded)
                break;
            else if (save_attempts > save_attempts_max)
            {
                printf(KRED "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d!. Giving up \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
                break;

            }
            else
            {
                printf(KYEL "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
            }
        }

        save_attempts = 0;
        writer.writeASCII<PointPose>(log_dir + "/KfCloudPoseExtra.pcd", *KfCloudPose, 18);
        while (true)
        {
            writer.writeASCII<PointPose>(log_dir + "/KfCloudPoseExtra.pcd", *KfCloudPose, 18);
            save_attempts++;

            bool saving_succeeded = false;
            if(pcl::io::loadPCDFile<PointPose>(log_dir + "/KfCloudPoseExtra.pcd", cloudTemp) == 0)
            {
                if (cloudTemp.size() == KfCloudPose->size())
                    saving_succeeded = true;
            }
            
            if (saving_succeeded)
                break;
            else if (save_attempts > save_attempts_max)
            {
                printf(KRED "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d!. Giving up \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
                break;

            }
            else
            {
                printf(KYEL "Saving failed!. Saved %5d vs %5d. Retry %2d / %2d \n" RESET,
                            cloudTemp.size(), KfCloudPose->size(), save_attempts, save_attempts_max );
            }
        }

        printf(KGRN "Logging the map completed.\n" RESET);


        printf(KYEL "Logging the map start ...\n" RESET);

        {
            lock_guard<mutex> lock(global_map_mtx);

            pcl::UniformSampling<PointXYZI> downsampler;
            downsampler.setRadiusSearch(max(leaf_size, 0.2));
            downsampler.setInputCloud(globalMap);
            downsampler.filter(*globalMap);

            printf("Logging global map: %s.\n", (log_dir + "/globalMap.pcd").c_str());
            pcl::io::savePCDFileBinary(log_dir + "/globalMap.pcd", *globalMap);

            #pragma omp parallel for num_threads(MAX_THREADS)
            for(int i = 0; i < KfCloudinW.size(); i++)
            {
                string file_name = log_dir_kf + "/KfCloudinW_" + zeroPaddedString(i, KfCloudinW.size()) + ".pcd";
                
                // printf("Logging KF cloud %s.\n", file_name.c_str());
                pcl::io::savePCDFileBinary(file_name, *KfCloudinW[i]);
            }
        }

        printf(KGRN "Logging the map completed.\n" RESET);


        printf(KYEL "Logging the loop ...\n" RESET);

        std::ofstream loop_log_file;
        loop_log_file.open(log_dir + "/loop_log.csv");
        loop_log_file.precision(std::numeric_limits<double>::digits10 + 1);

        for(auto &loop : loopPairs)
        {
            loop_log_file << loop.currPoseId << ", "
                          << loop.prevPoseId << ", "
                          << loop.JKavr << ", "
                          << loop.IcpFn << ", "
                          << loop.tf_Bp_Bc.pos(0) << ", "
                          << loop.tf_Bp_Bc.pos(1) << ", "
                          << loop.tf_Bp_Bc.pos(2) << ", "
                          << loop.tf_Bp_Bc.rot.x() << ", "
                          << loop.tf_Bp_Bc.rot.y() << ", "
                          << loop.tf_Bp_Bc.rot.z() << ", "
                          << loop.tf_Bp_Bc.rot.w() << endl;
        }

        loop_log_file.close();

        printf(KGRN "Logging the loop completed.\n" RESET);


        printf(KYEL "Logging the spline.\n" RESET);

        LogSpline(log_dir + "/spline_log.csv", *GlobalTraj, 0);

        printf(KYEL "Logging the spline completed.\n" RESET);

    }

    void LogSpline(string filename, PoseSplineX &traj, int outer_iteration)
    {
        std::ofstream spline_log_file;

        // Sample the spline from start to end
        spline_log_file.open(filename);
        spline_log_file.precision(std::numeric_limits<double>::digits10 + 1);

        // First row gives some metrics
        spline_log_file
                << "Dt: "         << traj.getDt()
                << ", Order: "    << SPLINE_N
                << ", Knots: "    << traj.numKnots()
                << ", MinTime: "  << traj.minTime()
                << ", MaxTime: "  << traj.maxTime()
                << ", OtrItr: "   << outer_iteration
                << endl;

        // Logging the knots
        for(int i = 0; i < traj.numKnots(); i++)
        {
            auto pose = traj.getKnot(i);
            auto pos = pose.translation(); auto rot = pose.so3().unit_quaternion();

            spline_log_file << i << ","
                            << traj.getKnotTime(i) << ","            
                            << pos.x() << "," << pos.y() << "," << pos.z() << ","
                            << rot.x() << "," << rot.y() << "," << rot.z() << "," << rot.w()
                            << endl;
        }

        spline_log_file.close();

    }

    void PublishOdom(ros::Publisher &pub, Vector3d &pos, Quaternd &qua,
                     Vector3d &vel, Vector3d &gyr, Vector3d &acc,
                     Vector3d &bg, Vector3d &ba, ros::Time stamp, string frame)
    {
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = frame;
        odom_msg.child_frame_id  = "body";

        odom_msg.pose.pose.position.x = pos.x();
        odom_msg.pose.pose.position.y = pos.y();
        odom_msg.pose.pose.position.z = pos.z();

        odom_msg.pose.pose.orientation.x = qua.x();
        odom_msg.pose.pose.orientation.y = qua.y();
        odom_msg.pose.pose.orientation.z = qua.z();
        odom_msg.pose.pose.orientation.w = qua.w();

        odom_msg.twist.twist.linear.x = vel.x();
        odom_msg.twist.twist.linear.y = vel.y();
        odom_msg.twist.twist.linear.z = vel.z();

        odom_msg.twist.twist.angular.x = gyr.x();
        odom_msg.twist.twist.angular.y = gyr.y();
        odom_msg.twist.twist.angular.z = gyr.z();

        odom_msg.twist.covariance[0] = acc.x();
        odom_msg.twist.covariance[1] = acc.y();
        odom_msg.twist.covariance[2] = acc.z();

        odom_msg.twist.covariance[3] = bg.x();
        odom_msg.twist.covariance[4] = bg.y();
        odom_msg.twist.covariance[5] = bg.z();
        odom_msg.twist.covariance[6] = ba.x();
        odom_msg.twist.covariance[7] = ba.y();
        odom_msg.twist.covariance[8] = ba.z();

        pub.publish(odom_msg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Estimator");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Estimator Started." RESET);

    Estimator estimator(nh_ptr);

    thread process_data(&Estimator::ProcessData, &estimator);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    estimator.SaveTrajLog();

    return 0;
}
