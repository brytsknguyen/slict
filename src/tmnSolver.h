#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"
#include "basalt/spline/posesplinex.h"

// Solver report
#include "slict/OptStat.h"
#include "slict/TimeLog.h"

// Utils
#include "utility.h"

// Association
#include "PointToMapAssoc.h"

// myGN solver
#include "factor/GyroAcceBiasFactorTMN.hpp"
#include "factor/Point2PlaneFactorTMN.hpp"
#include "factor/FlatGroundFactor.hpp"
struct ImuIdx
{
    ImuIdx(int &i_, int &j_, int &k_)
        : i(i_), j(j_), k(k_) {};
    int i; int j; int k;
};

struct lidarFeaIdx
{
    lidarFeaIdx(int &widx_, int &pointidx_, int &depth_, int absidx_)
        : wdidx(widx_), pointidx(pointidx_), depth(depth_), absidx(absidx_) {};

    int wdidx; int pointidx; int depth; int absidx;
};

class tmnSolver
{

public:
    
    // Constructor
    tmnSolver(ros::NodeHandlePtr &nh_);
    // Destructor
   ~tmnSolver();

    // Update the dimenstions of the optimization problem.
    void UpdateDimensions(int &imu_factors, int &ldr_factors, int knots);

    // Evaluate the imu factors to update the residual and jacobian
    void EvaluateImuFactors
    (
        PoseSplineX &traj, vector<SO3d> &xr, vector<Vector3d> &xp, Vector3d &xbg, Vector3d &xba,
        deque<deque<ImuSequence>> &SwImuBundle, vector<ImuIdx> &imuSelected, ImuBias imuBiasPrior,
        VectorXd &r, MatrixXd &J, double* cost
    );

    // Evaluate the lidar factors to update the residual and jacobian
    void EvaluateLdrFactors
    (
        PoseSplineX &traj, vector<SO3d> &xr, vector<Vector3d> &xp,
        deque<CloudXYZIPtr> &SwCloudDskDS,
        deque<vector<LidarCoef>> &SwLidarCoef,
        vector<lidarFeaIdx> &featureSelected,
        VectorXd &r, MatrixXd &J, double* cost
    );

    // Evaluate the flatground factors
    void EvaluateFlatGroundFactors
    (
        PoseSplineX &traj, vector<SO3d> &xr, vector<Vector3d> &xp,
        VectorXd &r, MatrixXd &J, double* cost
    );

    // Evaluate the marginalization factors to update the residual and jacobian
    void EvaluatePriFactors
    (
        int                  &iter,
        map<int, int>        &prev_knot_x,
        map<int, int>        &curr_knot_x,
        vector<SO3d>         &xr,
        vector<Vector3d>     &xp,
        Vector3d             &xbg,
        Vector3d             &xba,
        SparseMatrix<double> &bprior_sparse,
        SparseMatrix<double> &Hprior_sparse,
        VectorXd*            bprior_reduced,
        MatrixXd*            Hprior_reduced,
        double*              cost
    );

    // Solve the problem
    bool Solve
    (
        PoseSplineX               &traj,
        Vector3d                  &BIG,
        Vector3d                  &BIA,
        map<int, int>             &prev_knot_x,
        map<int, int>             &curr_knot_x,
        int                       &swNextBase,
        int                       &iter,
        deque<deque<ImuSequence>> &SwImuBundle,
        deque<CloudXYZIPtr>       &SwCloudDskDS,
        deque<vector<LidarCoef>>  &SwLidarCoef,
        vector<ImuIdx>            &imuSelected,
        vector<lidarFeaIdx>       &featureSelected,
        string                    &iekf_report,
        slict::OptStat            &report,
        slict::TimeLog            &tlog
    );

    // Utility to convert between prior forms
    void HbToJr(const MatrixXd &H, const VectorXd &b, MatrixXd &J, VectorXd &r);

    // Update the priors when there is a relocalization event
    void RelocalizePrior(SE3d tf);

private:

    // Node handle to get information needed
    ros::NodeHandlePtr nh;

    // gravity constant
    Vector3d GRAV;
    
    // IMU params
    double GYR_N = 1.0;
    double ACC_N = 1.0;
    double GYR_W = 1.0;
    double ACC_W = 1.0;

    // Lidar params
    double lidar_weight = 1.0;

    bool find_factor_cost = true;

    // Damping factor
    double lambda = 1.0;

    // Forcing two estimate
    bool flat_ground = false;
    double flat_ground_weight = 1.0;

    // Fuse marginalization
    bool fuse_marg = false;

    // Maximum change
    double dx_thres = 0.5;

    // Point to map association
    PointToMapAssoc *pma;

    // Maximum number of iteration
    int max_outer_iters;

    // Prior weight
    double prior_weight = 10;

    // Marginalization info
    MatrixXd Hkeep;
    VectorXd bkeep;
    MatrixXd Jm;            // 
    VectorXd rm;            // 
    // Marginalized states
    vector<SE3d> xse3_keep;
    Vector3d xbig_keep;
    Vector3d xbia_keep;
    MatrixXd dXkeep;

    // Index of knots in the marginalization hessian
    vector<pair<int, int>> knot_x_keep;

    int WINDOW_SIZE;
    int N_SUB_SEG;
    int SPLINE_N;

};