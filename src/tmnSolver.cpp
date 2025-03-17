#include "tmnSolver.h"

// Define the size of the params, residual, and Jacobian
#define XROT_SIZE  3                                        // Size of a control rot
#define XPOS_SIZE  3                                        // Size of a control pos
#define XSE3_SIZE  (XROT_SIZE + XPOS_SIZE)                  // Size of a control pose
#define XBIG_SIZE  3                                        // Size of the gyro bias
#define XBIA_SIZE  3                                        // Size of the acce bias
#define XBGA_SIZE  (XBIG_SIZE + XBIA_SIZE)
#define RESIMU_ROW 12                                       // Number of rows of an IMU residual block
#define RESIMU_COL 1                                        // Number of cols of an IMU residual block
#define RESLDR_ROW 1                                        // Number of rows of an lidar residual block
#define RESLDR_COL 1                                        // Number of rows of an lidar residual block
#define RESFGD_ROW 3                                        // Number of rows of an lidar residual block
#define RESFGD_COL 1                                        // Number of rows of an lidar residual block

// Local sizes of states,
int XROT_BASE;
int XPOS_BASE;
int XSE3_LSIZE;
int XSE3_LBASE;
int XBIG_LBASE;
int XBIA_LBASE;

// Global size of states
int XSE3_GSIZE;
int XALL_GSIZE;
int XSE3_GBASE;
int XBGA_GBASE;
int XBIG_GBASE;
int XBIA_GBASE;

// Global sizes of residual
int RESIMU_GSIZE;
int RESLDR_GSIZE;
int RESFGD_GSIZE = 0;
int RESALL_GSIZE;

void InsertZeroCol(MatrixXd &M, int col, int size)
{
    int Mrow = M.rows();
    int Mcol = M.cols();

    MatrixXd M_(Mrow, Mcol + size);
    M_ << M.leftCols(col), MatrixXd::Zero(Mrow, size), M.rightCols(Mcol - col);
    M.resize(Mrow, Mcol + size);
    M = M_;
}

void InsertZeroRow(MatrixXd &M, int row, int size)
{
    int Mrow = M.rows();
    int Mcol = M.cols();

    MatrixXd M_(Mrow + size, Mcol);
    M_ << M.topRows(row), MatrixXd::Zero(size, Mcol), M.bottomRows(Mrow - row);
    M.resize(Mrow + size, Mcol);
    M = M_;
}

void InsertZeroRow(VectorXd &M, int row, int size)
{
    int Mrow = M.rows();
    int Mcol = 1;

    VectorXd M_(Mrow + size, 1);
    M_ << M.topRows(row), MatrixXd::Zero(size, 1), M.bottomRows(Mrow - row);
    M.resize(Mrow + size, 1);
    M = M_;
}

bool GetBoolParam(ros::NodeHandlePtr &nh, string param, bool default_value)
{
    int param_;
    nh->param(param, param_, default_value == true ? 1 : 0);
    return (param_ == 0 ? false : true);
}

// Destructor
tmnSolver::~tmnSolver(){};

// Constructor
tmnSolver::tmnSolver(ros::NodeHandlePtr &nh_) : nh(nh_)
{
    // IMU noise
    nh->getParam("/GYR_N", GYR_N);
    nh->getParam("/GYR_W", GYR_W);
    nh->getParam("/ACC_N", ACC_N);
    nh->getParam("/ACC_W", ACC_W);

    // Dimension params
    nh->param("WINDOW_SIZE", WINDOW_SIZE, 4);
    nh->param("/N_SUB_SEG", N_SUB_SEG, 4);
    nh->param("/SPLINE_N", SPLINE_N, 4);
    
    // Gravity constants
    double GRAV_;
    nh->param("/GRAV", GRAV_, 9.82);
    GRAV = Vector3d(0, 0, GRAV_);

    // Lidar weight
    nh->getParam("/lidar_weight", lidar_weight);

    // Lidar weight
    nh->getParam("/flat_ground_weight", flat_ground_weight);

    // Damping factor
    nh->getParam("/lambda", lambda);

    // Select to whether to calculate the factor cost
    find_factor_cost = GetBoolParam(nh, "/find_factor_cost", true);

    // Select whether to force flat estimage
    flat_ground = GetBoolParam(nh, "/flat_ground", false);

    // Select marginalization fusion
    fuse_marg = GetBoolParam(nh, "/fuse_marg", false);

    // Maximum change of states
    nh->param("/dx_thres", dx_thres, 0.5);

    // PointToMap associator
    pma = new PointToMapAssoc(nh_);

    // Maximum number of iterations
    nh->getParam("/max_outer_iters", max_outer_iters);

    // Maximum number of iterations
    nh->getParam("/prior_weight", prior_weight);

    // Reset the marginalization
    knot_x_keep.clear();
};

// Change the size of variable dimension and observation dimension
void tmnSolver::UpdateDimensions(int &imu_factors, int &ldr_factors, int knots)
{
    // Local sizes of states
    XROT_BASE  = 0;
    XPOS_BASE  = XROT_SIZE;
    XSE3_LSIZE = SPLINE_N*XSE3_SIZE;
    XSE3_LBASE = 0;
    XBIG_LBASE = XSE3_LBASE + XSE3_LSIZE;
    XBIA_LBASE = XBIG_LBASE + XBIG_SIZE;

    // Global size of states
    XSE3_GSIZE = knots*XSE3_SIZE;
    XALL_GSIZE = XSE3_GSIZE + XBGA_SIZE;
    XBGA_GBASE = XSE3_GBASE + XSE3_GSIZE;
    XBIG_GBASE = XBGA_GBASE;
    XBIA_GBASE = XBIG_GBASE + XBIG_SIZE;

    // Global sizes of residual
    RESIMU_GSIZE = imu_factors*RESIMU_ROW;
    RESLDR_GSIZE = ldr_factors*RESLDR_ROW;
    RESFGD_GSIZE = 0;
    if(flat_ground)
        RESFGD_GSIZE = (knots - SPLINE_N + 1)*3;

    RESALL_GSIZE = RESIMU_GSIZE + RESLDR_GSIZE + RESFGD_GSIZE;
}

// Evaluate the imu factors to update residual and jacobian.
void tmnSolver::EvaluateImuFactors
(
    PoseSplineX &traj, vector<SO3d> &xr, vector<Vector3d> &xp, Vector3d &xbg, Vector3d &xba,
    deque<deque<ImuSequence>> &SwImuBundle, vector<ImuIdx> &imuSelected, ImuBias imuBiasPrior,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    #pragma omp parallel for num_threads(MAX_THREADS)
    for(int idx = 0; idx < imuSelected.size(); idx++)
    {
        int i = imuSelected[idx].i;
        int j = imuSelected[idx].j;
        int k = imuSelected[idx].k;

        double sample_time = SwImuBundle[i][j][k].t;

        auto   us = traj.computeTIndex(sample_time);
        double u  = us.first;
        int    s  = us.second;

        vector<SO3d> xr_local; vector<Vector3d> xp_local;

        // Add the parameter blocks for rotation
        for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
            xr_local.push_back(xr[knot_idx]);

        // Add the parameter blocks for position
        for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
            xp_local.push_back(xp[knot_idx]);

        // Create the factor
        typedef GyroAcceBiasFactorTMN gabFactor;
        gabFactor imuFactor(SwImuBundle[i][j][k], imuBiasPrior, GRAV, GYR_N, ACC_N, GYR_W, ACC_W, SPLINE_N, traj.getDt(), u);
        imuFactor.Evaluate(xr_local, xp_local, xbg, xba);

        int row = idx*RESIMU_ROW;
        int col = s*XSE3_SIZE;

        r.block(row, 0, RESIMU_ROW, 1         ).setZero();
        J.block(row, 0, RESIMU_ROW, XALL_GSIZE).setZero();

        // Copy data to big matrix
        r.block(row, 0,          RESIMU_ROW, RESIMU_COL) << imuFactor.residual;
        J.block(row, col,        RESIMU_ROW, XSE3_LSIZE) << imuFactor.jacobian.block(0, 0,          RESIMU_ROW, XSE3_LSIZE);
        J.block(row, XBIG_GBASE, RESIMU_ROW, XBIG_SIZE ) << imuFactor.jacobian.block(0, XBIG_LBASE, RESIMU_ROW, XBIG_SIZE );
        J.block(row, XBIA_GBASE, RESIMU_ROW, XBIA_SIZE ) << imuFactor.jacobian.block(0, XBIA_LBASE, RESIMU_ROW, XBIA_SIZE );
    }

    // Calculate the cost
    if (cost != NULL)
        *cost = pow(r.norm(), 2);
}

// Evaluate the lidar factors to update residual and jacobian
void tmnSolver::EvaluateLdrFactors
(
    PoseSplineX &traj, vector<SO3d> &xr, vector<Vector3d> &xp,
    deque<CloudXYZIPtr> &SwCloudDskDS,
    deque<vector<LidarCoef>> &SwLidarCoef,
    vector<lidarFeaIdx> &featureSelected,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    #pragma omp parallel for num_threads(MAX_THREADS)
    for (int idx = 0; idx < featureSelected.size(); idx++)
    {
        int i     = featureSelected[idx].wdidx;
        int k     = featureSelected[idx].pointidx;
        int depth = featureSelected[idx].depth;
        // int absidx = featureSelected[idx].absidx;

        auto &point = SwCloudDskDS[i]->points[k];
        int  point_idx = (int)(point.intensity);
        int  coeff_idx = k;

        LidarCoef &coef = SwLidarCoef[i][coeff_idx];

        double sample_time = coef.t;

        // // Get the new pose
        // SE3d pose = traj.pose(sample_time);
        
        // // New point coordinates
        // Vector3d finW_new = pose*coef.f;

        // // Find a better association
        // PointXYZIT fRaw; fRaw.x = coef.f(0); fRaw.y = coef.f(1); fRaw.z = coef.f(2); fRaw.t = sample_time;
        // pma->AssociatePointWithMap(fRaw, coef.fdsk, finW_new, activeSurfelMap, coef);

        auto   us = traj.computeTIndex(sample_time);
        double u  = us.first;
        int    s  = us.second;

        vector<SO3d> xr_local; vector<Vector3d> xp_local;

        // Add the parameter blocks for rotation
        for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
            xr_local.push_back(xr[knot_idx]);

        // Add the parameter blocks for position
        for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
            xp_local.push_back(xp[knot_idx]);

        // for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
        //     knot_check_ldr[knot_idx] += 1;

        // Calculate the jacobian of this lidar factor:
        typedef Point2PlaneFactorTMN p2pFactor;
        p2pFactor lidarFactor = p2pFactor(coef.finW, coef.f, coef.n, coef.plnrty*lidar_weight, SPLINE_N, traj.getDt(), u);

        // Calculate the residual and jacobian
        lidarFactor.Evaluate(xr_local, xp_local);

        int row = idx*RESLDR_ROW;
        int col = s*XSE3_SIZE;

        r.block(row, 0, RESLDR_ROW, 1         ).setZero();
        J.block(row, 0, RESLDR_ROW, XALL_GSIZE).setZero();

        r.block(row, 0,   RESLDR_ROW, RESLDR_COL) << lidarFactor.residual;
        J.block(row, col, RESLDR_ROW, XSE3_LSIZE) << lidarFactor.jacobian.block(0, 0, RESLDR_ROW, XSE3_LSIZE);
    }

    if (cost != NULL)
        *cost = pow(r.norm(), 2);
}

void tmnSolver::EvaluateFlatGroundFactors
(
    PoseSplineX &traj, vector<SO3d> &xr, vector<Vector3d> &xp,
    VectorXd &r, MatrixXd &J, double* cost
)
{
    #pragma omp parallel for num_threads(MAX_THREADS)
    for (int kidx = 0; kidx < traj.numKnots() - SPLINE_N + 1; kidx++)
    {
        double sample_time = traj.minTime() + (kidx + 0.5)*traj.getDt();
        auto   us = traj.computeTIndex(sample_time);
        double u  = us.first;
        int    s  = us.second;

        vector<SO3d> xr_local; vector<Vector3d> xp_local;

        // Add the parameter blocks for rotation
        for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
            xr_local.push_back(xr[knot_idx]);

        // Add the parameter blocks for position
        for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
            xp_local.push_back(xp[knot_idx]);

        // for (int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
        //     knot_check_ldr[knot_idx] += 1;

        // Calculate the jacobian of this lidar factor:
        typedef FlatGroundFactor FGFactor;
        FGFactor fgFactor = FGFactor(flat_ground_weight, SPLINE_N, traj.getDt(), u);

        // Calculate the residual and jacobian
        fgFactor.Evaluate(xr_local, xp_local);

        int row = kidx*RESFGD_ROW;
        int col = s*XSE3_SIZE;

        // r.block(row, 0, RESFGD_ROW, 1         ).setZero();
        // J.block(row, 0, RESFGD_ROW, XALL_GSIZE).setZero();

        r.block(row, 0,   RESFGD_ROW, RESFGD_COL) << fgFactor.residual;
        J.block(row, col, RESFGD_ROW, XSE3_LSIZE) << fgFactor.jacobian.block(0, 0, RESFGD_ROW, XSE3_LSIZE);

        // if (kidx == 0)
        // {
        //     cout << "rfgd" << endl << fgFactor.residual << endl;
        //     cout << "Jfgd" << endl << fgFactor.jacobian << endl;
        // }
    }
}

void tmnSolver::EvaluatePriFactors
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
)
{
    VectorXd rprior = VectorXd::Zero(XALL_GSIZE);
    MatrixXd Jprior = MatrixXd::Zero(XALL_GSIZE, XALL_GSIZE);
    VectorXd bprior = VectorXd::Zero(XALL_GSIZE);
    MatrixXd Hprior = MatrixXd::Zero(XALL_GSIZE, XALL_GSIZE);

    // Create marginalizing factor if there is states kept from last optimization       
    SparseMatrix<double> rprior_sparse;
    SparseMatrix<double> Jprior_sparse;

    // Copy the marginalized data to the Htilde
    if (knot_x_keep.size() != 0)
    {
        // Sanity check
        ROS_ASSERT(curr_knot_x[knot_x_keep.front().first] == 0);

        // Calculate the prior residual
        // #pragma omp parallel for num_threads(MAX_THREADS)
        for(int idx = 0; idx < knot_x_keep.size(); idx++)
        {
            pair<int, int> &knot_x = knot_x_keep[idx];

            int x_keep = knot_x.second;
            int x_curr = curr_knot_x[knot_x.first];

            int XROT_IDX = x_curr*XSE3_SIZE + XROT_BASE;
            int XPOS_IDX = x_curr*XSE3_SIZE + XPOS_BASE;

            // printf("xse3_keep: %d / %d, %f, %f, %f, %f\n",
            //         x_keep, xse3_keep.size(),
            //         xse3_keep[x_keep].so3().unit_quaternion().x(),
            //         xse3_keep[x_keep].so3().unit_quaternion().y(), 
            //         xse3_keep[x_keep].so3().unit_quaternion().z(), 
            //         xse3_keep[x_keep].so3().unit_quaternion().w() );

            Vector3d dr = (xse3_keep[x_keep].so3().inverse()*xr[x_curr]).log();
            Vector3d dp = xp[x_curr] - xse3_keep[x_keep].translation();

            rprior.block<XSE3_SIZE, 1>(x_curr*XSE3_SIZE, 0) << dr, dp;

            // printf("knot %d. xpcurr: %f, %f, %f. xpkeep: %f, %f, %f. dr: %f, %f, %f. dp: %f, %f, %f\n",
            //         knot_x.first,
            //         xp[x_curr].x(), xp[x_curr].y(), xp[x_curr].z(),
            //         xse3_keep[x_keep].translation().x(), xse3_keep[x_keep].translation().y(), xse3_keep[x_keep].translation().z(),
            //         dr.x(), dr.y(), dr.z(),
            //         dp.x(), dp.y(), dp.z()
            //       );

            if (dr.norm() < 1e-3 || dr.hasNaN())
                Jprior.block<XROT_SIZE, XROT_SIZE>(XROT_IDX, XROT_IDX) = Matrix3d::Identity(3,3);
            else
                Sophus::rightJacobianInvSO3(dr, Jprior.block<XROT_SIZE, XROT_SIZE>(XROT_IDX, XROT_IDX));

            Jprior.block<XPOS_SIZE, XPOS_SIZE>(XPOS_IDX, XPOS_IDX) = Matrix3d::Identity(3,3);
        }
        // Dont forget the bias states
        rprior.block<XBGA_SIZE, 1>(XBGA_GBASE, 0) << xbg - xbig_keep, xba - xbia_keep;
        Jprior.block<XBGA_SIZE, XBGA_SIZE>(XBGA_GBASE, XBGA_GBASE) = MatrixXd::Identity(XBGA_SIZE, XBGA_SIZE);

        // Copy the blocks in big step
        int XKEEP_SIZE = knot_x_keep.size()*XSE3_SIZE;
        Hprior.block(0, 0, XKEEP_SIZE, XKEEP_SIZE) = Hkeep.block(0, 0, XKEEP_SIZE, XKEEP_SIZE);
        Hprior.block(0, XBGA_GBASE, XKEEP_SIZE, XBGA_SIZE) = Hkeep.block(0, XKEEP_SIZE, XKEEP_SIZE, XBGA_SIZE);
        Hprior.block(XBGA_GBASE, 0, XBGA_SIZE, XKEEP_SIZE) = Hkeep.block(XKEEP_SIZE, 0, XBGA_SIZE, XKEEP_SIZE);

        // Update the b block
        bprior.block(0, 0, XKEEP_SIZE, 1) = bkeep.block(0, 0, XKEEP_SIZE, 1);
        bprior.block<XBGA_SIZE, 1>(XBGA_GBASE, 0) = bkeep.block<XBGA_SIZE, 1>(XKEEP_SIZE, 0);

        if (Hprior.hasNaN() || bprior.hasNaN())
            return;

        // Update the hessian
        rprior_sparse = rprior.sparseView(); rprior_sparse.makeCompressed();
        Jprior_sparse = Jprior.sparseView(); Jprior_sparse.makeCompressed();

        Hprior_sparse = Hprior.sparseView(); Hprior_sparse.makeCompressed();
        bprior_sparse = bprior.sparseView(); bprior_sparse.makeCompressed();

        bprior_sparse = bprior_sparse - Hprior_sparse*Jprior_sparse*rprior_sparse;
        Hprior_sparse = Jprior_sparse.transpose()*Hprior_sparse*Jprior_sparse;

        if(bprior_reduced != NULL)
        {
            *bprior_reduced = VectorXd(XKEEP_SIZE + XBGA_SIZE, 1);
            *bprior_reduced << bprior_sparse.block(0, 0, XKEEP_SIZE, 1).toDense(),
                               bprior_sparse.block(XBGA_GBASE, 0, XBGA_SIZE, 1).toDense();
        }

        if(Hprior_reduced != NULL)
        {
            *Hprior_reduced = MatrixXd(XKEEP_SIZE + XBGA_SIZE, XKEEP_SIZE + XBGA_SIZE);
            *Hprior_reduced << Hprior_sparse.block(0, 0, XKEEP_SIZE, XKEEP_SIZE).toDense(),
                               Hprior_sparse.block(0, XBGA_GBASE, XKEEP_SIZE, XBGA_SIZE).toDense(),
                               Hprior_sparse.block(XKEEP_SIZE, 0, XBGA_SIZE, XKEEP_SIZE).toDense(),
                               Hprior_sparse.block(XKEEP_SIZE, XBGA_GBASE, XBGA_SIZE, XBGA_SIZE).toDense();
        }

        if (cost != NULL)
        {
            // For cost computation
            VectorXd rprior_reduced(XKEEP_SIZE + XBGA_SIZE, 1);
            MatrixXd Jprior_reduced(XKEEP_SIZE + XBGA_SIZE, XKEEP_SIZE + XBGA_SIZE);
            rprior_reduced << rprior.block(0, 0, XKEEP_SIZE, 1), rprior.block(XBGA_GBASE, 0, XBGA_SIZE, 1);
            Jprior_reduced << Jprior.block(0, 0, XKEEP_SIZE, XKEEP_SIZE),
                              Jprior.block(0, XBGA_GBASE, XKEEP_SIZE, XBGA_SIZE),
                              Jprior.block(XKEEP_SIZE, 0, XBGA_SIZE, XKEEP_SIZE),
                              Jprior.block(XKEEP_SIZE, XBGA_GBASE, XBGA_SIZE, XBGA_SIZE);

            // printf("Sizes: rm: %d, %d. Jm: %d, %d. Jr: %d, %d. r: %d, %d\n",
            //         rm.rows(), rm.cols(),
            //         Jm.rows(), Jm.cols(),
            //         Jprior_reduced.rows(), Jprior_reduced.cols(),
            //         rprior_reduced.rows(), rprior_reduced.cols());

            if (cost != NULL)
                *cost = pow((rm + Jm*Jprior_reduced*rprior_reduced).norm(), 2);
        }
    }
}

void tmnSolver::HbToJr(const MatrixXd &H, const VectorXd &b, MatrixXd &J, VectorXd &r)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H);
    Eigen::VectorXd S = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes.eigenvalues().array() > 0).select(saes.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    J = S_sqrt.asDiagonal() * saes.eigenvectors().transpose();
    r = S_inv_sqrt.asDiagonal() * saes.eigenvectors().transpose() * b;
}


void tmnSolver::RelocalizePrior(SE3d tf)
{
    for (auto &se3 : xse3_keep)
    {
        se3 = tf*se3;
    }
}

// Solving the problem
bool tmnSolver::Solve
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
    string                    &bsu_report,
    slict::OptStat            &report,
    slict::TimeLog            &tlog
)
{
    TicToc tt_bsu;

    /* #region  */ TicToc tt_prep;

    static int prevSwNextBase = -1;
    if (prevSwNextBase != -1)
        ROS_ASSERT_MSG(prevSwNextBase == curr_knot_x.begin()->first,
                       "NextBase not matching: %d, %d",
                       prevSwNextBase, curr_knot_x.begin()->first);
    if (iter == 0)                       
        prevSwNextBase = swNextBase;

    // Total number of factors
    int imu_factors = imuSelected.size();
    int ldr_factors = featureSelected.size();

    UpdateDimensions(imu_factors, ldr_factors, traj.numKnots());

    // Only update the bias prior if we are in a new outer iteration
    static Vector3d BIGprior;
    static Vector3d BIAprior;
    if (iter == max_outer_iters - 1)
    {
        BIGprior = BIG;
        BIAprior = BIA;
    }

    // State estimates
    vector<SO3d>     xr;
    vector<Vector3d> xp;
    Vector3d         xbg;
    Vector3d         xba;

    // Copy the pose, the prior is also the current estimate
    for (int knot_idx = 0; knot_idx < traj.numKnots(); knot_idx++)
    {
        xr.push_back(traj.getKnotSO3(knot_idx));
        xp.push_back(traj.getKnotPos(knot_idx));
    }
    // Copy the bias
    xbg = BIG;
    xba = BIA;

    // Prepare the big matrices
    VectorXd RESIDUAL(RESIMU_GSIZE + RESLDR_GSIZE, 1);            // Residual
    MatrixXd JACOBIAN(RESIMU_GSIZE + RESLDR_GSIZE, XALL_GSIZE);   // Jacobian
    SparseMatrix<double> bprior_sparse;
    SparseMatrix<double> Hprior_sparse;

    /* #endregion  */ tt_prep.Toc();


    /* #region  */ TicToc tt_himu;
    
    double J0imu = 0;
    VectorXd RIMU(RESIMU_GSIZE, 1);
    MatrixXd JIMU(RESIMU_GSIZE, XALL_GSIZE);
    RIMU.setZero(); JIMU.setZero();
    EvaluateImuFactors(traj, xr, xp, xbg, xba, SwImuBundle, imuSelected, ImuBias(BIGprior, BIAprior), RIMU, JIMU, find_factor_cost ? &J0imu : NULL);

    /* #endregion */ tt_himu.Toc();


    /* #region  */ TicToc tt_hlidar;
    
    double J0ldr = 0;
    VectorXd RLDR(RESLDR_GSIZE, 1);
    MatrixXd JLDR(RESLDR_GSIZE, XALL_GSIZE);
    RLDR.setZero(); JLDR.setZero();
    EvaluateLdrFactors(traj, xr, xp, SwCloudDskDS, SwLidarCoef, featureSelected, RLDR, JLDR, find_factor_cost ? &J0ldr : NULL);

    /* #endregion */ tt_hlidar.Toc();


    /* #region  */ TicToc tt_flatground;
    
    double J0fgd = 0;
    VectorXd RFGD(RESFGD_GSIZE, 1);
    MatrixXd JFGD(RESFGD_GSIZE, XALL_GSIZE);
    if(flat_ground)
    {
        RFGD.setZero(); JFGD.setZero();
        EvaluateFlatGroundFactors(traj, xr, xp, RFGD, JFGD, find_factor_cost ? &J0fgd : NULL);
    }

    /* #endregion */ tt_flatground.Toc();


    /* #region  */ TicToc tt_hprior;

    double J0pri = 0;
    if(fuse_marg)
        EvaluatePriFactors(iter, prev_knot_x, curr_knot_x, xr, xp, xbg, xba, bprior_sparse, Hprior_sparse, NULL, NULL, find_factor_cost ? &J0pri : NULL);

    /* #endregion  */ tt_hprior.Toc();


    /* #region  */ TicToc tt_compute;

    auto AddrJtoHb = [](VectorXd &r, MatrixXd &J, SparseMatrix<double> &H, MatrixXd &b)
    {
        SparseMatrix<double> Jsparse = J.sparseView(); Jsparse.makeCompressed();    
        SparseMatrix<double> Jtp = Jsparse.transpose();

        if (b.size() == 0)
        {
            H =  Jtp*Jsparse;
            b = -Jtp*r;
        }
        else
        {
            H +=  Jtp*Jsparse;
            b += -Jtp*r;
        }
    };

    SparseMatrix<double> H; MatrixXd b;
    AddrJtoHb(RIMU, JIMU, H, b);
    AddrJtoHb(RLDR, JLDR, H, b);
    if(flat_ground)
    {
        assert(RFGD.maxCoeff() == 0.0);
        assert(JFGD.maxCoeff() == 0.0);
        
        assert(RFGD.minCoeff() == 0.0);
        assert(JFGD.minCoeff() == 0.0);
        
        AddrJtoHb(RFGD, JFGD, H, b);
    }

    if( fuse_marg
        && Hprior_sparse.rows() > 0
        && bprior_sparse.rows() > 0
      )
    {
        H += Hprior_sparse;
        b += bprior_sparse;
        // printf("Hprior: %f\n", Hprior_sparse.toDense().trace());
    }

    // Kalman method
    // SparseMatrix<double> Covinv(XALL_GSIZE, XALL_GSIZE); Covinv.setIdentity(); Covinv = Covinv*0.01;
    // SparseMatrix<double> I(XALL_GSIZE, XALL_GSIZE); I.setIdentity();
    // MatrixXd Sinv = S.toDense().inverse();
    // MatrixXd K    = Sinv*JtpRinv;
    // MatrixXd dX   = -K*r - (I - K*Jsparse)*JrInv*r_prior;

    // Direct solving (assuming non degenerative)
    // MatrixXd dX = -JtpRinvJ.toDense().inverse()*JtpRinv*r;

    MatrixXd dX = MatrixXd::Zero(XALL_GSIZE, 1);
    bool solver_failed = false;

    // Solving using dense QR
    // dX = S.toDense().colPivHouseholderQr().solve(b);

    // Solve using solver and LM method
    SparseMatrix<double> I(H.cols(), H.cols()); I.setIdentity();
    SparseMatrix<double> S = H + lambda/pow(2, (max_outer_iters - 1) - iter)*I;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(S);
    solver.factorize(S);
    solver_failed = solver.info() != Eigen::Success;
    dX = solver.solve(b);

    // Eigen::SparseQR<Eigen::SparseMatrix<double>, COLAMDOrdering<int>> solver;
    // solver.analyzePattern(S);
    // solver.factorize(S);
    // solver_failed = solver.info() != Eigen::Success;
    // // Calculate the increment
    // dX = solver.solve(b);

    // If solving is not successful, return false
    if (solver_failed || dX.hasNaN())
    {
        printf(KRED"Failed to solve!\n"RESET);

        for(int idx = 0; idx < traj.numKnots(); idx++)
            printf("XPOSE%02d. %7.3f, %7.3f\n", idx,
                    dX.block<3, 1>(idx*XSE3_SIZE + XROT_BASE, 0).norm(),
                    dX.block<3, 1>(idx*XSE3_SIZE + XPOS_BASE, 0).norm());
        printf("XBIAS%02d. %7.3f, %7.3f\n\n", 0, dX.block<3, 1>(XBIG_GBASE, 0).norm(), dX.block<3, 1>(XBIA_GBASE, 0).norm());

        // return false;
    }

    if (dX.norm() > dx_thres)
        dX = dX / dX.norm() * dx_thres;

    // cout << KYEL "Covariance: " RESET << endl << H.toDense() << endl;

    // Successful solve, now updating the states
    for(int idx = 0; idx < traj.numKnots(); idx++)
    {
        Vector3d dr = dX.block<3, 1>(idx*XSE3_SIZE + XROT_BASE, 0);
        Vector3d dp = dX.block<3, 1>(idx*XSE3_SIZE + XPOS_BASE, 0);
        
        // if (flat_ground)
        // {
        //     // dr(0) = 0;
        //     // dr(1) = 0;
        //     // dp(2) = 0;
        //     xr[idx] = xr[idx]*SO3d::exp(dr);
        //     xp[idx] += dp;
        // }
        // else
        {
            xr[idx] = xr[idx]*SO3d::exp(dr);
            xp[idx] += dp;
        }
    }
    xbg += dX.block<3, 1>(XBIG_GBASE, 0);
    xba += dX.block<3, 1>(XBIA_GBASE, 0);

    // Load the knots into the traj
    for(int knot_idx = 0; knot_idx < traj.numKnots(); knot_idx++)
    {
        // SE3d xk = X0*SE3d(xr[0], xp[0]).inverse()*SE3d(xr[knot_idx], xp[knot_idx]);
        SE3d xk = SE3d(xr[knot_idx], xp[knot_idx]);
        traj.setKnot(xk, knot_idx);
    }
    // Load the bias values
    BIG = xbg;
    BIA = xba;    

    /* #endregion */ tt_compute.Toc();


    /* #region */ TicToc tt_marg;

    double JKimu = 0;
    double JKldr = 0;
    double JKpri = 0;
    double JKfgd = 0;

    if (iter == 0 && fuse_marg)
    {
        VectorXd RIMU(RESIMU_GSIZE, 1);
        MatrixXd JIMU(RESIMU_GSIZE, XALL_GSIZE);
        RIMU.setZero(); JIMU.setZero();
        EvaluateImuFactors(traj, xr, xp, xbg, xba, SwImuBundle, imuSelected, ImuBias(BIGprior, BIAprior), RIMU, JIMU, find_factor_cost ? &JKimu : NULL);

        VectorXd RLDR(RESLDR_GSIZE, 1);
        MatrixXd JLDR(RESLDR_GSIZE, XALL_GSIZE);
        RLDR.setZero(); JLDR.setZero();
        EvaluateLdrFactors(traj, xr, xp, SwCloudDskDS, SwLidarCoef, featureSelected, RLDR, JLDR, find_factor_cost ? &JKldr : NULL);

        VectorXd RFGD(RESFGD_GSIZE, 1);
        MatrixXd JFGD(RESFGD_GSIZE, XALL_GSIZE);
        if(flat_ground)
        {
            RFGD.setZero(); JFGD.setZero();
            EvaluateFlatGroundFactors(traj, xr, xp, RFGD, JFGD, find_factor_cost ? &JKfgd : NULL);
        }

        SparseMatrix<double> bprior_final_sparse;
        SparseMatrix<double> Hprior_final_sparse;
        VectorXd bprior_final_reduced;
        MatrixXd Hprior_final_reduced;
        EvaluatePriFactors(iter, prev_knot_x, curr_knot_x, xr, xp, xbg, xba,
                           bprior_final_sparse, Hprior_final_sparse,
                           &bprior_final_reduced, &Hprior_final_reduced,
                           find_factor_cost ? &JKpri : NULL);

        // Determine the marginalized (removed) states
        map<int, int> x_knot_marg;
        for(auto &knot_x : curr_knot_x)
            if (knot_x.first < swNextBase)
                x_knot_marg[knot_x.second] = knot_x.first;

        // Find the marginalized imu residuals and the kept states
        map<int, int> x_knot_keep;

        vector<int> res_imu_marg;
        for(int idx = 0; idx < imuSelected.size(); idx++)
        {
            int i = imuSelected[idx].i;
            int j = imuSelected[idx].j;
            int k = imuSelected[idx].k;

            double sample_time = SwImuBundle[i][j][k].t;

            auto   us = traj.computeTIndex(sample_time);
            double u  = us.first;
            int    s  = us.second;

            if (x_knot_marg.find(s) != x_knot_marg.end())
            {
                for(int row_idx = 0; row_idx < RESIMU_ROW; row_idx++)
                    res_imu_marg.push_back(idx*RESIMU_ROW + row_idx);

                for(int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                    if(x_knot_marg.find(knot_idx) == x_knot_marg.end())
                        x_knot_keep[knot_idx] = curr_knot_x.begin()->first + knot_idx;    
            }
        }

        // Find the marginalized lidar residuals
        vector<int> res_ldr_marg;
        for (int idx = 0; idx < featureSelected.size(); idx++)
        {
            int i     = featureSelected[idx].wdidx;
            int k     = featureSelected[idx].pointidx;
            int depth = featureSelected[idx].depth;
            // int absidx = featureSelected[idx].absidx;

            auto &point = SwCloudDskDS[i]->points[k];
            int  point_idx = (int)(point.intensity);
            int  coeff_idx = k;

            LidarCoef &coef = SwLidarCoef[i][coeff_idx];

            double sample_time = coef.t;

            auto   us = traj.computeTIndex(sample_time);
            double u  = us.first;
            int    s  = us.second;

            if (x_knot_marg.find(s) != x_knot_marg.end())
            {
                for(int row_idx = 0; row_idx < RESLDR_ROW; row_idx++)
                    res_ldr_marg.push_back(idx*RESLDR_ROW + row_idx);

                for(int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                    if(x_knot_marg.find(knot_idx) == x_knot_marg.end())
                        x_knot_keep[knot_idx] = curr_knot_x.begin()->first + knot_idx;    
            }
        }

        // Find the marginalized fgd residuals
        vector<int> res_fgd_marg;
        if (flat_ground)
            for (int kidx = 0; kidx < traj.numKnots() - SPLINE_N + 1; kidx++)
            {
                double sample_time = traj.minTime() + (kidx + 0.5)*traj.getDt();
                auto   us = traj.computeTIndex(sample_time);
                double u  = us.first;
                int    s  = us.second;
    
                if (x_knot_marg.find(s) != x_knot_marg.end())
                {
                    for(int row_idx = 0; row_idx < RESFGD_ROW; row_idx++)
                        res_fgd_marg.push_back(kidx*RESFGD_ROW + row_idx);
    
                    for(int knot_idx = s; knot_idx < s + SPLINE_N; knot_idx++)
                        if(x_knot_marg.find(knot_idx) == x_knot_marg.end())
                            x_knot_keep[knot_idx] = curr_knot_x.begin()->first + knot_idx;    
                }
            }

        int MARG_ROW_SIZE = res_imu_marg.size() + res_ldr_marg.size() + res_fgd_marg.size();
        int MK_XSE3_SIZE = (x_knot_marg.size() + x_knot_keep.size())*XSE3_SIZE;

        // Extract the rows
        MatrixXd Jmarg = MatrixXd::Zero(MARG_ROW_SIZE, MK_XSE3_SIZE + XBGA_SIZE);
        MatrixXd rmarg = MatrixXd::Zero(MARG_ROW_SIZE, 1);
        int row_marg = 0;
        for(int row : res_imu_marg)
        {
            Jmarg.row(row_marg) << JIMU.block(row, 0, 1, MK_XSE3_SIZE), JIMU.block(row, XBGA_GBASE, 1, XBGA_SIZE) ;
            rmarg.row(row_marg) = RIMU.row(row);
            row_marg++;
        }
        for(int row : res_ldr_marg)
        {
            Jmarg.row(row_marg) << JLDR.block(row, 0, 1, MK_XSE3_SIZE), JLDR.block(row, XBGA_GBASE, 1, XBGA_SIZE) ;
            rmarg.row(row_marg) = RLDR.row(row);
            row_marg++;
        }
        for(int row : res_fgd_marg)
        {
            Jmarg.row(row_marg) << JFGD.block(row, 0, 1, MK_XSE3_SIZE), JFGD.block(row, XBGA_GBASE, 1, XBGA_SIZE) ;
            rmarg.row(row_marg) = RFGD.row(row);
            row_marg++;
        }

        // Calculate the post-optimization hessian and gradient
        SparseMatrix<double> Jsparse = Jmarg.sparseView(); Jsparse.makeCompressed();
        SparseMatrix<double> Jtp = Jsparse.transpose();
        SparseMatrix<double> H = Jtp*Jsparse;
        MatrixXd b = -Jtp*rmarg;

        // Determine whether the kept states can be distributed to the current hessian
        bool all_kept_retained = true;
        vector<pair<int, int>> knot_x_keep_retained;
        for (auto &knot_x : knot_x_keep)
        {
            if(x_knot_marg.find(knot_x.second) != x_knot_marg.end()
                || x_knot_keep.find(knot_x.second) != x_knot_keep.end())
                knot_x_keep_retained.push_back(knot_x);
            else
                all_kept_retained = false;
        }
        //Remember that the biases certainly go to the knot_x_keep states
        if (!all_kept_retained)
            printf(KRED"Knots in marg state not retained!\n"RESET);

        // Copy the previous marginalization observation to the current one
        if (Hprior_final_reduced.rows() != 0 && bprior_final_reduced.rows() !=0
            && all_kept_retained == true)
        {
            int XSE3_OLDPR_SIZE = knot_x_keep_retained.size()*XSE3_SIZE;

            // printf("Hpr %d x %d. XSE3_OLDPR_SIZE: %d. MK_XSE3_SIZE: %d. keep_retained: %d\n",
            //         Hprior_final_reduced.rows(), Hprior_final_reduced.rows(), XSE3_OLDPR_SIZE, MK_XSE3_SIZE, knot_x_keep_retained.size()*XSE3_SIZE);
            
            InsertZeroCol(Hprior_final_reduced, XSE3_OLDPR_SIZE, MK_XSE3_SIZE - knot_x_keep_retained.size()*XSE3_SIZE);
            InsertZeroRow(Hprior_final_reduced, XSE3_OLDPR_SIZE, MK_XSE3_SIZE - knot_x_keep_retained.size()*XSE3_SIZE);
            InsertZeroRow(bprior_final_reduced, XSE3_OLDPR_SIZE, MK_XSE3_SIZE - knot_x_keep_retained.size()*XSE3_SIZE);

            // printf("Hpr %d x %d. H: %d x %d. Hfn: %d x %d. XSE3_OLDPR_SIZE: %d. MK_XSE3_SIZE: %d. keep_retained: %d\n",
            //         Hprior_final_reduced.rows(), Hprior_final_reduced.cols(),
            //         H.rows(), H.cols(),
            //         Hprior_final_sparse.rows(),
            //         Hprior_final_sparse.cols(),
            //         XSE3_OLDPR_SIZE, MK_XSE3_SIZE, knot_x_keep_retained.size()*XSE3_SIZE);

            // // Sanity check
            // MatrixXd Hcheck = Hprior_final_reduced - Hprior_final_sparse;
            // MatrixXd bcheck = bprior_final_reduced - bprior_final_sparse;
            // assert(Hcheck.cwiseAbs().maxCoeff() < 1e-6);
            // assert(bcheck.cwiseAbs().maxCoeff() < 1e-6);

            SparseMatrix<double> H_extr_sparse = Hprior_final_reduced.sparseView(); H_extr_sparse.makeCompressed();
            SparseMatrix<double> b_extr_sparse = bprior_final_reduced.sparseView(); b_extr_sparse.makeCompressed();

            H += H_extr_sparse;
            b += b_extr_sparse;
        }

        // Divide the Hessian into corner blocks
        int MARG_GSIZE = x_knot_marg.size()*XSE3_SIZE;
        int KEEP_GSIZE = H.cols() - MARG_GSIZE;
        SparseMatrix<double> Hmm = H.block(0, 0, MARG_GSIZE, MARG_GSIZE);
        SparseMatrix<double> Hmk = H.block(0, MARG_GSIZE, MARG_GSIZE, KEEP_GSIZE);
        SparseMatrix<double> Hkm = H.block(MARG_GSIZE, 0, KEEP_GSIZE, MARG_GSIZE);
        SparseMatrix<double> Hkk = H.block(MARG_GSIZE, MARG_GSIZE, KEEP_GSIZE, KEEP_GSIZE);

        MatrixXd bm = b.block(0, 0, MARG_GSIZE, 1);
        MatrixXd bk = b.block(MARG_GSIZE, 0, KEEP_GSIZE, 1);

        // Save the schur complement
        MatrixXd Hmminv = Hmm.toDense().inverse();
        MatrixXd HkmHmminv = Hkm*Hmminv;
        Hkeep = Hkk - HkmHmminv*Hmk;
        bkeep = bk  - HkmHmminv*bm;

        // Convert Hb to Jr for easier use
        HbToJr(Hkeep, bkeep, Jm, rm);

        // Store the knot_x_keep for reference in the next loop
        knot_x_keep.clear(); xse3_keep.clear();
        for(auto &x_knot : x_knot_keep)
        {
            // The swNextBase will be the first state in the next sliding window
            knot_x_keep.push_back(make_pair(x_knot.second, x_knot.second - swNextBase));
            xse3_keep.push_back(traj.getKnot(x_knot.first));
        }
        xbig_keep = xbg;
        xbia_keep = xba;

        // Reset the prior if there is numerical issue
        if (Hkeep.hasNaN() || bkeep.hasNaN())
        {
            printf(KYEL "Schur Complement has NaN!\n" RESET);
            knot_x_keep.clear();
            Hkeep.setZero();
            bkeep.setZero();
        }
    }

    /* #endregion */ tt_marg.Toc();


    /* #region  */ TicToc tt_report;

    // Drafting the report

    report.surfFactors = ldr_factors;
    report.J0Surf = J0ldr;
    report.JKSurf = JKldr;
    
    report.imuFactors = imu_factors;
    report.J0Imu = J0imu;
    report.JKImu = JKimu;

    report.propFactors = 1;
    report.J0Prop = J0pri;
    report.JKProp = JKpri;

    report.velFactors = 0;
    report.J0Vel = -1;
    report.JKVel = -1;

    static double dj_thres = -1;
    if (dj_thres < 0)
        nh->param("/dj_thres", dj_thres, 0.05);

    report.J0 = J0ldr + J0imu + J0pri;
    report.JK = JKldr + JKimu + JKpri;

    /* #endregion */ tt_report.Toc();


    tt_bsu.Toc();

    bsu_report += "prep: "    + myprintf("%4.1f. ", tt_prep.GetLastStop());
    bsu_report += "hprior: "  + myprintf("%4.1f. ", tt_hprior.GetLastStop());
    bsu_report += "himu: "    + myprintf("%4.1f. ", tt_himu.GetLastStop());
    bsu_report += "hlidar: "  + myprintf("%4.1f. ", tt_hlidar.GetLastStop());
    bsu_report += "compute: " + myprintf("%4.1f. ", tt_compute.GetLastStop());
    bsu_report += "tt_marg: " + myprintf("%4.1f. ", tt_marg.GetLastStop());
    bsu_report += "bsu: "     + myprintf("%4.1f. ", tt_bsu.GetLastStop());
    bsu_report += "ldrF: "    + myprintf("%d, ",    ldr_factors);
    bsu_report += "|dX|: "    + myprintf("%7.3f. ", dX.norm());
    bsu_report += "|Hsc|: "   + myprintf("%7.3f. ", Hkeep.trace());
    bsu_report += "\n";

    tlog.t_prep.push_back(tt_prep.GetLastStop());
    tlog.t_hprior.push_back(tt_hprior.GetLastStop());
    tlog.t_himu.push_back(tt_himu.GetLastStop());
    tlog.t_hlidar.push_back(tt_hlidar.GetLastStop());
    tlog.t_compute.push_back(tt_compute.GetLastStop());
    tlog.t_marg.push_back(tt_marg.GetLastStop());
    tlog.t_solve.push_back(tt_bsu.GetLastStop());    

    // Vector3d rpy = Util::Quat2YPR(xr[0].unit_quaternion());
    // bsu_report += myprintf("XPOS%02d. Pos: %9.3f, %9.3f, %9.3f. rpy: %8.3f, %8.3f, %8.3f.\n",
    //                          0, xp[0].x(), xp[0].y(), xp[0].z(), rpy(0), rpy(1), rpy(2));
    // for(int idx = traj.numKnots() - SPLINE_N; idx < traj.numKnots(); idx++)
    // {
    //     Vector3d rpy = Util::Quat2YPR(xr[idx].unit_quaternion());
    //     bsu_report += myprintf("XPOS%02d. Pos: %9.3f, %9.3f, %9.3f. rpy: %8.3f, %8.3f, %8.3f.\n",
    //                              idx, xp[idx].x(), xp[idx].y(), xp[idx].z(), rpy(0), rpy(1), rpy(2));
    // }
    // bsu_report += myprintf("XBIAS.  BIG: %7.3f, %7.3f, %7.3f. BIG: %7.3f, %7.3f, %7.3f\n",
    //                          xbg(0), xbg(1), xbg(2), xba(0), xba(1), xba(2));
    // bsu_report += "\n";

    return true;
}
