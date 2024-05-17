#pragma once

#include <Eigen/Dense>

// Basalt
#include "basalt/spline/se3_spline.h"
#include "basalt/spline/ceres_spline_helper.h"
#include "basalt/spline/ceres_local_param.hpp"

using namespace std;
using namespace Eigen;

// Shortened typedef matching character length of Vector3d and Matrix3d
typedef Eigen::Quaterniond Quaternd;
typedef Eigen::Quaterniond Quaternf;


// Shorthand for sophus objects
typedef Sophus::SO3<double> SO3d;
typedef Sophus::SE3<double> SE3d;


/* #region "Dynamic order" spline -----------------------------------------------------------------------------------*/
typedef basalt::Se3Spline<4> PoseSpline4;
typedef basalt::Se3Spline<5> PoseSpline5;
typedef basalt::Se3Spline<6> PoseSpline6;
typedef basalt::Se3Spline<7> PoseSpline7;
typedef basalt::Se3Spline<8> PoseSpline8;
typedef basalt::Se3Spline<9> PoseSpline9;

// Function to set the sline
#define SPLINE_SET(SF, ...)                    \
    switch(X)                                  \
    {                                          \
        case(4): traj4.SF(__VA_ARGS__); break; \
        case(5): traj5.SF(__VA_ARGS__); break; \
        case(6): traj6.SF(__VA_ARGS__); break; \
        case(7): traj7.SF(__VA_ARGS__); break; \
        case(8): traj8.SF(__VA_ARGS__); break; \
        case(9): traj9.SF(__VA_ARGS__); break; \
    };                                         \

// Function to query the sline
#define SPLINE_GET(SF, ...)                    \
    switch(X)                                  \
    {                                          \
        case(4): return traj4.SF(__VA_ARGS__); \
        case(5): return traj5.SF(__VA_ARGS__); \
        case(6): return traj6.SF(__VA_ARGS__); \
        case(7): return traj7.SF(__VA_ARGS__); \
        case(8): return traj8.SF(__VA_ARGS__); \
        case(9): return traj9.SF(__VA_ARGS__); \
    };                                         \

class PoseSplineX
{

private:
    PoseSpline4 traj4;
    PoseSpline5 traj5;
    PoseSpline6 traj6;
    PoseSpline7 traj7;
    PoseSpline8 traj8;
    PoseSpline9 traj9;
    int X;
    
public:

    PoseSplineX(int X_, double dt)
    :
        X(X_),
        traj4(PoseSpline4(dt)),
        traj5(PoseSpline5(dt)),
        traj6(PoseSpline6(dt)),
        traj7(PoseSpline7(dt)),
        traj8(PoseSpline8(dt)),
        traj9(PoseSpline9(dt))
    {};

    void setStartTime(double t)
    {
        SPLINE_SET(setStartTime, t);
    };

    void extendKnotsTo(double t, SE3d pose)
    {
        SPLINE_SET(extendKnotsTo, t, pose);
    }

    void knots_push_back(SE3d pose)
    {
        SPLINE_SET(knots_push_back, pose);
    }

    void genRandomTrajectory(int n)
    {
        SPLINE_SET(genRandomTrajectory, n);
    }

    int numKnots()
    {
        SPLINE_GET(numKnots);
    }

    int order()
    {
        return X;
    }

    MatrixXd blendingMatrix(bool cummulative = false)
    {
        SPLINE_GET(blendingMatrix, cummulative);
    }

    double getDt()
    {
        SPLINE_GET(getDt);
    }

    void setKnot(SE3d pose, int i)
    {
        SPLINE_SET(setKnot, pose, i);
    }

    SE3d getKnot(int i)
    {
        SPLINE_GET(getKnot, i);
    }

    double getKnotTime(int i)
    {
        SPLINE_GET(getKnotTime, i);
    }

    SO3d &getKnotSO3(int i)
    {
        SPLINE_GET(getKnotSO3, i);
    }

    Vector3d &getKnotPos(int i)
    {
        SPLINE_GET(getKnotPos, i);
    }

    double minTime()
    {
        SPLINE_GET(minTime);
    }

    double maxTime()
    {
        SPLINE_GET(maxTime);
    }

    SE3d pose(double t)
    {
        SPLINE_GET(pose, t);
    }

    Vector3d rotVelBody(double t)
    {
        SPLINE_GET(rotVelBody, t);
    }

    Vector3d transVelWorld(double t)
    {
        SPLINE_GET(transVelWorld, t);
    }

    Vector3d transAccelWorld(double t)
    {
        SPLINE_GET(transAccelWorld, t);
    }

    pair<double, size_t> computeTIndex(double t)
    {
        SPLINE_GET(computeTIndex, t);
    }

    bool TimeIsValid(double time, double tolerance = 0)
    {
        return (minTime() + tolerance < time && time < maxTime() - tolerance);
    }
};

/* #endregion "Dynamic order" spline --------------------------------------------------------------------------------*/