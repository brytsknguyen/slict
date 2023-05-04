/**
* This file is part of SLICT.
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
// Created by Thien-Minh Nguyen on 01/08/22.
//

#pragma once

#ifndef _Util_LIDAR_ODOMETRY_H_
#define _Util_LIDAR_ODOMETRY_H_

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <random>
#include <mutex>

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/search/impl/search.hpp>
// #include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/common/common.h>
// #include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
// #include <pcl/filters/filter.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "glob.h"

#include <ufo/math/vector3.h>
#include <ufo/map/point_cloud.h>
#include <ufo/map/surfel_map.h>

// #include <sophus/se3.hpp>

// Ceres
#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET "\033[0m"

#define yolo() printf("Hello line: %s:%d. \n", __FILE__ , __LINE__);
#define yolos(...) printf("Hello line: %s:%d. ", __FILE__, __LINE__); printf(__VA_ARGS__); std::cout << std::endl;
#define MAX_THREADS std::thread::hardware_concurrency()

/* #region  Custom point type definition --------------------------------------------------------*/

struct PointOuster
{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t  ring;
    // uint16_t ambient; // Available in NTU VIRAL and multicampus datasets
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointOuster,
                                 (float, x, x) (float, y, y) (float, z, z)
                                 (float, intensity, intensity)
                                 (uint32_t, t, t)
                                 (uint16_t, reflectivity, reflectivity)
                                 (uint8_t,  ring, ring)
                                //  (uint16_t, ambient, ambient)
                                 (uint32_t, range, range))

struct PointVelodyne
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PointVelodyne,
                                  (float, x, x) (float, y, y) (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (float, time, time))

struct PointHesai
{
    PCL_ADD_POINT4D
    float intensity;
    double timestamp;
    uint16_t ring;                   ///< laser ring number
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointHesai,
                                 (float, x, x) (float, y, y) (float, z, z)
                                 (float, intensity, intensity)
                                 (double, timestamp, timestamp)
                                 (uint16_t, ring, ring))

struct PointTQXYZI 
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;              // preferred way of adding a XYZ+padding
    double t;
    float  qx;
    float  qy;
    float  qz;
    float  qw;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(PointTQXYZI,
                                 (float,  x, x) (float,  y, y) (float,  z, z)
                                 (float,  intensity, intensity)
                                 (double, t,  t)
                                 (float,  qx, qx)
                                 (float,  qy, qy)
                                 (float,  qz, qz)
                                 (float,  qw, qw))

struct PointXYZIT
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    float t;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
                                 (float, x, x) (float, y, y) (float, z, z)
                                 (float, intensity, intensity) (float, t, t))

typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointXYZI PointXYZI;
typedef PointTQXYZI PointPose;

typedef pcl::PointCloud<PointXYZ> CloudXYZ;
typedef pcl::PointCloud<PointXYZI> CloudXYZI;
typedef pcl::PointCloud<PointXYZIT> CloudXYZIT;
typedef pcl::PointCloud<PointPose> CloudPose;
typedef pcl::PointCloud<PointOuster> CloudOuster;
typedef pcl::PointCloud<PointVelodyne> CloudVelodyne;

typedef pcl::PointCloud<PointXYZ>::Ptr CloudXYZPtr;
typedef pcl::PointCloud<PointXYZI>::Ptr PointCloudPtr;
typedef pcl::PointCloud<PointXYZI>::Ptr CloudXYZIPtr;
typedef pcl::PointCloud<PointXYZIT>::Ptr CloudXYZITPtr;
typedef pcl::PointCloud<PointPose>::Ptr CloudPosePtr;
typedef pcl::PointCloud<PointOuster>::Ptr CloudOusterPtr;
typedef pcl::PointCloud<PointVelodyne>::Ptr CloudVelodynePtr;

typedef pcl::KdTreeFLANN<PointXYZI> KdFLANN;
typedef pcl::KdTreeFLANN<PointXYZI>::Ptr KdFLANNPtr;

/* #endregion  Custom point type definition -----------------------------------------------------*/


/* #region  Image pointer shortened name -------*/

typedef sensor_msgs::Image::Ptr RosImgPtr;
typedef sensor_msgs::Image::ConstPtr RosImgConstPtr;
// typedef map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> VisFeatureFrame;
// typedef pair<double, VisFeatureFrame> StampedVisFeatureFrame;

/* #endregion Image pointer shortened name -----*/

// Shortened typedef matching character length of Vector3d and Matrix3d
typedef Eigen::Quaterniond Quaternd;
typedef Eigen::Quaterniond Quaternf;

template <typename T>
void pointConverter(Eigen::MatrixXd &base2LidarTf, T &point)
{
    Eigen::Vector3d p_lidar(point.x, point.y, point.z);
    Eigen::Vector3d p_baselink = base2LidarTf.block<3, 3>(0, 0) * p_lidar + base2LidarTf.block<3, 1>(0, 3);

    point.x = p_baselink.x();
    point.y = p_baselink.y();
    point.z = p_baselink.z();
}

class TicToc
{
public:
    TicToc() : elapsed_seconds_(0)
    {
        Tic();
    }

    void Tic()
    {
        start_ = std::chrono::system_clock::now();
    }

    double Toc()
    {
        end_ = std::chrono::system_clock::now();
        elapsed_seconds_ = end_ - start_;
        return elapsed_seconds_.count() * 1000;
    }

    double GetLastStop()
    {
        return elapsed_seconds_.count() * 1000;
    }

#define LASTSTOP(x) printf(#x".LastStop : %f\n", x.GetLastStop());
#define TOCPRINT(x) printf(#x".Toc : %f\n", x.Toc());

private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
    std::chrono::duration<double> elapsed_seconds_;
};

template <typename T = double>
struct myTf
{
    Eigen::Quaternion<T>   rot;
    Eigen::Matrix<T, 3, 1> pos;

    myTf Identity()
    {
        return myTf();
    }
    
    myTf(const myTf<T> &other)
    {
        rot = other.rot;
        pos = other.pos;
    }

    myTf()
    {
        rot = Quaternd(1, 0, 0, 0);
        pos = Vector3d(0, 0, 0);
    }

    myTf(Eigen::Quaternion<T> rot_in, Eigen::Matrix<T, 3, 1> pos_in)
    {
        this->rot = rot_in;
        this->pos = pos_in;
    }

    myTf(Eigen::Matrix<T, 3, 3> rot_in, Eigen::Matrix<T, 3, 1> pos_in)
    {
        this->rot = Quaternion<T>(rot_in);
        this->pos = pos_in;
    }

    template <typename Tin>
    myTf(Eigen::Matrix<Tin, 4, 4> tfMat)
    {
        Eigen::Matrix<T, 3, 3> M = tfMat.block(0, 0, 3, 3).template cast<T>();
        this->rot = Quaternion<T>(M);
        this->pos = tfMat.block(0, 3, 3, 1).template cast<T>();
    }

    myTf(PointPose point)
    {
        this->rot = Quaternion<T>(point.qw, point.qx, point.qy, point.qz);
        this->pos << point.x, point.y, point.z;
    }

    myTf(nav_msgs::Odometry odom)
    {
        this->rot = Quaternion<T>(odom.pose.pose.orientation.w,
                                    odom.pose.pose.orientation.x,
                                    odom.pose.pose.orientation.y,
                                    odom.pose.pose.orientation.z);
                                    
        this->pos << odom.pose.pose.position.x,
                     odom.pose.pose.position.y,
                     odom.pose.pose.position.z;
    }

    myTf(Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform)
    {
        this->rot = Eigen::Quaternion<T>{transform.linear()}.normalized();
        this->pos = transform.translation();
    }

    Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform() const
    {
        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform;
        transform.linear() = rot.normalized().toRotationMatrix();
        transform.translation() = pos;
        return transform;
    }

    Eigen::Matrix<T, 4, 4> tfMat() const
    {
        Eigen::Matrix<T, 4, 4> M = Matrix<T, 4, 4>::Identity();
        M.block(0, 0, 3, 3) = rot.normalized().toRotationMatrix();
        M.block(0, 3, 3, 1) = pos;
        return M;
    }

    PointXYZI Point3D() const
    {
        PointXYZI p;

        p.x = (float)pos.x();
        p.y = (float)pos.y();
        p.z = (float)pos.z();

        p.intensity = -1;

        return p;
    }
    
    PointPose Pose6D() const
    {
        PointPose p;

        p.t = -1;

        p.x = (float)pos.x();
        p.y = (float)pos.y();
        p.z = (float)pos.z();

        p.qx = (float)rot.x();
        p.qy = (float)rot.y();
        p.qz = (float)rot.z();
        p.qw = (float)rot.w();

        p.intensity = -1;

        return p;
    }

    PointPose Pose6D(double time) const
    {
        PointPose p;

        p.t = time;
        
        p.x = (float)pos.x();
        p.y = (float)pos.y();
        p.z = (float)pos.z();

        p.qx = (float)rot.x();
        p.qy = (float)rot.y();
        p.qz = (float)rot.z();
        p.qw = (float)rot.w();

        p.intensity = -1;

        return p;
    }

    // template <typename Tout = double>
    // Sophus::SE3<Tout> getSE3() const
    // {
    //     return Sophus::SE3<Tout>(this->rot.template cast<Tout>(),
    //                              this->pos.template cast<Tout>());
    // }

    double roll() const
    {
        return atan2(rot.x()*rot.w() + rot.y()*rot.z(), 0.5 - (rot.x()*rot.x() + rot.y()*rot.y()))/M_PI*180.0;
    }

    double pitch() const
    {
        return asin(-2*(rot.x()*rot.z() - rot.w()*rot.y()))/M_PI*180.0;
    }

    double yaw() const
    {
        return atan2(rot.x()*rot.y() + rot.w()*rot.z(), 0.5 - (rot.y()*rot.y() + rot.z()*rot.z()))/M_PI*180.0;
    }

    myTf inverse() const
    {
        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform_inv = this->transform().inverse();
        myTf tf_inv;
        tf_inv.rot = transform_inv.linear();
        tf_inv.pos = transform_inv.translation();
        return tf_inv;
    }

    myTf operator*(const myTf &other) const
    {
        Eigen::Transform<T, 3, Eigen::TransformTraits::Affine> transform_out = this->transform() * other.transform();
        return myTf(transform_out);
    }

    Vector3d operator*(const Vector3d &v) const
    {
        return (rot*v + pos);
    }

    template <typename NewType>
    myTf<NewType> cast() const
    {
        myTf<NewType> tf_new{this->rot.template cast<NewType>(), this->pos.template cast<NewType>()};
        return tf_new;
    }

    friend std::ostream &operator<<(std::ostream &os, const myTf &tf)
    {
        os << tf.pos.x() << " " << tf.pos.y() << " " << tf.pos.z() << " " << tf.rot.w() << " "
           << tf.rot.x() << " " << tf.rot.y() << " " << tf.rot.z();
        return os;
    }
}; // class myTf

typedef myTf<> mytf;

class ImuSample
{
public:

    ImuSample();
    ImuSample(ImuSample const& other)
    {
        t = other.t;
        u = other.u;
        s = other.s;
        gyro = other.gyro;
        acce = other.acce;
    }
    ImuSample(double t_, Vector3d gyro_, Vector3d acce_)
        : t(t_), gyro(gyro_), acce(acce_) {};

    double t;
    double u = -1;
    double s = -1;
    Vector3d gyro = Vector3d(0, 0, 0);
    Vector3d acce = Vector3d(0, 0, 0);
};

class ImuSequence
{
public:
    
   ~ImuSequence()
    {
        data.clear();
    }

    ImuSequence()
    {
        data = deque<ImuSample>();
    }

    ImuSequence(ImuSequence const &other)
    {
        data = other.data;
    }

    ImuSequence(deque<ImuSample> const &data_)
    {
        data = data_;
    }

    void push_back(ImuSample const &imuSample)
    {
        data.push_back(imuSample);
    }

    void push_front(ImuSample const &imuSample)
    {
        data.push_front(imuSample);
    }

    ImuSequence subsequence(double start_time, double final_time)
    {
        ImuSequence newSequence;
        for(int i = 0; i < data.size(); i++ )
        {
            if (start_time < data[i].t && data[i].t <= final_time)
            {
                newSequence.push_back(data[i]);
            }
        }

        if (newSequence.front().t != start_time)
            newSequence.push_front(this->interpolate(start_time));

        if (newSequence.back().t != final_time)
            newSequence.push_back(this->interpolate(final_time));

        return newSequence;    
    }

    ImuSample interpolate(double t)
    {
        if (t < data.front().t)
        {
            printf(KMAG "Point time %f is earlier than [%f, %f]. "
                        "Returning sample at start time.\n" RESET, t, data.front().t, data.back().t);

            return data.front();    
        }

        if (t > data.back().t)
        {
            printf(KMAG "Point time %f is later than [%f, %f]. "
                        "Returning sample at start time.\n" RESET, t, data.front().t, data.back().t);

            return data.back();    
        }

        for(int i = 0; i < data.size(); i++)
        {
            // Check if we can return the sample at the exact time
            if (data[i].t == t)
                return data[i];
            
            if (data[i].t < t && t < data[i+1].t)
            {
                double s = (t - data[i].t)/(data[i+1].t - data[i].t);
                
                Vector3d gyro = (1-s)*data[i].gyro + s*data[i+1].gyro;
                Vector3d acce = (1-s)*data[i].acce + s*data[i+1].acce;

                return ImuSample(t, gyro, acce);
            }
        }
    }

    ImuSample& operator[](size_t i)
    {
        return data[i];
    }

    void operator+=(ImuSequence const &s)
    {
        if (this->data.size() == 0)
            this->data = s.data;
        else
        {
            for(auto sample : s.data)
            {
                if (this->data.back().t < sample.t)
                    this->data.push_back(sample);
            }
        }
    }

    ImuSequence operator+(const ImuSequence &s)
    {
        ImuSequence s_(*this); s_ += s;
        return s_;
    }
    
    ImuSample& front()
    {
        return data.front();
    }
    
    ImuSample& back()
    {
        return data.back();
    }

    size_t size()
    {
        return data.size();
    }

    bool empty()
    {
        return data.empty();
    }
    
    double startTime()
    {
        return data.front().t;
    }

    double finalTime()
    {
        return data.back().t;
    }

    deque<ImuSample> data;
};

class ImuBias
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuBias();
    ImuBias(ImuBias const& other)
    {
        gyro_bias = other.gyro_bias;
        acce_bias = other.acce_bias;
    }
    ImuBias(Vector3d gyro_bias_, Vector3d acce_bias_)
        : gyro_bias(gyro_bias_), acce_bias(acce_bias_) {};

    Vector3d gyro_bias = Vector3d(0, 0, 0);
    Vector3d acce_bias = Vector3d(0, 0, 0);
};

struct LidarCoef
{
    double   t   = -1;    // Time stamp
    int      ptIdx;       // Index of points in the pointcloud
    Vector3d f;           // Coordinate of point in body frame
    Vector3d fdsk;        // Coordinate of point in body frame, deskewed
    Vector3d finW;        // Coordinate of point in world frame, for vizualization
    Vector4d n;           // Score-multipled Hesse normal vector
    int scale;            // Scale at which the associated surfel is at
    int surfNp;           // Number of points in the associated surfel
    double plnrty =  0;   // Surfel planarity
    double d2P    = -1;   // Distance to the plane
    double u      = -1;   // Start time of the containing segment
    double s      = -1;   // Scaled time within the used segment
    double dt     = -1;   // Time interval of the containing segment

    deque<ufo::map::Node> ufonode;
    // Clear up the buffer in the destructor
    ~LidarCoef() { ufonode.clear(); };
};

class ImuProp
{
public:

    deque<double> t;

    deque<Quaternd> Q;
    deque<Vector3d> P;
    deque<Vector3d> V;

    Vector3d bg; Vector3d ba; Vector3d grav;
    
    deque<Vector3d> gyr;
    deque<Vector3d> acc;

    ~ImuProp() {}

    ImuProp()
    {
        t = { 0 };

        Q = { Quaternd(1, 0, 0, 0) };
        P = { Vector3d(0, 0, 0) } ;
        V = { Vector3d(0, 0, 0) };

        bg = Vector3d(0, 0, 0);
        ba = Vector3d(0, 0, 0);

        gyr = { Vector3d(0, 0, 0) };
        acc = { Vector3d(0, 0, 0) };

        grav = Vector3d(0, 0, 0);
    }

    ImuProp(Quaternd Q0,  Vector3d P0,  Vector3d V0,
            Vector3d bg_, Vector3d ba_, Vector3d gyr0, Vector3d acc0, Vector3d grav_, double t0)
    {
        reset(Q0, P0, V0, bg_, ba_, gyr0, acc0, grav_, t0);
    }

    ImuProp(Quaternd &Q0, Vector3d &P0, Vector3d &V0, Vector3d &bg_, Vector3d &ba_, Vector3d &grav_, ImuSequence &imuSeq, int dir=0)
    {
        if (dir >= 0) // Forward propagation
        {
            reset(Q0, P0, V0, bg_, ba_, imuSeq.front().gyro, imuSeq.front().acce, grav_, imuSeq.front().t);
            for(int i = 1; i < imuSeq.size(); i++)
                forwardPropagate(imuSeq[i].gyro, imuSeq[i].acce, imuSeq[i].t);
        }
        else // Backward propagation
        {
            reset(Q0, P0, V0, bg_, ba_, imuSeq.back().gyro, imuSeq.back().acce, grav_, imuSeq.back().t);
            for(int i = imuSeq.size() - 1; i >= 0; i--)
                backwardPropagate(imuSeq[i].gyro, imuSeq[i].acce, imuSeq[i].t);
        }
    }

    void reset(Quaternd &Q0, Vector3d &P0, Vector3d &V0, Vector3d &bg_, Vector3d &ba_, Vector3d &gyr0, Vector3d &acc0, Vector3d &grav_, double t0)
    {
        t = { t0 };
        
        Q = { Q0 };
        P = { P0 };
        V = { V0 };

        gyr = { gyr0 };
        acc = { acc0 };

        bg = bg_; ba = ba_; grav = grav_;
    }

    void forwardPropagate(Vector3d &gyrn, Vector3d &accn, double tn)
    {
        double to = t.back();

        Quaternd Qo = Q.back();
        Vector3d Po = P.back();
        Vector3d Vo = V.back();

        Vector3d gyro = gyr.back();
        Vector3d acco = acc.back();

        // Time step
        double dt = tn - to;

        // Orientation
        Vector3d un_gyr = 0.5 * (gyro + gyrn) - bg;
        Quaternd Qn = Qo * deltaQ(un_gyr * dt);

        // Position
        Vector3d un_acco = Qo * (acco - ba) - grav;
        Vector3d un_accn = Qn * (accn - ba) - grav;
        Vector3d un_acc  = 0.5 * (un_acco + un_accn);

        Vector3d Pn = Po + dt * Vo + 0.5 * dt * dt * un_acc;
        Vector3d Vn = Vo + dt * un_acc;

        // Store the data
        t.push_back(tn);

        Q.push_back(Qn);
        P.push_back(Pn);
        V.push_back(Vn);

        gyr.push_back(gyrn);
        acc.push_back(accn);
    }

    void backwardPropagate(Vector3d &gyro, Vector3d &acco, double to)
    {
        double tn = t.front();

        Quaternd Qn = Q.front();
        Vector3d Pn = P.front();
        Vector3d Vn = V.front();

        Vector3d gyrn = gyr.front();
        Vector3d accn = acc.front();

        // Time step
        double dt = tn - to;

        // Orientation
        Vector3d un_gyr = 0.5 * (gyro + gyrn) - bg;
        Quaternd Qo = Qn * deltaQ(-un_gyr * dt);

        // Position
        Vector3d un_acco = Qo * (acco - ba) - grav;
        Vector3d un_accn = Qn * (accn - ba) - grav;
        Vector3d un_acc  = 0.5 * (un_acco + un_accn);

        Vector3d Vo = Vn - dt * un_acc;
        Vector3d Po = Pn - dt * Vo - 0.5 * dt * dt * un_acc;

        // Store the data
        t.push_front(to);

        Q.push_front(Qo);
        P.push_front(Po);
        V.push_front(Vo);

        gyr.push_front(gyro);
        acc.push_front(acco);
    }

    mytf getFrontTf()
    {
        return (mytf(Q.front(), P.front()));
    }

    mytf getBackTf()
    {
        return (mytf(Q.back(), P.back()));
    }

    mytf getTf(double ts)
    {
        ROS_ASSERT(t.size() >= 1);

        if (ts < t.front())
        {
            // if ( fabs(ts - t.front()) > 5e-3 )
                printf(KYEL "Point time %.6f is earlier than [%.6f, %.6f]. "
                            "Returning pose at start time.\n" RESET, ts, t.front(), t.back());

            return (mytf(Q.front(), P.front()));
        }
        else if (ts > t.back())
        {
            // if ( fabs(ts - time.back()) > 5e-3 )
                printf(KYEL "Point time %.6f is later than [%.6f, %.6f]. "
                            "Returning pose at end time.\n" RESET, ts, t.front(), t.back());
            return (mytf(Q.back(), P.back()));
        }
        else
        {
            // Find the pose that fit the time
            for(int i = 0; i < t.size() - 1; i++)
            {
                if( t[i] <= ts && ts <= t[i+1] )
                {
                    double s = (ts - t[i])/(t[i+1] - t[i]);
                    Quaternd Qs = Q[i]*Quaternd::Identity().slerp(s, Q[i].inverse()*Q[i+1]);
                    Vector3d Ps = (1 - s)*P[i] + s*P[i+1];

                    return (mytf(Qs, Ps));   
                }
            }
        }
    }

    double getStartTime()
    {
        return t.front();
    }

    double getEndTime()
    {
        return t.back();
    }

    Vector3d getStartV()
    {
        return V.front();
    }

    Vector3d getEndV()
    {
        return V.back();
    }

    void setG(Vector3d &g)
    {
        grav = g;
    }

    int size()
    {
        return t.size();
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;

        Scalar_t theta_nrm = theta.norm();

        if (theta_nrm < 1e-5)
        {
            Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
            half_theta /= static_cast<Scalar_t>(2.0);
            dq.w() = static_cast<Scalar_t>(1.0);
            dq.x() = half_theta.x();
            dq.y() = half_theta.y();
            dq.z() = half_theta.z();
        }
        else
        {
            Scalar_t costheta = cos(theta_nrm / 2);
            Scalar_t sintheta = sin(theta_nrm / 2);
            Eigen::Matrix<Scalar_t, 3, 1> quat_vec = theta / theta_nrm * sintheta;

            dq.w() = costheta;
            dq.vec() = quat_vec;
        }

        return dq;
    }
};

// Interfacing ufo::map data with Eigen
namespace ufo::math
{
    template<typename T>
    Eigen::Vector3d toEigen(ufo::math::Vector3<T> v)
    {
        return Eigen::Vector3d(v.x, v.y, v.z);
    }
}

template <typename PointType>
void insertCloudToSurfelMap(ufo::map::SurfelMap &map,
                            pcl::PointCloud<PointType> &pclCloud)
{
    int cloudSize = pclCloud.size();

    ufo::map::PointCloud ufoCloud;
    ufoCloud.resize(cloudSize);

    #pragma omp parallel for num_threads(omp_get_max_threads())
    for(int i = 0; i < cloudSize; i++)
    {
        ufoCloud[i].x = (float)pclCloud.points[i].x;
        ufoCloud[i].y = (float)pclCloud.points[i].y;
        ufoCloud[i].z = (float)pclCloud.points[i].z;
    }

    map.insertSurfelPoint(std::begin(ufoCloud), std::end(ufoCloud));
}

namespace Util
{
    void ComputeCeresCost(vector<ceres::internal::ResidualBlock *> &res_ids,
                          double &cost, ceres::Problem &problem)
    {
        if (res_ids.size() == 0)
        {
            cost = -1;
            return;
        }

        ceres::Problem::EvaluateOptions e_option;
        e_option.residual_blocks = res_ids;
        e_option.num_threads = omp_get_max_threads();
        problem.Evaluate(e_option, &cost, NULL, NULL, NULL);
    }

    template <typename PointType>
    sensor_msgs::PointCloud2 publishCloud(ros::Publisher &thisPub,
                                          pcl::PointCloud<PointType> &thisCloud,
                                          ros::Time thisStamp, std::string thisFrame)
    {
        sensor_msgs::PointCloud2 tempCloud;
        pcl::toROSMsg(thisCloud, tempCloud);
        tempCloud.header.stamp = thisStamp;
        tempCloud.header.frame_id = thisFrame;
        // if (thisPub.getNumSubscribers() != 0)
            thisPub.publish(tempCloud);
        return tempCloud;
    }

    static float wrapTo360(float angle)
    {
        angle = fmod(angle, 360);
        if (angle < 0)
            angle += 360;
        return angle;
    }

    static double wrapTo180(double angle)
    {
        angle = fmod(angle + 180,360);
        if (angle < 0)
            angle += 360;
        return angle - 180;
    }

    template <typename PoinT>
    static float pointDistance(PoinT p)
    {
        return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }

    template <typename Derived>
    static typename Derived::Scalar angleDiff(const Eigen::QuaternionBase<Derived> &q1, const Eigen::QuaternionBase<Derived> &q2)
    {
        return (Eigen::AngleAxis<typename Derived::Scalar>(q1.inverse()*q2).angle()*180.0/M_PI);
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;

        Scalar_t theta_nrm = theta.norm();

        if (theta_nrm < 1e-5)
        {
            Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
            half_theta /= static_cast<Scalar_t>(2.0);
            dq.w() = static_cast<Scalar_t>(1.0);
            dq.x() = half_theta.x();
            dq.y() = half_theta.y();
            dq.z() = half_theta.z();
        }
        else
        {
            Scalar_t costheta = cos(theta_nrm / 2);
            Scalar_t sintheta = sin(theta_nrm / 2);
            Eigen::Matrix<Scalar_t, 3, 1> quat_vec = theta / theta_nrm * sintheta;

            dq.w() = costheta;
            dq.vec() = quat_vec;
        }

        // printf("dq: %f, %f, %f, %f. norm: %f\n", dq.x(), dq.y(), dq.z(), dq.w(), dq.norm());

        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
               q(2), typename Derived::Scalar(0), -q(0),
              -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
        return ans;
    }

    static Eigen::Matrix3d SO3Jright(const Eigen::AngleAxis<double> &phi)
    {
        Eigen::Matrix3d ans;
        double theta = phi.angle();
        Eigen::Matrix3d ux = skewSymmetric(phi.axis());

        if (theta == 0.0)
        {
            ans = Eigen::Matrix3d::Identity();
        }
        else
        {
            ans = Eigen::Matrix3d::Identity() - (1 - cos(theta)) / theta * ux + (theta - sin(theta)) / theta * ux * ux;
        }

        return ans;
    }

    static Eigen::Matrix3d SO3JrightInv(const Eigen::AngleAxis<double> &phi)
    {
        Eigen::Matrix3d ans;
        double theta = phi.angle();
        Eigen::Matrix3d ux = skewSymmetric(phi.axis());

        if (theta == 0.0)
        {
            ans = Eigen::Matrix3d::Identity();
        }
        else
        {
            ans = Eigen::Matrix3d::Identity() + 0.5 * theta * ux + (1 - theta * (1 + cos(theta)) / 2 / sin(theta)) * ux * ux;
        }

        return ans;
    }

    static Eigen::Matrix3d SO3JleftInv(const Eigen::AngleAxis<double> &phi)
    {
        Eigen::Matrix3d ans;
        double theta = phi.angle();
        Eigen::Matrix3d ux = skewSymmetric(phi.axis());

        if (theta == 0.0)
        {
            ans = Eigen::Matrix3d::Identity();
        }
        else
        {
            ans = Eigen::Matrix3d::Identity() - 0.5 * theta * ux + (1 - theta * (1 + cos(theta)) / 2 / sin(theta)) * ux * ux;
        }

        return ans;
    }

    static Eigen::Matrix3d YPR2Rot(Eigen::Vector3d ypr)
    {
        double y = ypr(0) / 180.0 * M_PI;
        double p = ypr(1) / 180.0 * M_PI;
        double r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix3d Rz;
        Rz << cos(y), -sin(y), 0.0,
              sin(y),  cos(y), 0.0,
              0.0,     0.0,    1;

        Eigen::Matrix3d Ry;
        Ry << cos(p), 0.0, sin(p),
              0.0,    1.0, 0.0,
             -sin(p), 0.0, cos(p);

        Eigen::Matrix3d Rx;
        Rx << 1.0, 0.0,     0.0,
              0.0, cos(r), -sin(r),
              0.0, sin(r),  cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Vector3d Rot2YPR(const Eigen::Matrix3d R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    static Eigen::Vector3d Quat2YPR(const Eigen::Quaterniond Q)
    {
        return Rot2YPR(Q.toRotationMatrix());
    }

    // Finding the roll and pitch from gravity reading.
    static Eigen::Matrix3d grav2Rot(const Eigen::Vector3d &g)
    {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = Util::Rot2YPR(R0).x();
        R0 = Util::YPR2Rot(Eigen::Vector3d(-yaw, 0, 0)) * R0;
        // R0 = Util::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
        return R0;
    }

    template <typename PointT>
    static PointT transform_point(const mytf &tf, const PointT &pi)
    {
        Vector3d pos = tf.rot * Vector3d(pi.x, pi.y, pi.z) + tf.pos;
        
        PointT po = pi;
        po.x = pos.x();
        po.y = pos.y();
        po.z = pos.z();

        return po;
    }

    template <typename PointT>
    static bool PointIsValid(const PointT &p)
    {
        return (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
                && !std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z));
    }

}; // namespace Util

#define _CRT_NO_VA_START_VALIDATION
inline std::string myprintf(const std::string& format, ...)
{
    va_list args;
    va_start(args, format);
    size_t len = std::vsnprintf(NULL, 0, format.c_str(), args);
    va_end(args);
    std::vector<char> vec(len + 1);
    va_start(args, format);
    std::vsnprintf(&vec[0], len + 1, format.c_str(), args);
    va_end(args);
    
    return string(vec.begin(), vec.end() - 1);
}

#endif