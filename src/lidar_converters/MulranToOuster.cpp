/**
 * This file is part of slict.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.npn at ieee dot org>,
 * Division of RPL, KTH Royal Institute of Technology
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

#include "utility.h"

// const int queueLength = 2000;

using namespace std;
using namespace Eigen;
using namespace pcl;

struct PointMulran
{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    int32_t  ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointMulran,
                                 (float, x, x) (float, y, y) (float, z, z)
                                 (float, intensity, intensity)
                                 (uint32_t, t, t)
                                 (int32_t, ring, ring)
)

typedef pcl::PointCloud<PointMulran> CloudMulran;
typedef pcl::PointCloud<PointMulran>::Ptr CloudMulranPtr;

class MulranToOuster
{
private:
    // Node handler
    ros::NodeHandlePtr nh_ptr;

    ros::Subscriber mulranCloudSub;
    ros::Publisher ousterCloudPub;

    int NUM_CORE;

    // bool remove_human_body = true;

public:
    // Destructor
    ~MulranToOuster() {}

    MulranToOuster(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        NUM_CORE = omp_get_max_threads();

        mulranCloudSub = nh_ptr->subscribe<sensor_msgs::PointCloud2>("/Ouster_Points", 50, &MulranToOuster::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        ousterCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/os_cloud_node/points", 50);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msgIn)
    {
        CloudMulran laserCloudMulran;
        pcl::fromROSMsg(*msgIn, laserCloudMulran);

        int cloudsize = laserCloudMulran.size();

        CloudOuster laserCloudOuster;
        laserCloudOuster.points.resize(cloudsize);
        laserCloudOuster.is_dense = true;

        #pragma omp parallel for num_threads(NUM_CORE)
        for (size_t i = 0; i < cloudsize; i++)
        {
            auto &src = laserCloudMulran.points[i];
            auto &dst = laserCloudOuster.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity;
            dst.ring = src.ring;
            dst.t = src.t;
            dst.range = sqrt(src.x * src.x + src.y * src.y + src.z * src.z)*1000.0;
            // printf("Point %d, Time: %d\n", i, src.t);
        }

        Util::publishCloud(ousterCloudPub, laserCloudOuster, msgIn->header.stamp, msgIn->header.frame_id);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "velodyne_to_ouster");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Velodyne to Ouster started" RESET);

    MulranToOuster C2P(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}