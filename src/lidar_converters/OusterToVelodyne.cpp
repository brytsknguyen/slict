
/**
 * This file is part of SLICT.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.npn at ieee dot org>,
 * Division of RPL, KTH Royal Institute of Technology
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

#include "utility.h"

// #include <livox_ros_driver/CustomMsg.h>

// const int queueLength = 2000;

using namespace std;
using namespace Eigen;
using namespace pcl;

class OusterToVelodyne
{
private:
    // Node handler
    RosNodeHandlePtr nh_ptr;

    rclcpp::Subscription<RosPc2Msg>::SharedPtr ousterCloudSub;
    rclcpp::Publisher<RosPc2Msg>::SharedPtr velodyneCloudPub;

    int NUM_CORE;

    // bool remove_human_body = true;

public:
    // Destructor
    ~OusterToVelodyne() {}

    OusterToVelodyne(RosNodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        NUM_CORE = omp_get_max_threads();

        ousterCloudSub = nh_ptr->create_subscription<RosPc2Msg>("/os_cloud_node/points", 50, &OusterToVelodyne::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        velodyneCloudPub = nh_ptr->create_publisher<RosPc2Msg>("/velodyne_points", 50);
    }

    void cloudHandler(const RosPc2Msg::ConstSharedPtr &msgIn)
    {
        CloudOuster laserCloudOuster;
        pcl::fromROSMsg(*msgIn, laserCloudOuster);

        int cloudsize = laserCloudOuster.size();

        CloudVelodyne laserCloudVelodyne;
        laserCloudVelodyne.points.resize(cloudsize);
        laserCloudVelodyne.is_dense = true;

        static double ousterToHsIntensity = 1.0;

        #pragma omp parallel for num_threads(NUM_CORE)
        for (size_t i = 0; i < cloudsize; i++)
        {
            auto &src = laserCloudOuster.points[i];
            auto &dst = laserCloudVelodyne.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity * ousterToHsIntensity;
            dst.ring = src.ring;
            dst.time = src.t * 1e-9f;  // Converting from nanoseconds to seconds
        }

        Util::publishCloud(velodyneCloudPub, laserCloudVelodyne, msgIn->header.stamp, "velodyne");
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ouster_to_velodyne");
    ros::NodeHandle nh("~");
    RosNodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    RINFO(KGRN "----> Ouster to Velodyne started" RESET);

    OusterToVelodyne O2V(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}
