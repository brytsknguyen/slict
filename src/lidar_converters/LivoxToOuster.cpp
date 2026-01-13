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

#include "livox_ros_driver2/msg/custom_msg.hpp"

// const int queueLength = 2000;

using namespace std;
using namespace Eigen;
using namespace pcl;

typedef livox_ros_driver2::msg::CustomMsg LivoxCustomMsg;

class LivoxToOuster
{
private:
    // Node handler
    RosNodeHandlePtr nh_ptr;

    rclcpp::Subscription<LivoxCustomMsg>::SharedPtr livoxCloudSub;
    rclcpp::Publisher<RosPc2Msg>::SharedPtr ousterCloudPub;

    double intensityConvCoef = -1;

    int NUM_CORE;

    // bool remove_human_body = true;

public:
    // Destructor
    ~LivoxToOuster() {}

    LivoxToOuster(RosNodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        NUM_CORE = omp_get_max_threads();

        // Coefficient to convert the intensity from livox to ouster
        intensityConvCoef = 1.0;
        Util::GetParam(nh_ptr, "intensityConvCoef", intensityConvCoef);

        // Create subscription of livox cloud
        livoxCloudSub = nh_ptr->create_subscription<LivoxCustomMsg>("/livox/lidar", 50, std::bind(&LivoxToOuster::cloudHandler, this, std::placeholders::_1));
        // Create publisher of ouster cloud
        ousterCloudPub = nh_ptr->create_publisher<RosPc2Msg>("/livox/lidar_ouster", 50);
    }

    void cloudHandler(const LivoxCustomMsg::ConstSharedPtr &msgIn)
    {
        int cloudsize = msgIn->points.size();

        CloudOuster laserCloudOuster;
        laserCloudOuster.points.resize(cloudsize);
        laserCloudOuster.is_dense = true;

        #pragma omp parallel for num_threads(NUM_CORE)
        for (size_t i = 0; i < cloudsize; i++)
        {
            auto &src = msgIn->points[i];
            auto &dst = laserCloudOuster.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.reflectivity * intensityConvCoef;
            dst.ring = src.line;
            dst.t = src.offset_time;
            dst.range = sqrt(src.x * src.x + src.y * src.y + src.z * src.z)*1000;
        }

        Util::publishCloud(ousterCloudPub, laserCloudOuster, msgIn->header.stamp, msgIn->header.frame_id);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    RosNodeHandlePtr nh_ptr = rclcpp::Node::make_shared("livox_to_ouster");

    RINFO(KGRN "----> Livox to Ouster started" RESET);
    LivoxToOuster C2P(nh_ptr);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(nh_ptr);
    executor.spin();

    return 0;
}