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

class HesaiToOuster
{
private:
    // Node handler
    RosNodeHandlePtr nh_ptr;

    rclcpp::Subscription<RosPc2Msg>::SharedPtr hesaiCloudSub;
    rclcpp::Publisher<RosPc2Msg>::SharedPtr ousterCloudPub;

    bool remove_human_body = true;

public:
    // Destructor
    ~HesaiToOuster() {}

    HesaiToOuster(RosNodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        hesaiCloudSub = nh_ptr->create_subscription<RosPc2Msg>("/hesai/pandar", 50, std::bind(&HesaiToOuster::cloudHandler, this, std::placeholders::_1));
        ousterCloudPub = nh_ptr->create_publisher<RosPc2Msg>("/os_cloud_node/points", 50);
    }

    void cloudHandler(const RosPc2Msg::ConstSharedPtr &msgIn)
    {
        pcl::PointCloud<PointHesai> laserCloudHesai;
        pcl::fromROSMsg(*msgIn, laserCloudHesai);

        int cloudsize = laserCloudHesai.size();

        CloudOuster laserCloudOuster;
        laserCloudOuster.points.resize(cloudsize);
        laserCloudOuster.is_dense = true;

        static double hsToOusterIntensity = 1500.0/255.0;

        #pragma omp parallel for num_threads(MAX_THREADS)
        // double max_intensity = -1;
        for (size_t i = 0; i < cloudsize; i++)
        {
            auto &src = laserCloudHesai.points[i];
            auto &dst = laserCloudOuster.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity * hsToOusterIntensity;
            dst.ring = src.ring;
            dst.t = (int)((src.timestamp - rclcpp::Time(msgIn->header.stamp).seconds()) * 1e9f);
            dst.range = sqrt(src.x*src.x + src.y*src.y + src.z*src.z)*1000.0;

            // Remove points on the carrier
            if(remove_human_body)
            {
                double yaw = Util::wrapTo360(atan2(dst.y, dst.x)*180/M_PI);
                if (yaw > 38 && yaw < 142 && dst.range < 0.5)
                {
                    dst.range = 0.0;
                    dst.x = dst.y = dst.z = 0;
                }
            }

            // max_intensity = (dst.intensity > max_intensity)?dst.intensity:max_intensity;
        }

        Util::publishCloud(ousterCloudPub, laserCloudOuster, msgIn->header.stamp, msgIn->header.frame_id);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    RosNodeHandlePtr nh_ptr = rclcpp::Node::make_shared("hesai_to_ouster");

    RINFO(KGRN "----> hesai to Ouster started" RESET);
    HesaiToOuster H2O(nh_ptr);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(nh_ptr);
    executor.spin();

    return 0;
}