/**
 * This file is part of splio.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.npn at ieee dot org>,
 * Division of RPL, KTH Royal Institute of Technology
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

#include "utility.h"

// const int queueLength = 2000;

using namespace std;
using namespace Eigen;
using namespace pcl;

class M2DGRToOuster
{
private:
    // Node handler
    ros::NodeHandlePtr nh_ptr;

    ros::Subscriber velodyneCloudSub;
    ros::Publisher ousterCloudPub;

    int NUM_CORE;

    // bool remove_human_body = true;

public:
    // Destructor
    ~M2DGRToOuster() {}

    M2DGRToOuster(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        NUM_CORE = omp_get_max_threads();

        // int remove_human_body_;
        // nh_ptr->param("remove_human_body", remove_human_body_, 0);
        // remove_human_body = (remove_human_body_ == 0)?false:true;

        velodyneCloudSub = nh_ptr->subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 50, &M2DGRToOuster::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        ousterCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/os_cloud_node/points", 50);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msgIn)
    {
        // Check if the pointcloud has time
        static int has_time = -1;
        if (has_time == -1)
        {
            for(auto &field : msgIn->fields)
                if (field.name == "time")
                {
                    has_time = 1;
                    break;
                }

            if (has_time == -1)
                has_time = 0;

            printf(KRED "Msg has no time field\n" RESET);
        }

        CloudVelodyne laserCloudVelodyne;
        pcl::fromROSMsg(*msgIn, laserCloudVelodyne);

        int cloudsize = laserCloudVelodyne.size();

        CloudOuster laserCloudOuster;
        laserCloudOuster.points.resize(cloudsize);
        laserCloudOuster.is_dense = true;

        static double hsToOusterIntensity = 1.0;

        #pragma omp parallel for num_threads(NUM_CORE)
        for (size_t i = 0; i < cloudsize; i++)
        {
            auto &src = laserCloudVelodyne.points[i];
            auto &dst = laserCloudOuster.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity * hsToOusterIntensity;
            dst.ring = src.ring;
            dst.t = (src.time + 0.1) * 1e9f;
            dst.range = sqrt(src.x * src.x + src.y * src.y + src.z * src.z)*1000.0;
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

    M2DGRToOuster C2P(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}