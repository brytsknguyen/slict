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

class BPearlToOuster
{
private:
    // Node handler
    ros::NodeHandlePtr nh_ptr;

    ros::Subscriber bpearlCloudSub;
    ros::Publisher ousterCloudPub;

    bool remove_human_body = true;

public:
    // Destructor
    ~BPearlToOuster() {}

    BPearlToOuster(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        bpearlCloudSub = nh_ptr->subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 50, &BPearlToOuster::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        ousterCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/os_cloud_node/points", 50);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msgIn)
    {
        pcl::PointCloud<PointBPearl> laserCloudBPearl;
        pcl::fromROSMsg(*msgIn, laserCloudBPearl);
        
        int cloudsize = laserCloudBPearl.size();

        CloudOuster laserCloudOuster;
        laserCloudOuster.points.resize(cloudsize);
        laserCloudOuster.is_dense = true;

        static double hsToOusterIntensity = 1500.0/255.0;

        #pragma omp parallel for num_threads(MAX_THREADS)
        // double max_intensity = -1;
        for (size_t i = 0; i < cloudsize; i++)
        {
            // printf("Scan start %f. Scan end: %f. PC Stamp: %f\n",
            //         laserCloudBPearl.front().timestamp, laserCloudBPearl.back().timestamp,
            //         msgIn->header.stamp.toSec());

            auto &src = laserCloudBPearl.points[i];
            auto &dst = laserCloudOuster.points[i];
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity * hsToOusterIntensity;
            dst.ring = src.ring;
            dst.t = (int)((src.timestamp- laserCloudBPearl.points[0].timestamp) * 1e9f);
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

        Util::publishCloud(ousterCloudPub, laserCloudOuster,
                           msgIn->header.stamp - ros::Duration(0.1),
                           msgIn->header.frame_id);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "bpearl_to_ouster");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> BPearl to Ouster started" RESET);

    BPearlToOuster B2O(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}