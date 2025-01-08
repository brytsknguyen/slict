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

struct PointHelios
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    uint32_t sec;
    uint32_t usec;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PointHelios,
                                  (float, x, x) (float, y, y) (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring)
                                  (uint32_t, sec, sec)
                                  (uint32_t, usec, usec))

typedef pcl::PointCloud<PointHelios> CloudHelios;
typedef pcl::PointCloud<PointHelios>::Ptr CloudHeliosPtr;


class HeliosToOuster
{
private:
    // Node handler
    ros::NodeHandlePtr nh_ptr;

    ros::Subscriber HeliosCloudSub;
    ros::Publisher ousterCloudPub;

    int NUM_CORE;

    // bool remove_human_body = true;

public:
    // Destructor
    ~HeliosToOuster() {}

    HeliosToOuster(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        NUM_CORE = omp_get_max_threads();
        HeliosCloudSub = nh_ptr->subscribe<sensor_msgs::PointCloud2>("/cloud_helios", 50, &HeliosToOuster::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        ousterCloudPub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/helios_os_cloud_node/points", 50);
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

        CloudHelios laserCloudHelios;
        pcl::fromROSMsg(*msgIn, laserCloudHelios);

        // Sort the pointcloud by stamp
        std::sort(laserCloudHelios.points.begin(), laserCloudHelios.points.end(),
                  [](const PointHelios& pa, const PointHelios& pb)
                  {
                    double ta = pa.sec + pa.usec/1.0e6;
                    double tb = pb.sec + pb.usec/1.0e6;
                    return ta < tb;
                  });

        double headerStamp = laserCloudHelios.points.front().sec + laserCloudHelios.points.front().usec/1.0e6;
        double timebase = laserCloudHelios.points.front().sec + laserCloudHelios.points.front().usec/1.0e6;

        int cloudsize = laserCloudHelios.size();
        CloudOuster laserCloudOuster;
        laserCloudOuster.points.resize(cloudsize);
        laserCloudOuster.is_dense = true;
        static double hsToOusterIntensity = 1.0;

        #pragma omp parallel for num_threads(NUM_CORE)
        for (size_t i = 0; i < cloudsize; i++)
        {
            auto &src = laserCloudHelios.points[i];
            auto &dst = laserCloudOuster.points[i];
            double srcTime = src.sec + src.usec/1.0e6;
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.intensity = src.intensity * hsToOusterIntensity;
            dst.ring = src.ring;
            dst.t = (srcTime - timebase) * 1e9f;
            // printf("pidx: %d. Time: %f, %d. Header: %f\n", i, srcTime, dst.t, msgIn->header.stamp.toSec());
            dst.range = sqrt(src.x * src.x + src.y * src.y + src.z * src.z)*1000;
        }

        Util::publishCloud(ousterCloudPub, laserCloudOuster, ros::Time(headerStamp), msgIn->header.frame_id);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "helios_to_ouster");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Helios to Ouster started" RESET);

    HeliosToOuster C2P(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}