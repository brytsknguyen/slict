/**
* This file is part of merge_lidar.
* 
* Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
* School of EEE
* Nanyang Technological Univertsity, Singapore
* 
* For more information please see <https://britsknguyen.github.io>.
* or <https://github.com/britsknguyen/merge_lidar>.
* If you use this code, please cite the respective publications as
* listed on the above websites.
* 
* merge_lidar is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* merge_lidar is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with merge_lidar.  If not, see <http://www.gnu.org/licenses/>.
*/

//
// Created by Thien-Minh Nguyen on 15/12/20.
//

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

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <deque>
#include <thread>
#include <condition_variable>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// #include "utility.h"

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define RESET "\033[0m"

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

typedef pcl::PointCloud<PointOuster> CloudOuster;
typedef pcl::PointCloud<PointOuster>::Ptr CloudOusterPtr;

struct CloudPacket
{
    double startTime;
    double endTime;

    CloudOusterPtr cloud;
    
    CloudPacket(){};
    CloudPacket(double startTime_, double endTime_, CloudOusterPtr cloud_)
        : startTime(startTime_), endTime(endTime_), cloud(cloud_)
    {}
};

using namespace std;
using namespace Eigen;

class MergeLidar
{

private:
        
    // Node handler
    ros::NodeHandlePtr nh_ptr;

    // Subscribers
    vector<ros::Subscriber> lidar_sub;

    mutex lidar_buf_mtx;
    mutex lidar_leftover_buf_mtx;

    deque<deque<CloudPacket>> lidar_buf;
    deque<deque<CloudPacket>> lidar_leftover_buf;

    ros::Publisher merged_pc_pub;

    // Lidar extrinsics
    deque<Matrix3d> R_B_L;
    deque<Vector3d> t_B_L;

    vector<int> lidar_channels;
    deque<int> lidar_ring_offset;
    bool lidar_ring_offset_set = false;
    mutex channel_mutex;

    int MAX_THREAD = std::thread::hardware_concurrency();
    int Nlidar;

    double cutoff_time = -1;
    double cutoff_time_new = -1;

    thread sync_lidar;
    
public:
    // Destructor
    ~MergeLidar() {}

    MergeLidar(ros::NodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        // Initialize the variables and subsribe/advertise topics here
        Initialize();
    }

    void Initialize()
    {

        /* #region Lidar --------------------------------------------------------------------------------------------*/
        
        // Read the lidar topic
        vector<string> lidar_topic = {"/os_cloud_node/points"};
        nh_ptr->getParam("lidar_topic", lidar_topic);

        Nlidar = lidar_topic.size();

        // Read the extrincs of lidars
        vector<double> lidar_extr = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        nh_ptr->getParam("lidar_extr", lidar_extr);

        ROS_ASSERT_MSG( (lidar_extr.size() / 16) == Nlidar,
                        "Lidar extrinsics not complete: %d < %d (= %d*16)\n",
                         lidar_extr.size(), Nlidar, Nlidar*16);

        printf("Received %d lidar(s) with extrinsics: \n", Nlidar);
        for(int i = 0; i < Nlidar; i++)
        {
            // Confirm the topics
            printf("Lidar topic #%02d: %s\n", i, lidar_topic[i].c_str());

            Matrix4d extrinsicTf = Matrix<double, 4, 4, RowMajor>(&lidar_extr[i*16]);
            cout << "extrinsicTf: " << endl;
            cout << extrinsicTf << endl;

            R_B_L.push_back(extrinsicTf.block<3, 3>(0, 0));
            t_B_L.push_back(extrinsicTf.block<3, 1>(0, 3));

            lidar_buf.push_back(deque<CloudPacket>(0));
            lidar_leftover_buf.push_back(deque<CloudPacket>(0));

            // Subscribe to the lidar topic
            lidar_sub.push_back(nh_ptr->subscribe<sensor_msgs::PointCloud2>
                                            (lidar_topic[i], 100,
                                             boost::bind(&MergeLidar::PcHandler, this,
                                                         _1, i, (int)extrinsicTf(3, 3),
                                                         extrinsicTf(3, 2))));
        }

        // nh_ptr->getParam("lidar_channels", lidar_channels);
        // if (lidar_channels.size() != Nlidar)
        // {
        //     printf(KRED "Lidar channel params missing: %d params vs %d Lidar" RESET,
        //             lidar_channels.size(), Nlidar);
        //     exit(-1);
        // }
        // Create the ring offsets
        lidar_channels = vector<int>(Nlidar, -1);
        lidar_ring_offset = deque<int>(Nlidar, 0);
        // for(int i = 1; i < Nlidar; i++)
        //     lidar_ring_offset[i] = lidar_ring_offset[i-1] + lidar_channels[i-1];

        merged_pc_pub = nh_ptr->advertise<sensor_msgs::PointCloud2>("/os_cloud_node_merged", 100);

        /* #endregion Lidar -----------------------------------------------------------------------------------------*/

        // Create the synchronizing threads
        sync_lidar = thread(&MergeLidar::SyncLidar, this);

    }

    void PcHandler(const sensor_msgs::PointCloud2::ConstPtr &msg, int idx, int stamp_type, double time_offset)
    {
        double startTime = -1, endTime = -1;
        if (stamp_type == 1)
        {
            startTime = msg->header.stamp.toSec() + time_offset;
            endTime = startTime + 0.1;
        }
        else
        {
            endTime = msg->header.stamp.toSec() + time_offset;
            startTime = endTime - 0.1;
        }

        // Convert the cloud msg to pcl
        CloudOusterPtr cloud_inL(new CloudOuster());
        pcl::fromROSMsg(*msg, *cloud_inL);
        
        // Check for the ring number so we can find the offset
        if(!lidar_ring_offset_set)
            checkLidarChannel(cloud_inL, idx);
        if(!lidar_ring_offset_set)
            return;

        int pointsTotal = cloud_inL->size();
        // Create the body-referenced cloud
        CloudOusterPtr cloud_inB(new CloudOuster()); cloud_inB->resize(pointsTotal);

        // Transform and copy the points
        #pragma omp parallel for num_threads(MAX_THREAD)
        for(int i = 0; i < pointsTotal; i++)
        {
            PointOuster &point_inL = cloud_inL->points[i];

            Vector3d p_inL(point_inL.x, point_inL.y, point_inL.z);
            p_inL = R_B_L[idx]*p_inL + t_B_L[idx];

            // Copy the point
            PointOuster point_inB = point_inL;
            
            // Replace the point coordinates
            point_inB.x = p_inL(0);
            point_inB.y = p_inL(1);
            point_inB.z = p_inL(2);

            // Renumber the ring
            point_inB.ring += lidar_ring_offset[idx];

            // Push the data into buffer
            cloud_inB->points[i] = point_inB;
        }

        lidar_buf_mtx.lock();
        lidar_buf[idx].push_back(CloudPacket(startTime, endTime, cloud_inB));
        lidar_buf_mtx.unlock();
    }

    bool checkLidarChannel(CloudOusterPtr &cloud_inL, int idx)
    {
        std::lock_guard<mutex> lg(channel_mutex);

        int pointsTotal = cloud_inL->size();

        if(lidar_channels[idx] == -1)
        {
            for(int i = 0; i < pointsTotal; i++)
                lidar_channels[idx] = max(lidar_channels[idx], cloud_inL->points[i].ring + 1);
            printf("Lidar %d is found to have %d channels:\n", idx, lidar_channels[idx]);
        }

        // Exits the callback if any lidar channel has not been checked
        for(int lidx = 0; lidx < lidar_channels.size(); lidx++)
            if(lidar_channels[idx] == -1)
                return false;

        // All lidar has been checked. Calculate the channels offset
        printf("All lidar channels checked.\n");
        for(int lidx = 1; lidx < lidar_channels.size(); lidx++)
        {
            lidar_ring_offset[lidx] = lidar_ring_offset[lidx-1] + lidar_channels[lidx-1];
            printf("Lidar %d ring offset: %d\n", lidx, lidar_ring_offset[lidx]);
        }

        lidar_ring_offset_set = false;

        return true;
    }

    void SyncLidar()
    {
        while(ros::ok())
        {
            // Loop if the secondary buffers don't over lap
            if(!LidarBufReady())
            {
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }

            if (lidar_buf.size() > 1)
            {
                printf("Buf 0: Start: %.3f. End: %.3f / %.3f. Points: %d / %d. Size: %d\n"
                       "Buf 1: Start: %.3f. End: %.3f / %.3f. Points: %d / %d. Size: %d\n",
                        lidar_buf[0].front().startTime,
                        lidar_buf[0].front().endTime, lidar_buf[0].back().endTime,
                        lidar_buf[0].front().cloud->size(), lidar_buf[0].back().cloud->size(), lidar_buf[0].size(),
                        lidar_buf[1].front().startTime,
                        lidar_buf[1].front().endTime, lidar_buf[1].back().endTime,
                        lidar_buf[1].front().cloud->size(), lidar_buf[1].back().cloud->size(), lidar_buf[1].size());
            }

            // Extract the points
            CloudPacket extracted_points;
            ExtractLidarPoints(extracted_points);

            printf("Extracted: Start: %.3f. End: %.3f. Points: %d. CoT: %.3f. New CoT: %.3f\n",
                    extracted_points.startTime, extracted_points.endTime, extracted_points.cloud->size(),
                    cutoff_time, cutoff_time_new);
            if(!lidar_buf[0].empty())
            {
                printf("Buf 0: Start: %.3f. End: %.3f / %.3f. Points: %d / %d. Size: %d\n",
                       lidar_buf[0].front().startTime,
                       lidar_buf[0].front().endTime, lidar_buf[0].back().endTime,
                       lidar_buf[0].front().cloud->size(), lidar_buf[0].back().cloud->size(), lidar_buf[0].size());    
            }

            if (lidar_buf.size() > 1)
            {
                if(!lidar_buf[1].empty())
                {
                    printf("Buf 1: Start: %.3f. End: %.3f / %.3f. Points: %d / %d. Size: %d\n",
                            lidar_buf[1].front().startTime,
                            lidar_buf[1].front().endTime, lidar_buf[1].back().endTime,
                            lidar_buf[1].front().cloud->size(), lidar_buf[1].back().cloud->size(), lidar_buf[1].size());
                }
            }
            cout << endl;

            // Update the cutoff time
            cutoff_time = cutoff_time_new;

            // Publish the merged pointcloud
            if(extracted_points.cloud->size() != 0)
                publishCloud(merged_pc_pub, *extracted_points.cloud, ros::Time(extracted_points.startTime), string("body"));
        }
    }

    bool LidarBufReady()
    {
        // If any buffer is empty, lidar is not ready
        for(int i = 0; i < Nlidar; i++)
        {
            if (lidar_buf[i].empty())
            {
                return false;
            }
        }
        
        // If any secondary buffer's end still has not passed the first endtime in the primary buffer, loop
        for(int i = 1; i < Nlidar; i++)
        {
            if (lidar_buf[i].back().endTime < lidar_buf[0].front().endTime)
            {
                return false;
            }
        }

        return true;    
    }

    void ExtractLidarPoints(CloudPacket &extracted_points)
    {
        // Initiate the cutoff time.
        if (cutoff_time == -1)
            cutoff_time = lidar_buf[0].front().startTime;

        cutoff_time_new = lidar_buf[0].front().endTime;

        // Go through each buffer and extract the valid points
        deque<CloudOusterPtr> extracted_clouds(Nlidar);
        for(int i = 0; i < Nlidar; i++)
        {
            extracted_clouds[i] = CloudOusterPtr(new CloudOuster());

            CloudPacket leftover_cloud;
            leftover_cloud.startTime = cutoff_time_new;
            leftover_cloud.endTime = -1;
            leftover_cloud.cloud = CloudOusterPtr(new CloudOuster());

            while(!lidar_buf[i].empty())
            {
                CloudPacket &front_cloud = lidar_buf[i].front();

                if ( front_cloud.endTime < cutoff_time )
                {
                    lock_guard<mutex> lock(lidar_buf_mtx);
                    lidar_buf[i].pop_front();
                    continue;
                }
                else if( front_cloud.startTime > cutoff_time_new )
                    break;

                // Copy the points into the extracted pointcloud or the leftover
                for( auto &point : front_cloud.cloud->points )
                {
                    double point_time = point.t / 1.0e9 + front_cloud.startTime;

                    if (point_time >= cutoff_time && point_time <= cutoff_time_new)
                    {
                        extracted_clouds[i]->push_back(point);
                        extracted_clouds[i]->points.back().t = (uint32_t)((point_time - cutoff_time)*1.0e9);
                    }
                    else if (point_time > cutoff_time_new)
                    {
                        leftover_cloud.cloud->push_back(point);
                        leftover_cloud.cloud->points.back().t = (uint32_t)((point_time - cutoff_time_new)*1.0e9);

                        if (point_time > leftover_cloud.endTime)
                            leftover_cloud.endTime = point_time;
                    }
                }

                {
                    lock_guard<mutex> lock(lidar_buf_mtx);
                    lidar_buf[i].pop_front();
                }

                // Check the leftover buffer and insert extra points
                while(lidar_leftover_buf[i].size() != 0)
                {
                    if (lidar_leftover_buf[i].front().endTime < cutoff_time)
                    {
                        lidar_leftover_buf[i].pop_front();
                        continue;
                    }

                    if (lidar_leftover_buf[i].front().startTime > cutoff_time_new)
                        continue;

                    // Extract the first packet
                    CloudPacket leftover_frontcloud = lidar_leftover_buf[i].front();
                    lidar_leftover_buf[i].pop_front();

                    // Insert the leftover points back in the buffer
                    for( auto &point : leftover_frontcloud.cloud->points )
                    {
                        double point_time = point.t / 1.0e9 + leftover_frontcloud.startTime;

                        if (point_time >= cutoff_time && point_time <= cutoff_time_new)
                        {
                            extracted_clouds[i]->push_back(point);
                            extracted_clouds[i]->points.back().t = (uint32_t)((point_time - cutoff_time)*1.0e9);
                        }
                        else if (point_time > cutoff_time_new)
                        {
                            leftover_cloud.cloud->push_back(point);
                            leftover_cloud.cloud->points.back().t = (uint32_t)((point_time - cutoff_time_new)*1.0e9);

                            if (point_time > leftover_cloud.endTime)
                                leftover_cloud.endTime = point_time;
                        }
                    }
                }

                if (i == 0)
                    break;
            }

            if (leftover_cloud.cloud->size() > 0)
                lidar_leftover_buf[i].push_back(leftover_cloud);
        }

        // Merge the extracted clouds
        extracted_points.startTime = cutoff_time;
        extracted_points.endTime = cutoff_time_new;
        extracted_points.cloud = CloudOusterPtr(new CloudOuster());
        for(int i = 0; i < Nlidar; i++)
            *extracted_points.cloud += *extracted_clouds[i];
    }

    void publishCloud(ros::Publisher &pub, CloudOuster &cloud, ros::Time thisStamp, std::string thisFrame)
    {
        sensor_msgs::PointCloud2 cloud_;
        pcl::toROSMsg(cloud, cloud_);
        cloud_.header.stamp = thisStamp;
        cloud_.header.frame_id = thisFrame;
        pub.publish(cloud_);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "merge_lidar");
    ros::NodeHandle nh("~");
    ros::NodeHandlePtr nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Merge Lidar Started." RESET);

    MergeLidar sensor_sync(nh_ptr);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();
    
    return 0;
}
