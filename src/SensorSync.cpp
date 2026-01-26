/**
* This file is part of slict.
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

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <deque>
#include <thread>
#include <condition_variable>

#include <Eigen/Dense>
// #include <cv_bridge/cv_bridge.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
// #include "tf/transform_broadcaster.h"
// #include "tf2_ros/static_transform_broadcaster.h"
#include "image_transport/image_transport.hpp"
#include "slict/msg/feature_cloud.hpp"

// Package specials
// #include "preprocess.hpp"
#include "utility.h"

typedef slict::msg::FeatureCloud slictFCMsg;

struct CloudPacket
{
    double startTime;
    double endTime;

    CloudXYZITPtr cloud;

    CloudPacket(){};
    CloudPacket(double startTime_, double endTime_, CloudXYZITPtr cloud_)
        : startTime(startTime_), endTime(endTime_), cloud(cloud_)
    {}
};

class SensorSync
{
private:

    // Node handler
    RosNodeHandlePtr nh_ptr;

    // Subscribers
    vector<rclcpp::Subscription<RosPc2Msg>::SharedPtr> lidar_sub;
    rclcpp::Subscription<RosImuMsg>::SharedPtr imu_sub;

    mutex lidar_buf_mtx;
    mutex lidar_leftover_buf_mtx;
    mutex imu_buf_mtx;

    deque<deque<CloudPacket>> lidar_buf;
    deque<deque<CloudPacket>> lidar_leftover_buf;
    deque<RosImuMsg::ConstSharedPtr> imu_buf;

    mutex merged_cloud_buf_mtx;
    deque<CloudPacket> merged_cloud_buf;

    rclcpp::Publisher<RosPc2Msg>::SharedPtr merged_pc_pub;
    rclcpp::Publisher<slictFCMsg>::SharedPtr data_pub;

    // Lidar extrinsics
    deque<Matrix3d> R_B_L;
    deque<Vector3d> t_B_L;

    // IMU extrinsics
    Matrix3d R_B_I;
    Vector3d t_B_I;

    int MAX_THREAD = std::thread::hardware_concurrency();
    int Nlidar;
    int Nimu;

    double cutoff_time = -1;
    double cutoff_time_new = -1;
    double min_range = 0.5;
    vector<int64_t> ds_rate = {1};
    int sweep_len = 1;

    thread sync_lidar;
    thread sync_data;

    double acce_scale = 1.0;

public:
    // Destructor
    ~SensorSync() {}

    SensorSync(RosNodeHandlePtr &nh_ptr_) : nh_ptr(nh_ptr_)
    {
        // Initialize the variables and subsribe/advertise topics here
        Initialize();
    }

    void Initialize()
    {

        /* #region Lidar --------------------------------------------------------------------------------------------*/

        // Read the lidar topic
        vector<string> lidar_topic = {"/os_cloud_node/points"};
        Util::GetParam(nh_ptr, "lidar_topic", lidar_topic);

        Nlidar = lidar_topic.size();

        // Read the extrincs of lidars
        vector<double> lidar_extr = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        Util::GetParam(nh_ptr, "lidar_extr", lidar_extr);

        ROS_ASSERT_MSG( (lidar_extr.size() / 16) == Nlidar,
                        "Lidar extrinsics not complete: %d < %d (= %d*16)\n",
                         lidar_extr.size(), Nlidar, Nlidar*16);

        printf("Received %d lidar(s) with extrinsics: \n", Nlidar);
        for(int lidx = 0; lidx < Nlidar; lidx++)
        {
            // Confirm the topics
            printf("Lidar topic #%02d: %s\n", lidx, lidar_topic[lidx].c_str());

            Matrix4d extrinsicTf = Matrix<double, 4, 4, RowMajor>(&lidar_extr[lidx*16]);
            cout << "extrinsicTf: " << endl;
            cout << extrinsicTf << endl;

            R_B_L.push_back(extrinsicTf.block<3, 3>(0, 0));
            t_B_L.push_back(extrinsicTf.block<3, 1>(0, 3));

            lidar_buf.push_back(deque<CloudPacket>(0));
            lidar_leftover_buf.push_back(deque<CloudPacket>(0));

            const double time_offset = extrinsicTf(3, 2);
            const int stamp_type = static_cast<int>(extrinsicTf(3, 3));

            // Subscribe to the lidar topic
            lidar_sub.push_back(
                nh_ptr->create_subscription<RosPc2Msg>(
                    lidar_topic[lidx],
                    rclcpp::QoS(100),
                    [this, lidx, time_offset, stamp_type](RosPc2Msg::ConstSharedPtr msg)
                    {
                        this->PcHandler(msg, lidx, time_offset, stamp_type);
                    }
                )
            );
        }

        Util::GetParam(nh_ptr, "min_range", min_range);
        printf("Lidar minimum range: %f\n", min_range);

        Util::GetParam(nh_ptr, "ds_rate", ds_rate);
        printf("Downsamping rate: ");
        for(auto rate : ds_rate)
            printf("%d ", rate);
        cout << endl;

        Util::GetParam(nh_ptr, "sweep_len", sweep_len);
        printf("Sweep len: %d\n", sweep_len);

        // Advertise lidar topic
        string merged_lidar_topic;
        Util::GetParam(nh_ptr, "merged_lidar_topic", merged_lidar_topic);
        merged_pc_pub = nh_ptr->create_publisher<RosPc2Msg>(merged_lidar_topic, 100);

        /* #endregion Lidar -----------------------------------------------------------------------------------------*/


        /* #region IMU ----------------------------------------------------------------------------------------------*/

        // Read the IMU topic
        string imu_topic = "/imu_vn_100/imu";
        Util::GetParam(nh_ptr, "imu_topic", imu_topic);

        // Get the scale factor
        Util::GetParam(nh_ptr, "acce_scale", acce_scale);

        // Read the extrincs of lidars
        vector<double> imu_extr = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        Util::GetParam(nh_ptr, "imu_extr", imu_extr);

        // Confirm the topic(s)
        printf("IMU topic: %s\n", imu_topic.c_str());

        Matrix4d extrinsicTf = Matrix<double, 4, 4, RowMajor>(&imu_extr[0]);
        cout << "extrinsicTf: " << endl;
        cout << extrinsicTf << endl;

        R_B_I = extrinsicTf.block<3, 3>(0, 0);
        t_B_I = extrinsicTf.block<3, 1>(0, 3);

        // Subscribe to the IMU topic
        imu_sub = nh_ptr->create_subscription<RosImuMsg>(imu_topic, 10000, std::bind(&SensorSync::ImuHandler, this, std::placeholders::_1));

        /* #endregion IMU -------------------------------------------------------------------------------------------*/

        data_pub = nh_ptr->create_publisher<slictFCMsg>("/sensors_sync", 100);

        // Create the synchronizing threads
        sync_lidar = thread(&SensorSync::SyncLidar, this);
        sync_data  = thread(&SensorSync::SyncData, this);
    }

    void PcHandler(const RosPc2Msg::ConstSharedPtr &msg, int idx, double time_offset, int stamp_type)
    {
        if (idx == 0)
        {
            // Lump the pointclouds together
            typedef RosPc2Msg::ConstPtr RosCloudPtr;
            static deque<RosCloudPtr> cloudBuf;
            cloudBuf.push_back(msg);

            if (cloudBuf.size() >= sweep_len)
            {
                double startTime = -1, endTime = -1;
                CloudXYZITPtr cloud_inB(new CloudXYZIT());

                // Extract the point, cloud by cloud
                for (RosCloudPtr &cloudMsg : cloudBuf)
                {
                    CloudOusterPtr cloud_inL(new CloudOuster());
                    pcl::fromROSMsg(*cloudMsg, *cloud_inL);

                    double cloud_start_time = -1, cloud_end_time = -1;
                    double sweep_dur = (cloud_inL->points.back().t - cloud_inL->points.front().t)/1.0e9;
                    double sweep_dur_err = fabs(sweep_dur - 0.1);
                    // ROS_ASSERT_MSG(sweep_dur_err < 5e-3, "Sweep length %f not exactly 0.1s.\n", sweep_dur, sweep_dur_err);
                    // Calculate the proper time stamps at the two ends

                    if (stamp_type == 1)
                    {
                        cloud_start_time = Util::toDouble(cloudMsg->header.stamp) + time_offset;
                        cloud_end_time   = cloud_start_time + sweep_dur;
                    }
                    else
                    {
                        cloud_end_time   = Util::toDouble(cloudMsg->header.stamp) + time_offset;
                        cloud_start_time = cloud_end_time - sweep_dur;
                    }

                    startTime = (startTime < 0) ? cloud_start_time : min(startTime, cloud_start_time);
                    endTime = (endTime < 0) ? cloud_end_time : max(endTime, cloud_end_time);

                    // Check the min dist and restamp the points
                    for(int i = 0; i < cloud_inL->size(); i++)
                    {
                        PointOuster &point_inL = cloud_inL->points[i];

                        // Discard of the point if range is zero
                        if (point_inL.range/1000.0 < min_range)
                            continue;

                        Vector3d p_inL(point_inL.x, point_inL.y, point_inL.z);
                        p_inL = R_B_L[idx]*p_inL + t_B_L[idx];

                        PointXYZIT point_inB;
                        point_inB.x = p_inL(0);
                        point_inB.y = p_inL(1);
                        point_inB.z = p_inL(2);
                        point_inB.intensity = point_inL.intensity;
                        point_inB.t = point_inL.t / 1.0e9 + cloud_start_time;

                        cloud_inB->push_back(point_inB);
                    }
                }

                lidar_buf_mtx.lock();
                lidar_buf[idx].push_back(CloudPacket(startTime, endTime, cloud_inB));
                lidar_buf_mtx.unlock();

                cloudBuf.clear();
            }
        }
        else if (idx != 0)
        {
            double startTime, endTime;
            if (stamp_type == 1)
            {
                startTime = Util::toDouble(msg->header.stamp) + time_offset;
                endTime   = startTime + 0.1;
            }
            else
            {
                endTime   = Util::toDouble(msg->header.stamp) + time_offset;
                startTime = endTime - 0.1;
            }

            CloudOusterPtr cloud_inL(new CloudOuster());
            pcl::fromROSMsg(*msg, *cloud_inL);

            CloudXYZITPtr cloud_inB(new CloudXYZIT());

            // Check the min dist and restamp the points
            for(int i = 0; i < cloud_inL->size(); i++)
            {
                PointOuster &point_inL = cloud_inL->points[i];

                // Discard of the point if range is zero
                if (point_inL.range/1000.0 < min_range)
                    continue;

                Vector3d p_inL(point_inL.x, point_inL.y, point_inL.z);
                p_inL = R_B_L[idx]*p_inL + t_B_L[idx];

                PointXYZIT point_inB;
                point_inB.x = p_inL(0);
                point_inB.y = p_inL(1);
                point_inB.z = p_inL(2);
                point_inB.intensity = point_inL.intensity;
                point_inB.t = point_inL.t / 1e9 + startTime;

                cloud_inB->push_back(point_inB);
            }

            lidar_buf_mtx.lock();
            lidar_buf[idx].push_back(CloudPacket(startTime, endTime, cloud_inB));
            lidar_buf_mtx.unlock();
        }
        else
            return;
    }

    void ImuHandler(const RosImuMsg::ConstSharedPtr &msg)
    {
        imu_buf_mtx.lock();

        if (acce_scale != 1.0)
        {
            RosImuMsg::Ptr scaled_imu(new RosImuMsg());

            *scaled_imu = *msg;
            scaled_imu->linear_acceleration.x *= acce_scale;
            scaled_imu->linear_acceleration.y *= acce_scale;
            scaled_imu->linear_acceleration.z *= acce_scale;

            imu_buf.push_back(scaled_imu);
        }
        else
            imu_buf.push_back(msg);

        imu_buf_mtx.unlock();
    }

    void SyncLidar()
    {
        while(rclcpp::ok())
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

            // Store the merged pointcloud
            if(extracted_points.cloud->size() != 0)
            {
                lock_guard<mutex> lock(merged_cloud_buf_mtx);
                merged_cloud_buf.push_back(extracted_points);
            }
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

        // If any secondary buffer's end still has not passed the first endtime in the primary buffer, loop lah!
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
        deque<CloudXYZITPtr> extracted_clouds(Nlidar);
        for(int i = 0; i < Nlidar; i++)
        {
            extracted_clouds[i] = CloudXYZITPtr(new CloudXYZIT());

            CloudPacket leftover_cloud;
            leftover_cloud.startTime = cutoff_time_new;
            leftover_cloud.endTime = -1;
            leftover_cloud.cloud = CloudXYZITPtr(new CloudXYZIT());

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

                int one_in_n = ds_rate[i];
                static int ds_count = -1;
// yolos("lidx: %d, div: %d, points: %d. Time: %f, %f", i, one_in_n, front_cloud.cloud->points.size(), front_cloud.cloud->points.front().t, front_cloud.cloud->points.back().t);
                // Copy the points into the extracted pointcloud or the leftover
                for( auto &point : front_cloud.cloud->points )
                {
                    ds_count++; if (ds_count % one_in_n != 0) continue; // Skip one every

                    double point_time = point.t;
// yolos("lidx: %d, div: %d, points: %d. Time: %f, %f. Point time: %f", i, one_in_n, front_cloud.cloud->points.size(), front_cloud.cloud->points.front().t, front_cloud.cloud->points.back().t, point_time);

                    if (point_time >= cutoff_time && point_time <= cutoff_time_new)
                    {
                        extracted_clouds[i]->push_back(point);
                        // extracted_clouds[i]->points.back().t = point_time - cutoff_time;
                    }
                    else if (point_time > cutoff_time_new)
                    {
                        leftover_cloud.cloud->push_back(point);
                        // leftover_cloud.cloud->points.back().t = point_time - cutoff_time_new;

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
                        double point_time = point.t;

                        if (point_time >= cutoff_time && point_time <= cutoff_time_new)
                        {
                            extracted_clouds[i]->push_back(point);
                            // extracted_clouds[i]->points.back().t = point_time - cutoff_time;
                        }
                        else if (point_time > cutoff_time_new)
                        {
                            leftover_cloud.cloud->push_back(point);
                            // leftover_cloud.cloud->points.back().t = point_time - cutoff_time_new;

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
        extracted_points.cloud = CloudXYZITPtr(new CloudXYZIT());
        for(int i = 0; i < Nlidar; i++)
            *extracted_points.cloud += *extracted_clouds[i];
        }

    bool ImuEmpty()
    {
        return imu_buf.empty();
    }

    double ImuEndTime()
    {
        return Util::toDouble(imu_buf.back()->header.stamp);
    }

    double ImuStartTime()
    {
        return Util::toDouble(imu_buf.front()->header.stamp);
    }

    void SyncData()
    {
        while (true)
        {
            /* #region Probing the key buffers ----------------------------------------------------------------------*/

            // Case 0: If any buffer is empty, then loop until all have data
            if (merged_cloud_buf.empty() || ImuEmpty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Case 1: If latest imu data is still earlier than the earliest
            // feature message, then wait for these measurements to catch up:

            // |___________[mfc]___mfc___mfc____________
            // |_imu_[imu]______________________________
            // |----------------------------------------> t+
            if (merged_cloud_buf.front().endTime > ImuEndTime())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // Case 2: If earliest imu data is later than the earliest mfc's
            // end time, then discard the earliest cloud message.
            // This should only occur at the beginning

            // |_[mfc]___mfc___mfc__mfc________________
            // |_____________________[imu]_imu_imu_____
            // |---------------------------------------> t+
            if (merged_cloud_buf.front().endTime < ImuStartTime())
            {
                printf(KYEL "Popping mfc. Start: %.3f, End: %.3f. Size: %d\n" RESET,
                             merged_cloud_buf.front().startTime,
                             merged_cloud_buf.front().endTime,
                             merged_cloud_buf.front().cloud->size());

                merged_cloud_buf_mtx.lock();
                merged_cloud_buf.pop_front();
                merged_cloud_buf_mtx.unlock();
                continue;
            }

            // Case 3: A regular scenario, extract all imu measurement in the scan period

            // |________mfc___mfc___mfc________________
            // |_imu_imu_imu_imu_______________________
            // |---------------------------------------> t+

            // Extract the cloud packet
            CloudPacket merged_cloud = merged_cloud_buf.front();
            merged_cloud_buf_mtx.lock();
            merged_cloud_buf.pop_front();
            merged_cloud_buf_mtx.unlock();

            slictFCMsg msg;
            msg.header.stamp    = Util::toRosTime(merged_cloud.startTime);
            msg.extracted_cloud = Util::publishCloud(merged_pc_pub, *merged_cloud.cloud, Util::toRosTime(merged_cloud.startTime), string("body"));
            msg.scan_start_time = merged_cloud.startTime;
            msg.scan_end_time   = merged_cloud.endTime;

            vector<RosImuMsg> &imu_bundle = msg.imu_msgs;
            double imu_start_time = -1, imu_end_time = -1;
            while(!imu_buf.empty())
            {
                RosImuMsg imu_sample = *imu_buf.front();
                // imu_sample.header.seq = 0;

                double imu_stamp = Util::toDouble(imu_sample.header.stamp);

                if ( imu_stamp <= merged_cloud.endTime )
                {
                    lock_guard<mutex> lock(imu_buf_mtx);
                    imu_buf.pop_front();
                }
                else
                {
                    // ASSUMPTION: Data from IMU i is available and has been stored
                    // ROS_ASSERT(!imu_bundle.empty());
                    // ROS_ASSERT_MSG(imu_bundle.back().header.seq == 0,
                    //                 "seq: %d. i: %d. Sz: %d. scan_end_time: %f. IMUTime: %f\n",
                    //                 imu_bundle.back().header.seq, 0, imu_buf.size(), merged_cloud.endTime, imu_stamp);

                    // Linearly interpolating the imu sample
                    double ta = Util::toDouble(imu_bundle.back().header.stamp);
                    double tb = Util::toDouble(imu_sample.header.stamp);

                    // ASSUMPTION: IMU is spaced out
                    assert( tb - ta > 0 );

                    Vector3d gyro_ta(imu_bundle.back().angular_velocity.x,
                                     imu_bundle.back().angular_velocity.y,
                                     imu_bundle.back().angular_velocity.z);
                    Vector3d gyro_tb(imu_sample.angular_velocity.x,
                                     imu_sample.angular_velocity.y,
                                     imu_sample.angular_velocity.z);
                    Vector3d acce_ta(imu_bundle.back().linear_acceleration.x,
                                     imu_bundle.back().linear_acceleration.y,
                                     imu_bundle.back().linear_acceleration.z);
                    Vector3d acce_tb(imu_sample.linear_acceleration.x,
                                     imu_sample.linear_acceleration.y,
                                     imu_sample.linear_acceleration.z);

                    // Make an interpolated sample
                    double t_itp = merged_cloud.endTime;
                    double s = (t_itp - ta)/(tb - ta);
                    Vector3d gyro_itp = (1-s)*gyro_ta + s*gyro_tb;
                    Vector3d acce_itp = (1-s)*acce_ta + s*acce_tb;

                    imu_stamp = t_itp;
                    imu_sample.header.stamp = Util::toRosTime(t_itp);
                    // imu_sample.header.seq = 0;
                    imu_sample.angular_velocity.x = gyro_itp(0);
                    imu_sample.angular_velocity.y = gyro_itp(1);
                    imu_sample.angular_velocity.z = gyro_itp(2);
                    imu_sample.linear_acceleration.x = acce_itp(0);
                    imu_sample.linear_acceleration.y = acce_itp(1);
                    imu_sample.linear_acceleration.z = acce_itp(2);
                }

                if (imu_stamp >= merged_cloud.startTime && imu_stamp <= merged_cloud.endTime)
                {
                    imu_bundle.push_back(imu_sample);

                    if (imu_start_time == -1 || imu_start_time > imu_stamp)
                        imu_start_time = imu_stamp;

                    if (imu_end_time == -1 || imu_end_time < imu_stamp)
                        imu_end_time = imu_stamp;
                }

                if (imu_stamp == merged_cloud.endTime)
                    break;
            }

            /* #endregion Probing the key buffers -------------------------------------------------------------------*/

            // Publish the synchronized data
            // if(imu_bundle.size() != 0)
            //     printf("imu_bundle: %f -> %f \n", Util::toDouble(imu_bundle.front().header.stamp), Util::toDouble(imu_bundle.back().header.stamp));
            data_pub->publish(msg);
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    RosNodeHandlePtr nh_ptr = rclcpp::Node::make_shared("sensor_sync");

    RINFO(KGRN "----> Sensor Sync Started." RESET);
    SensorSync SS(nh_ptr);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(nh_ptr);
    executor.spin();

    return 0;
}
