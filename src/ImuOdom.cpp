/**
 * This file is part of slict.
 *
 * Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot
 * sg>, School of EEE Nanyang Technological Univertsity, Singapore
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

typedef sensor_msgs::Imu rosImuMsg;
typedef sensor_msgs::Imu::ConstPtr rosImuMsgPtr;
typedef nav_msgs::Odometry rosOdomMsg;
typedef nav_msgs::Odometry::ConstPtr rosOdomMsgPtr;

ros::NodeHandlePtr nh_ptr;

// Constant gravity vector
Vector3d GRAV(0, 0, 9.82);

mutex imu_buf_mtx;
std::shared_ptr<ImuProp> imu_prop = nullptr;
rosOdomMsgPtr opt_odom;

Matrix4d T_B_V;
Matrix3d R_B_V;
Vector3d t_B_V;

rosOdomMsg pred_odom;

void imuCB(const rosImuMsgPtr &msg)
{
    rosOdomMsg opt_odom_last;

    {
        lock_guard<mutex> lg(imu_buf_mtx);

        if(imu_prop == nullptr)
            return;

        if(imu_prop->t.back() == 0)
            return;

        if (msg->header.stamp.toSec() <= imu_prop->t.back())
            return;

        imu_prop->forwardPropagate(msg);

        opt_odom_last = *opt_odom;
    }

    // Update the IMU odom message    
    
    // Publish the tf for easy tracking
    // static double last_tf_pub_time = -1;
    
    // Transform the pose to the vehicle frame
    static mytf tf_B_V(R_B_V, t_B_V);
    mytf tf_W_V;
    if(fabs(imu_prop->t.back() - opt_odom->header.stamp.toSec()) < 1.0 )
        tf_W_V = imu_prop->getBackTf()*tf_B_V;
    else
        tf_W_V = myTf(opt_odom_last)*tf_B_V;
        
    pred_odom.header.frame_id = opt_odom->header.frame_id;
    pred_odom.header.stamp    = ros::Time(imu_prop->t.back());
    pred_odom.child_frame_id  = opt_odom_last.child_frame_id;

    pred_odom.pose.pose.position.x = tf_W_V.pos.x();
    pred_odom.pose.pose.position.y = tf_W_V.pos.y();
    pred_odom.pose.pose.position.z = tf_W_V.pos.z();

    pred_odom.pose.pose.orientation.x = tf_W_V.rot.x();
    pred_odom.pose.pose.orientation.y = tf_W_V.rot.y();
    pred_odom.pose.pose.orientation.z = tf_W_V.rot.z();
    pred_odom.pose.pose.orientation.w = tf_W_V.rot.w();

    pred_odom.twist.twist.linear.x = imu_prop->V.back().x();
    pred_odom.twist.twist.linear.y = imu_prop->V.back().y();
    pred_odom.twist.twist.linear.z = imu_prop->V.back().z();

    // Advertise the imu predicted odom and publish it
    static ros::Publisher pred_odom_pub = nh_ptr->advertise<nav_msgs::Odometry>("/pred_odom_W_V", 10);
    pred_odom_pub.publish(pred_odom);

    static tf::TransformBroadcaster tfbr;
    tf::Transform transform;
    transform.setOrigin(tf::Vector3(tf_W_V.pos(0), tf_W_V.pos(1), tf_W_V.pos(2)));
    transform.setRotation(tf::Quaternion(tf_W_V.rot.w(), tf_W_V.rot.x(), tf_W_V.rot.y(), tf_W_V.rot.z()));
    tfbr.sendTransform(tf::StampedTransform(transform, ros::Time::now(), pred_odom.header.frame_id, pred_odom.child_frame_id));
}

void odomCB(const rosOdomMsgPtr &msg)
{
    lock_guard<mutex> lg(imu_buf_mtx);

    opt_odom = msg;

    // // Repropagate the pose
    double t_odom = msg->header.stamp.toSec();
    Quaternd orientation(
        msg->pose.pose.orientation.w,
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z);
    Vector3d position(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z);
    Vector3d velocity(
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z);
    Vector3d angular_velocity(
        msg->twist.twist.angular.x,
        msg->twist.twist.angular.y,
        msg->twist.twist.angular.z);
    Vector3d linear_acceleration(
        msg->twist.covariance[0],
        msg->twist.covariance[1],
        msg->twist.covariance[2]);
    Vector3d bias_g(
        msg->twist.covariance[3],
        msg->twist.covariance[4],
        msg->twist.covariance[5]);
    Vector3d bias_a(
        msg->twist.covariance[6],
        msg->twist.covariance[7],
        msg->twist.covariance[8]);

    std::shared_ptr<ImuProp> imu_prop_(new ImuProp(orientation, position, velocity, bias_g, bias_a,
                                                   angular_velocity, linear_acceleration, GRAV, t_odom));

    if(imu_prop != nullptr)
    {
        for(int i = 0; i < imu_prop->size(); i++)
            if(imu_prop->t[i] >= t_odom)
                imu_prop_->forwardPropagate(imu_prop->gyr[i], imu_prop->acc[i], imu_prop->t[i]);
    }

    imu_prop = imu_prop_;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ImuOdom");
    ros::NodeHandle nh("~");
    nh_ptr = boost::make_shared<ros::NodeHandle>(nh);

    ROS_INFO(KGRN "----> Imu Odometry Started." RESET);

    double GRAV_ = 9.82;
    nh_ptr->param("/GRAV", GRAV_, 9.82);
    GRAV = Vector3d(0, 0, GRAV_);

    vector<double> T_B_V_ = {1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1};
    nh_ptr->getParam("/T_B_V", T_B_V_);
    T_B_V = Matrix<double, 4, 4, RowMajor>(&T_B_V_[0]);
    R_B_V = T_B_V.block<3,3>(0, 0);
    t_B_V = T_B_V.block<3,1>(0, 3);

    printf("Received T_B_V: \n");
    cout << T_B_V << endl;

    // Create the common message
    // pred_odom.child_frame_id  = "";

    // Subscribe to the IMU topic
    string imu_topic = nh_ptr->param("/imu_topic", imu_topic);
    ros::Subscriber imu_sub = nh_ptr->subscribe(imu_topic, 10, imuCB);

    // Subscribe to the odom topic
    ros::Subscriber odom_sub = nh_ptr->subscribe("/opt_odom", 10, odomCB);

    ros::MultiThreadedSpinner spinner(0);
    spinner.spin();

    return 0;
}
