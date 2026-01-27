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
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>



RosNodeHandlePtr nh_ptr;

// Constant gravity vector
Vector3d GRAV(0, 0, 9.82);

mutex imu_buf_mtx;
std::shared_ptr<ImuProp> imu_prop = nullptr;
RosOdomMsg::ConstSharedPtr opt_odom;

Matrix4d T_B_V;
Matrix3d R_B_V;
Vector3d t_B_V;

void imuCB(const RosImuMsg::ConstSharedPtr &msg)
{
    RosOdomMsg opt_odom_last;

    {
        lock_guard<mutex> lg(imu_buf_mtx);

        if(imu_prop == nullptr)
            return;

        if(imu_prop->t.back() == 0)
            return;

        if (Util::toDouble(msg->header.stamp) <= imu_prop->t.back())
            return;

        imu_prop->forwardPropagate(msg);

        opt_odom_last = *opt_odom;
    }

    // Transform the pose to the vehicle frame
    static mytf tf_B_V(R_B_V, t_B_V);
    mytf tf_W_V;
    if(fabs(imu_prop->t.back() - Util::toDouble(opt_odom->header.stamp) < 1.0))
        tf_W_V = imu_prop->getBackTf()*tf_B_V;
    else
        tf_W_V = myTf(opt_odom_last)*tf_B_V;

    // Advertise the imu predicted odom and publish it
    static auto pred_odom_pub = nh_ptr->create_publisher<RosOdomMsg>("pred_odom_W_V", 10);
    // Predict odom
    RosOdomMsg pred_odom;
    pred_odom.header.frame_id = opt_odom->header.frame_id;
    pred_odom.header.stamp    = Util::toRosTime(imu_prop->t.back());
    pred_odom.child_frame_id  = opt_odom_last.child_frame_id;
    // trans
    pred_odom.pose.pose.position.x = tf_W_V.pos.x();
    pred_odom.pose.pose.position.y = tf_W_V.pos.y();
    pred_odom.pose.pose.position.z = tf_W_V.pos.z();
    // rot
    pred_odom.pose.pose.orientation.x = tf_W_V.rot.x();
    pred_odom.pose.pose.orientation.y = tf_W_V.rot.y();
    pred_odom.pose.pose.orientation.z = tf_W_V.rot.z();
    pred_odom.pose.pose.orientation.w = tf_W_V.rot.w();
    // twist
    pred_odom.twist.twist.linear.x = imu_prop->V.back().x();
    pred_odom.twist.twist.linear.y = imu_prop->V.back().y();
    pred_odom.twist.twist.linear.z = imu_prop->V.back().z();
    // Publish
    pred_odom_pub->publish(pred_odom);

    // Create the tf broadcaster
    static std::shared_ptr<tf2_ros::TransformBroadcaster> tfbr = std::make_shared<tf2_ros::TransformBroadcaster>(nh_ptr);;
    // Publish the tf data
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp    = nh_ptr->now();                // rclcpp::Time
    tf.header.frame_id = pred_odom.header.frame_id;    // parent
    tf.child_frame_id  = pred_odom.child_frame_id;     // child
    // trans
    tf.transform.translation.x = tf_W_V.pos(0);
    tf.transform.translation.y = tf_W_V.pos(1);
    tf.transform.translation.z = tf_W_V.pos(2);
    // rot
    tf.transform.rotation.w = tf_W_V.rot.w();
    tf.transform.rotation.x = tf_W_V.rot.x();
    tf.transform.rotation.y = tf_W_V.rot.y();
    tf.transform.rotation.z = tf_W_V.rot.z();
    // broadcast
    tfbr->sendTransform(tf);
}

void odomCB(const RosOdomMsg::ConstSharedPtr &msg)
{
    lock_guard<mutex> lg(imu_buf_mtx);

    opt_odom = msg;

    // // Repropagate the pose
    double t_odom = Util::toDouble(msg->header.stamp);
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
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    nh_ptr = rclcpp::Node::make_shared("imuodom", options);

    RINFO(KGRN "----> Imu Odometry Started." RESET);

    double GRAV_ = 9.82;
    Util::GetParam(nh_ptr, "GRAV", GRAV_, 9.82);
    GRAV = Vector3d(0, 0, GRAV_);

    vector<double> T_B_V_ = {1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1};
    Util::GetParam(nh_ptr, "T_B_V", T_B_V_);
    T_B_V = Matrix<double, 4, 4, RowMajor>(&T_B_V_[0]);
    R_B_V = T_B_V.block<3,3>(0, 0);
    t_B_V = T_B_V.block<3,1>(0, 3);

    printf("Received T_B_V: \n");
    cout << T_B_V << endl;

    // Subscribe to the IMU topic
    string imu_topic; Util::GetParam(nh_ptr, "imu_topic", imu_topic);
    auto imu_sub = nh_ptr->create_subscription<RosImuMsg>(imu_topic, rclcpp::QoS(10), imuCB);

    // Subscribe to the odom topic
    auto odom_sub = nh_ptr->create_subscription<RosOdomMsg>("opt_odom", rclcpp::QoS(10), odomCB);

    // Spin
    rclcpp::spin(nh_ptr);
    rclcpp::shutdown();

    return 0;
}
