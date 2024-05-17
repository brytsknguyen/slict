# This file is part of slict.
# 
# Copyright (C) 2020 Thien-Minh Nguyen <thienminh.nguyen at ntu dot edu dot sg>,
# School of EEE
# Nanyang Technological Univertsity, Singapore
# 
# For more information please see <https://britsknguyen.github.io>.
# or <https://github.com/brytsknguyen/slict>.
# If you use this code, please cite the respective publications as
# listed on the above websites.
# 
# slict is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# slict is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with slict.  If not, see <http://www.gnu.org/licenses/>.

#
# Created by Thien-Minh Nguyen on 15/12/20.
#

# Cloud Info
Header header 

int32[]   startRingIndex
int32[]   endRingIndex

int32[]   pointColInd                    # point column index in range image
float32[] pointRange                     # point range 

float64   scanStartTime
float64   scanEndTime

# To inform the pose of the pointcloud
geometry_msgs/Pose pose

# Point cloud messages
sensor_msgs/PointCloud2 extracted_cloud  # extracted cloud full
sensor_msgs/PointCloud2 edge_cloud       # extracted corner feature
sensor_msgs/PointCloud2 surf_cloud       # extracted surface feature

# Imu messges from start time to end time
sensor_msgs/Imu[] imu_msgs