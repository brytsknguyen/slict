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
Header  header   # Header time stamp is the same with the latest feature batch stamp

int32   OptNum                  # Optimization counter
int32   OptNumSub               # Outer iterations for optimization

float32 trun                    # Time from the beginning of processing
float32 tlp                     # Time between the last two optimization runs
float32 tbuildceres             # Time to build ceres problem and solve
float32 tpreopt                 # Time from the beginning of the loop til the optimization
float32 tpostopt                # Time from the beginning of the loop til the optimization
float32 tkf                     # Time searching for key frames
float32 tfb2w                   # Time transforming the CFCs from body to the world frame
float32 tslv                    # Actual optimization solving time
float32 tslvmax                 # Maximum Optimization solving time
float32 timgtr                  # Image tracking time
float32 tsurf                   # Local surface map building time
float32 tedge                   # Local edge map building time
float32 tmapimg                 # Total time for map building time and img tracking
float32 timubuf                 # Time period of the IMU samples waiting in the buffer
int32   iters                   # Number of iterations
int32   fixed_knot_min          # Index of first fixed knots
int32   fixed_knot_max          # Index of last fixed knots

geometry_msgs/Quaternion Qest   # Rotation estimated in the latest state
geometry_msgs/Point Pest        # Translation estimated in the latest state
geometry_msgs/Point Vest        # Velocity estimated in the latest state

geometry_msgs/Quaternion Qimu   # Rotation propagated to the latest state
geometry_msgs/Point Pimu        # Translation propagated to the latest state
geometry_msgs/Point Vimu        # Velocity propagated to the latest state

geometry_msgs/Quaternion Qdsk   # Rotation estimated for the deskew proccess
geometry_msgs/Point Pdsk        # Translation estimated for the deskew proccess

geometry_msgs/Quaternion Qref   # Rotation relative to the last key frame
geometry_msgs/Point Pref        # Translation relative to the last key frame

int32   lastKfId                # ID of the key frame that Prel and Qrel are relative to

int32   imuBuf                  # Number of IMU samples waiting in the buffer.
int32   mfcBuf                  # Number of feature batch waiting in the buffer.
int32   odomBuf                 # Number of odometry samples
int32   uwbBuf                  # Number of UWB samples waiting in the buffer.
int32   camBuf                  # Number of image pairs in buffer.
int32   visFea                  # Number of tracked features
int32   visMarg                 # Number of marginalized depths
int32   visPosD                 # Number of positive depth features (successful triangulation)
int32   visNegD                 # Number of negative depth features (failed to triangulate)
int32   visFrm                  # Number of feature frames, should be equal sliding window in normal condition
int32   keyfrm                  # Number of key frames in global map.
int32   kfcand                  # Number of key frame candidates in buffer
float32 margPerc                # Percentage of marginalized points

int32   localSurfMapSize        # Size of the surface map
int32   localSurfMapDSSize      # Size of the surface map after downsampling
int32   localEdgeMapSize        # Size of the edge map
int32   localEdgeMapDSSize      # Size of the edge map after downsampling

int32   coeffDSRate             # The downsample rate of the features to be put through FMM

int8    ancs                    # Number of anchors used
float32 dist_bias               # Distance bias of UWB
int8    deskewed                # Flag to show if last cloud has been deskewed
int8    openExEstimation        # Flag to show if extrinsic estimation is started
int8    tdIsConst               # Flag to show if td estimation is enabled
float32 visTdAvr                # Average visual time delay
float32 visTdMax                # Maximum visual time delay
float32 visTdMin                # Minimum visual time delay
int32   visBe4OpRmv             # Visual features removed before optimization
int32   visAftOpRmv             # Visual features removed after optimization
int8    camStreamCont           # Whether the camera stream is continuous
int32   locMapKfNear            # Number of neareast key frames
int32   locMapKfLast            # Number of last key frames
int32   locMapKfInRadius        # Number of key frames in the radius

int32   surfFactors             # Number of surface point2map associations, equal the factors
int32   edgeFactors             # Number of edge point2map associations, equal the factors
int32   imuFactors              # Number of IMU factors
int32   propFactors             # Number of IMU-propagated factors
int32   velFactors              # Number of Velocity factors
int32   odomFactors             # Number of odometry factors
int32   uwbFactors              # Number of all UWB factors
int32   uwbFactorsBasic         # Number of basic UWB factors
int32   uwbTotalSamples         # Number of all UWB samples since the beginning
int32   uwbTotalOutliers        # Number of all UWB samples since the beginning
int32   uwbSlWdSamples          # Number of all UWB samples in the sliding window
int32   uwbSlWdOutliers         # Number of all UWB samples since the beginning
int32   visFeaFactors           # Number of visual factors
int32   v2mFactors              # Number of visual-feature-to-map factors

float64 J0                      # Original cost in the optimization
float64 JK                      # Final cost in the optimization

float64 J0Surf                  # Original cost of surface factors
float64 JKSurf                  # Final cost of surface factors

float64 J0Edge                  # Original cost of edge factors
float64 JKEdge                  # Final cost of edge factors

float64 J0Imu                   # Original cost of pretintegration factors
float64 JKImu                   # Final cost of pretintegration factors

float64 J0Prop                  # Original cost of pretintegration factors
float64 JKProp                  # Final cost of pretintegration factors

float64 J0Vel                   # Original cost of velocity factors
float64 JKVel                   # Final cost of velocity factors

float64 J0Odom                  # Original cost of pretintegration factors
float64 JKOdom                  # Final cost of pretintegration factors

float64 J0UWB                   # Original cost of UWB factors
float64 JKUWB                   # Final cost of UWB factors

float64 J0Vis                   # Original cost of visual features
float64 JKVis                   # Final cost of visual features

float64 J0MMM                   # Original cost of visual features to map factors
float64 JKMMM                   # Final cost of visual features to map factors

# Bundle Adjustment status
int32   BANum                   # Counter
int32   BAItr                   # Number of iterations

float32 BALoopTime              # Time from one optimization to another
float32 BASolveTime             # Ceres Solving time
float32 BAPreOptTime            # Time for preprocessing befor optimization time

int32   BARelPoseFactors        # Number of 4DOF pose factors
int32   BALoopFactors           # Number of UWB factors

float32 BAJ0                    # Initial cost of bundle adjustment
float32 BAJK                    # Final cost of the bundle adjustment

float32 BAJ0RelPose             # Initial cost for the 4DOF pose
float32 BAJKRelPose             # Final cost of the 4DOF pose
float32 BAJ0Loop                # Initial cost of UWB
float32 BAJKLoop                # Final cost for UWB

float64 BASVDL2W1               # svd coefficients
float64 BASVDL2W2               # svd coefficients
float64 BASVDL2W3               # svd coefficients

float64 BAYawL2W                # yaw pitch roll of the L2W transform
float64 BAPitchL2W              # yaw pitch roll of the L2W transform
float64 BARollL2W               # yaw pitch roll of the L2W transform

float64 BATransXL2W             # x y z of the L2W transform
float64 BATransYL2W             # x y z of the L2W transform
float64 BATransZL2W             # x y z of the L2W transform

float64 BAUWBBias               # estimated bias in the BA process

int8    BASVDPassed             # x y z of the L2W transform
int8    BAL2WFix                # x y z of the L2W transform