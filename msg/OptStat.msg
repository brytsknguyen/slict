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
std_msgs/Header header          # Header time stamp is the same with the latest feature batch stamp

int32 opt_num                   # Optimization counter
int32 opt_num_sub               # Outer iterations for optimization

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

geometry_msgs/Quaternion qest   # Rotation estimated in the latest state
geometry_msgs/Point pest        # Translation estimated in the latest state
geometry_msgs/Point vest        # Velocity estimated in the latest state

geometry_msgs/Quaternion qimu   # Rotation propagated to the latest state
geometry_msgs/Point pimu        # Translation propagated to the latest state
geometry_msgs/Point vimu        # Velocity propagated to the latest state

geometry_msgs/Quaternion qdsk   # Rotation estimated for the deskew proccess
geometry_msgs/Point pdsk        # Translation estimated for the deskew proccess

geometry_msgs/Quaternion qref   # Rotation relative to the last key frame
geometry_msgs/Point pref        # Translation relative to the last key frame

int32   last_kf_id              # ID of the key frame that Prel and Qrel are relative to

int32   imubuf                  # Number of IMU samples waiting in the buffer.
int32   mfcbuf                  # Number of feature batch waiting in the buffer.
int32   odombuf                 # Number of odometry samples
int32   uwbbuf                  # Number of UWB samples waiting in the buffer.
int32   cambuf                  # Number of image pairs in buffer.
int32   visfea                  # Number of tracked features
int32   vismarg                 # Number of marginalized depths
int32   visposd                 # Number of positive depth features (successful triangulation)
int32   visnegd                 # Number of negative depth features (failed to triangulate)
int32   visfrm                  # Number of feature frames, should be equal sliding window in normal condition
int32   keyfrm                  # Number of key frames in global map.
int32   kfcand                  # Number of key frame candidates in buffer
float32 margperc                # Percentage of marginalized points

int32   local_surf_map_size     # Size of the surface map
int32   local_surf_map_ds_size  # Size of the surface map after downsampling
int32   local_edge_map_size     # Size of the edge map
int32   local_edge_map_ds_size  # Size of the edge map after downsampling

int32   coeff_ds_rate           # The downsample rate of the features to be put through FMM

int8    ancs                    # Number of anchors used
float32 dist_bias               # Distance bias of UWB
int8    deskewed                # Flag to show if last cloud has been deskewed
int8    open_xtrz_estimation    # Flag to show if extrinsic estimation is started
int8    td_is_const             # Flag to show if td estimation is enabled
float32 vis_td_avr              # Average visual time delay
float32 vis_td_max              # Maximum visual time delay
float32 vis_td_min              # Minimum visual time delay
int32   vis_be4op_rmv           # Visual features removed before optimization
int32   vis_aftop_rmv           # Visual features removed after optimization
int8    cam_stream_cont         # Whether the camera stream is continuous
int32   locmap_kf_near          # Number of neareast key frames
int32   locmap_kf_last          # Number of last key frames
int32   locmap_kf_in_radius     # Number of key frames in the radius

int32   surf_factors            # Number of surface point2map associations, equal the factors
int32   edge_factors            # Number of edge point2map associations, equal the factors
int32   imu_factors             # Number of IMU factors
int32   prop_factors            # Number of IMU-propagated factors
int32   vel_factors             # Number of Velocity factors
int32   odom_factors            # Number of odometry factors
int32   uwb_factors             # Number of all UWB factors
int32   uwb_factors_basic       # Number of basic UWB factors
int32   uwb_total_samples       # Number of all UWB samples since the beginning
int32   uwb_total_outliers      # Number of all UWB samples since the beginning
int32   uwb_slwd_samples        # Number of all UWB samples in the sliding window
int32   uwb_slwd_outliers       # Number of all UWB samples since the beginning
int32   visfea_factors          # Number of visual factors
int32   v2m_factors             # Number of visual-feature-to-map factors

float64 j0                      # Original cost in the optimization
float64 jk                      # Final cost in the optimization

float64 j0surf                  # Original cost of surface factors
float64 jksurf                  # Final cost of surface factors

float64 j0edge                  # Original cost of edge factors
float64 jkedge                  # Final cost of edge factors

float64 j0imu                   # Original cost of pretintegration factors
float64 jkimu                   # Final cost of pretintegration factors

float64 j0prop                  # Original cost of pretintegration factors
float64 jkprop                  # Final cost of pretintegration factors

float64 j0vel                   # Original cost of velocity factors
float64 jkvel                   # Final cost of velocity factors

float64 j0odom                  # Original cost of pretintegration factors
float64 jkodom                  # Final cost of pretintegration factors

float64 j0uwb                   # Original cost of UWB factors
float64 jkuwb                   # Final cost of UWB factors

float64 j0vis                   # Original cost of visual features
float64 jkvis                   # Final cost of visual features

float64 j0mmm                   # Original cost of visual features to map factors
float64 jkmmm                   # Final cost of visual features to map factors

# Bundle Adjustment status
int32   banum                   # Counter
int32   baitr                   # Number of iterations

float32 balooptime              # Time from one optimization to another
float32 basolvetime             # Ceres Solving time
float32 bapreopttime            # Time for preprocessing befor optimization time

int32   barelposefactors        # Number of 4DOF pose factors
int32   baloopfactors           # Number of UWB factors

float32 baj0                    # Initial cost of bundle adjustment
float32 bajk                    # Final cost of the bundle adjustment

float32 baj0relpose             # Initial cost for the 4DOF pose
float32 bajkrelpose             # Final cost of the 4DOF pose
float32 baj0loop                # Initial cost of UWB
float32 bajkloop                # Final cost for UWB

float64 basvdl2w1               # svd coefficients
float64 basvdl2w2               # svd coefficients
float64 basvdl2w3               # svd coefficients

float64 bayawl2w                # yaw pitch roll of the L2W transform
float64 bapitchl2w              # yaw pitch roll of the L2W transform
float64 barolll2w               # yaw pitch roll of the L2W transform

float64 batransxl2w             # x y z of the L2W transform
float64 batransyl2w             # x y z of the L2W transform
float64 batranszl2w             # x y z of the L2W transform

float64 bauwbbias               # estimated bias in the BA process

int8    basvdpassed             # x y z of the L2W transform
int8    bal2wfix                # x y z of the L2W transform