cmake_minimum_required(VERSION 3.0.2)
project(slict)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

## Find catkin macros and libraries native to ubuntu
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  tf
  tf_conversions
  roscpp
  rospy
  cv_bridge
  pcl_conversions
  pcl_ros
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  message_generation
  message_runtime
  image_transport
  ufomap_msgs
  ufomap_ros
  livox_ros_driver
  livox_ros_driver2
)

find_package(ufomap REQUIRED)

## Find 3rd-party packages and libraries
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
# find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

## Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY msg
  FILES
  FeatureCloud.msg
  # KeyFrameData.msg
  OptStat.msg
  TimeLog.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  globalMapsPublish.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
)

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_runtime 
  message_generation
)

###########
## Build ##
###########

# Include directories
include_directories(
	include
  include/ikdTree
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
  # ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
)


## Declare a C++ library

# Main LIO
add_executable(${PROJECT_NAME}_estimator src/Estimator.cpp include/ikdTree/ikd_Tree.cpp src/mySolver.cpp src/PointToMapAssoc.cpp)
add_dependencies(${PROJECT_NAME}_estimator ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_estimator PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_estimator ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Synchronizer for lidar and IMU
add_executable(${PROJECT_NAME}_sensorsync src/SensorSync.cpp)
add_dependencies(${PROJECT_NAME}_sensorsync ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_sensorsync PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_sensorsync ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# IMU prediction
add_executable(${PROJECT_NAME}_imu_odom src/ImuOdom.cpp)
add_dependencies(${PROJECT_NAME}_imu_odom ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_imu_odom PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_imu_odom ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)

# Converter of different lidar formats to ouster
# Velodyne
add_executable(${PROJECT_NAME}_velodyne_to_ouster src/lidar_converters/VelodyneToOuster.cpp)
add_dependencies(${PROJECT_NAME}_velodyne_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_velodyne_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_velodyne_to_ouster ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Reverse Velodyne
add_executable(${PROJECT_NAME}_OusterToVelodyne src/lidar_converters/OusterToVelodyne.cpp)
add_dependencies(${PROJECT_NAME}_OusterToVelodyne ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_OusterToVelodyne PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_OusterToVelodyne ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Livox avia, mid-70
add_executable(${PROJECT_NAME}_livox_to_ouster src/lidar_converters/LivoxToOuster.cpp)
add_dependencies(${PROJECT_NAME}_livox_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_livox_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_livox_to_ouster ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Livox mid-360
add_executable(${PROJECT_NAME}_livox2_to_ouster src/lidar_converters/Livox2ToOuster.cpp)
add_dependencies(${PROJECT_NAME}_livox2_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_livox2_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_livox2_to_ouster ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Hesai
add_executable(${PROJECT_NAME}_hesai_to_ouster src/lidar_converters/HesaiToOuster.cpp)
add_dependencies(${PROJECT_NAME}_hesai_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_hesai_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_hesai_to_ouster ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Bpearl
add_executable(${PROJECT_NAME}_bpearl_to_ouster src/lidar_converters/BPearlToOuster.cpp)
add_dependencies(${PROJECT_NAME}_bpearl_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_bpearl_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_bpearl_to_ouster ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# Mulran dataset
add_executable(${PROJECT_NAME}_mulran_to_ouster src/lidar_converters/MulranToOuster.cpp)
add_dependencies(${PROJECT_NAME}_mulran_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mulran_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mulran_to_ouster ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)
# M2dgr dataset
add_executable(${PROJECT_NAME}_m2dgr_to_ouster src/lidar_converters/M2DGRToOuster.cpp)
add_dependencies(${PROJECT_NAME}_m2dgr_to_ouster ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_m2dgr_to_ouster PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_m2dgr_to_ouster ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} UFO::Map)