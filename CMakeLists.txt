cmake_minimum_required(VERSION 3.8)
project(slict)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17 -Wfatal-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Ros packages
find_package(tf2_ros REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# Message packages
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(cf_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# ROS wrapped packges
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(ufomap REQUIRED)
find_package(ufomap_msgs REQUIRED)
find_package(ufomap_ros REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Other packages
find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# ## Find catkin macros and libraries native to ubuntu
# find_package(catkin REQUIRED COMPONENTS
#   dynamic_reconfigure
#   cv_bridge
#   pcl_conversions
#   pcl_ros
#   std_msgs
#   sensor_msgs
#   geometry_msgs
#   nav_msgs
#   message_generation
#   message_runtime
#   image_transport
#   ufomap_msgs
#   ufomap_ros
#   livox_ros_driver
#   livox_ros_driver2
# )

## Find 3rd-party packages and libraries
# find_package(ufomap REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PCL    REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres  REQUIRED)
find_package(Sophus REQUIRED)

# ## Generate messages in the 'msg' folder
# add_message_files(
#   DIRECTORY msg
#   FILES
#   FeatureCloud.msg
#   OptStat.msg
#   TimeLog.msg
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# To generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FeatureCloud.msg"
  "msg/OptStat.msg"
  "msg/TimeLog.msg"
  "srv/GlobalMapsPublish.srv"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs builtin_interfaces
)
ament_export_dependencies(rosidl_default_runtime)

# Include directories
include_directories(
	include
  include/ikdTree
	${PCL_INCLUDE_DIRS}
  # ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIR}
)


# Main LIO
add_executable(${PROJECT_NAME}_estimator src/Estimator.cpp include/ikdTree/ikd_Tree.cpp src/tmnSolver.cpp src/PointToMapAssoc.cpp)
target_include_directories(${PROJECT_NAME}_estimator PRIVATE /opt/ros/jazzy/include)
ament_target_dependencies(${PROJECT_NAME}_estimator
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  image_transport
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
rosidl_target_interfaces(${PROJECT_NAME}_estimator ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_compile_options(${PROJECT_NAME}_estimator PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_estimator
  ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}
  ${cpp_typesupport_target}
  OpenMP::OpenMP_CXX
  fmt::fmt
  UFO::Map
)
install(TARGETS
  ${PROJECT_NAME}_estimator
  DESTINATION lib/${PROJECT_NAME}
)

# Sensor sync
add_executable(${PROJECT_NAME}_sensorsync src/SensorSync.cpp)
target_include_directories(${PROJECT_NAME}_sensorsync PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_sensorsync
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  image_transport
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
rosidl_target_interfaces(${PROJECT_NAME}_sensorsync ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_compile_options(${PROJECT_NAME}_sensorsync PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_estimator
  ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}
  ${cpp_typesupport_target}
  OpenMP::OpenMP_CXX
  fmt::fmt
  UFO::Map
)# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_sensorsync
  DESTINATION lib/${PROJECT_NAME}
)

# IMU prediction
add_executable(${PROJECT_NAME}_imu_odom src/ImuOdom.cpp)
target_include_directories(${PROJECT_NAME}_imu_odom PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_imu_odom
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  image_transport
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
rosidl_target_interfaces(${PROJECT_NAME}_imu_odom ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_compile_options(${PROJECT_NAME}_imu_odom PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_imu_odom
  ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}
  ${cpp_typesupport_target}
  OpenMP::OpenMP_CXX
  fmt::fmt
  UFO::Map
)# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_imu_odom
  DESTINATION lib/${PROJECT_NAME}
)

# startregion Converter of different lidar formats to ouster ----------------------------------------------------------

# Velodyne
add_executable(${PROJECT_NAME}_velodyne_to_ouster src/lidar_converters/VelodyneToOuster.cpp)
target_include_directories(${PROJECT_NAME}_velodyne_to_ouster PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_velodyne_to_ouster
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
target_compile_options(${PROJECT_NAME}_velodyne_to_ouster PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_velodyne_to_ouster ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt UFO::Map)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_velodyne_to_ouster
  DESTINATION lib/${PROJECT_NAME}
)

# Livox avia, mid-70, mid360
add_executable(${PROJECT_NAME}_livox_to_ouster src/lidar_converters/LivoxToOuster.cpp)
target_include_directories(${PROJECT_NAME}_livox_to_ouster PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_livox_to_ouster
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  livox_ros_driver2
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
target_compile_options(${PROJECT_NAME}_livox_to_ouster PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_livox_to_ouster ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt UFO::Map)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_livox_to_ouster
  DESTINATION lib/${PROJECT_NAME}
)

# Hesai
add_executable(${PROJECT_NAME}_hesai_to_ouster src/lidar_converters/HesaiToOuster.cpp)
target_include_directories(${PROJECT_NAME}_hesai_to_ouster PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_hesai_to_ouster
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
target_compile_options(${PROJECT_NAME}_hesai_to_ouster PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_hesai_to_ouster ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt UFO::Map)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_hesai_to_ouster
  DESTINATION lib/${PROJECT_NAME}
)

# Bpearl
add_executable(${PROJECT_NAME}_bpearl_to_ouster src/lidar_converters/BPearlToOuster.cpp)
target_include_directories(${PROJECT_NAME}_bpearl_to_ouster PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_bpearl_to_ouster
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
target_compile_options(${PROJECT_NAME}_bpearl_to_ouster PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_bpearl_to_ouster ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt UFO::Map)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_bpearl_to_ouster
  DESTINATION lib/${PROJECT_NAME}
)

# Mulran dataset
add_executable(${PROJECT_NAME}_mulran_to_ouster src/lidar_converters/MulranToOuster.cpp)
target_include_directories(${PROJECT_NAME}_mulran_to_ouster PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_mulran_to_ouster
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
target_compile_options(${PROJECT_NAME}_mulran_to_ouster PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_mulran_to_ouster ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt UFO::Map)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_mulran_to_ouster
  DESTINATION lib/${PROJECT_NAME}
)

# M2dgr dataset
add_executable(${PROJECT_NAME}_m2dgr_to_ouster src/lidar_converters/M2DGRToOuster.cpp)
target_include_directories(${PROJECT_NAME}_m2dgr_to_ouster PRIVATE /opt/ros/jazzy/include) # To include sophus for some stupid reason
ament_target_dependencies(${PROJECT_NAME}_m2dgr_to_ouster
  rclcpp
  tf2_ros
  visualization_msgs
  sensor_msgs std_msgs geometry_msgs nav_msgs tf2_msgs
  pcl_conversions
  Sophus
  ufomap
  ufomap_msgs
  ufomap_ros
)
target_compile_options(${PROJECT_NAME}_m2dgr_to_ouster PRIVATE ${OpenMP_CXX_FLAGS} "-isystem ${PCL_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME}_m2dgr_to_ouster ${CERES_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX fmt::fmt UFO::Map)
# Link the executable to the workspace library
install(TARGETS
  ${PROJECT_NAME}_m2dgr_to_ouster
  DESTINATION lib/${PROJECT_NAME}
)

# closeregion Converter of different lidar formats to ouster ----------------------------------------------------------

# Required to generate the message
ament_export_dependencies(rosidl_default_runtime)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()