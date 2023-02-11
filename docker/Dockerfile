FROM ros:noetic-perception

ENV CERES_VERSION="2.1.0"
ENV PCL_VERSION="1.10.0"
# ENV USER="root"
ENV CATKIN_WS=/root/catkin_ws
ENV DATA_PATH=/root/dataset
ENV LIBRARY=/root/library

RUN mkdir -p $CATKIN_WS/src
RUN mkdir -p $DATA_PATH
RUN mkdir -p $LIBRARY

# set up processor number used to compile library
RUN if [ "x$(nproc)" = "x1" ] ; then export USE_PROC=1 ; else export USE_PROC=$(($(nproc)/2)) ; fi && \
    # Install dependencies
    apt-get update && apt-get install -y \
    git \
    cmake \
    nano \
    ros-${ROS_DISTRO}-tf-conversions \
    libatlas-base-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev && \
    # Build and install Ceres
    cd $LIBRARY && \
    git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout ${CERES_VERSION} && \
    mkdir build && cd build && \
    cmake .. && \
    make -j${USE_PROC} install && \
    cd ../.. && \
    rm -rf ./ceres-solver && \
    mkdir -p $CATKIN_WS/src/ufomap && \
    mkdir -p $CATKIN_WS/src/slict && \
    mkdir -p $CATKIN_WS/src/livox_ros_driver && \
    cd

# Clone UFOMap and checkout the devel_surfel branch
RUN git clone https://github.com/brytsknguyen/ufomap $CATKIN_WS/src/ufomap && \
    cd $CATKIN_WS/src/ufomap && git checkout devel_surfel && cd

# Clone and checkout the livox_ros_driver
RUN git clone https://github.com/brytsknguyen/livox_ros_driver $CATKIN_WS/src/livox_ros_driver

# Copy slict code
COPY ./ $CATKIN_WS/src/slict

# Build the catkin workspace
WORKDIR $CATKIN_WS
RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash; cd ${CATKIN_WS}; catkin_make_isolated' && \
    sed -i '/exec "$@"/i source "/root/catkin_ws/devel_isolated/setup.bash"' /ros_entrypoint.sh