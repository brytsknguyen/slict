#!/bin/bash

# Kill all child processes when this script is killed
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

DATA_PATH=${1:-/media/tmn/mySataSSD3/NTU_VIRAL}
DOCKER_IMG=brytsknguyen/slict-noetic-focal

# Pull the image from docker hub
docker pull $DOCKER_IMG

PKG_DIR=$(rospack find slict)

# Launch rviz in background to give terminal to slict
(rviz -d ${PKG_DIR}/launch/ntuviral.rviz) &

# Launch slict
docker run \
        -it \
        --rm \
        --net=host \
        -v ${PKG_DIR}:/root/catkin_ws/src/slict \
        -v ${DATA_PATH}:/root/dataset/ \
        -e USER=root \
        $DOCKER_IMG \
        /bin/bash -c \
        "cd /root/catkin_ws/; \
         catkin_make_isolated; \
         source devel_isolated/setup.bash; \
         roslaunch slict run_ntuviral.launch data_path:=/root/dataset/ exp_log_dir:=/root/slict_logs/ntuviral"