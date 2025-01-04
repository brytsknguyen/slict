#!bin/bash

ROS_PKG=slict

catkin build $ROS_PKG
source $ROS_PKG/../../devel/setup.bash

# Get the current directory
CURR_DIR=$(pwd)
echo CURRENT DIR: $CURR_DIR

# Get the location of the package
PACKAGE_DIR=$(rospack find $ROS_PKG)
echo $ROS_PKG DIR: $PACKAGE_DIR

# Enable screen capture
CAPTURE_SCREEN=false;
# Enable data logging
LOG_DATA=true;

export DATASET_LOCATION="/root/LAURA/ros1_export"
export LAUNCH_FILE=run_laura.launch

# Find the available sequences
SEQUENCES=( 
            # $DATASET_LOCATION/alley_fast_north_south
            # $DATASET_LOCATION/alley_loop
            $DATASET_LOCATION/alley_loop_fast
            # $DATASET_LOCATION/grove_clockwise
            # $DATASET_LOCATION/grove_counterclockwise
            # $DATASET_LOCATION/town_clockwise
            # $DATASET_LOCATION/town_counterclockwise
            # $DATASET_LOCATION/town_courtyard_clockwise
            # $DATASET_LOCATION/town_trees_a_building
          )

EPOC_DIR=/root/slict_logs/LAURA

for n in {1..1}; do

  EPOC_DIR_N=${EPOC_DIR}/try_$n

  for seq in ${SEQUENCES[@]};
  do
  (
      printf "\n"
      seq_basename="$(basename $seq)"
      printf "Sequence: $seq. Basename: $seq_basename\n"

      ./run_one_bag.sh _LAUNCH_FILE=$LAUNCH_FILE \
                       _ROS_PKG=$ROS_PKG \
                       _ROSBAG_PATH="$seq.bag" \
                       _CAPTURE_SCREEN=$CAPTURE_SCREEN \
                       _LOG_DATA=$LOG_DATA \
                       _LOG_PATH="$EPOC_DIR_N/result_$seq_basename" \
                       _LOOP_EN=0 \
                      #  _IMU_TOPIC=/vn200/imu \

      printf "\n"
  )
  done

done