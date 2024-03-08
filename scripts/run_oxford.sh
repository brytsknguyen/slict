#!bin/bash

ROS_PKG=slict

catkin build $ROS_PKG
# source /home/$USER/dev_ws/devel/setup.bash

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

LAUNCH_FILE=run_oxford.launch
DATASET_LOCATION=/media/tmn/mySataSSD1/DATASETS/NewerCollegeDataset

# Find the available sequences
SEQUENCES=( 
            $DATASET_LOCATION/06_dynamic_spinning
            $DATASET_LOCATION/01_short_experiment
            $DATASET_LOCATION/02_long_experiment
            $DATASET_LOCATION/05_quad_with_dynamics
            # $DATASET_LOCATION/07_parkland_mound
          )

EPOC_DIR=/media/tmn/mySataSSD1/DATASETS/NewerCollegeDataset/Experiment/slict/oxford_noloop_26112024

for n in {1..5}; do

  EPOC_DIR_N=${EPOC_DIR}/try_$n

  for seq in ${SEQUENCES[@]};
  do
  (
      printf "\n"
      seq_basename="$(basename $seq)"
      printf "Sequence: $seq. Basename: $seq_basename\n"

      ./run_one_bag.sh _LAUNCH_FILE=$LAUNCH_FILE \
                       _ROS_PKG=$ROS_PKG \
                       _ROSBAG_PATH="$seq/*.bag" \
                       _CAPTURE_SCREEN=$CAPTURE_SCREEN \
                       _LOG_DATA=$LOG_DATA \
                       _LOG_PATH="$EPOC_DIR_N/result_$seq_basename" \
                       _LOOP_EN=0 \
                      #  _IMU_TOPIC=/vn200/imu \

      printf "\n"
  )
  done

done