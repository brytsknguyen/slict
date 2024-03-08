#!bin/bash

ROS_PKG=slict

catkin build $ROS_PKG
source /home/$USER/ros_ws/slict_ws/devel/setup.bash

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

export DATASET_LOCATION=/media/tmn/mySataSSD1/DATASETS/MCDVIRAL/PublishedSequencesUnzipped
export LAUNCH_FILE=run_mcdviral.launch

# Find the available sequences
SEQUENCES=( 
            $DATASET_LOCATION/kth_day_06
            $DATASET_LOCATION/kth_day_09
            $DATASET_LOCATION/kth_day_10
            $DATASET_LOCATION/kth_night_01
            $DATASET_LOCATION/kth_night_04
            $DATASET_LOCATION/kth_night_05
            # $DATASET_LOCATION/ntu_day_01
            # $DATASET_LOCATION/ntu_day_02
            # $DATASET_LOCATION/ntu_day_10
            # $DATASET_LOCATION/ntu_night_04
            # $DATASET_LOCATION/ntu_night_08
            # $DATASET_LOCATION/ntu_night_13
            $DATASET_LOCATION/tuhh_day_02
            $DATASET_LOCATION/tuhh_day_03
            $DATASET_LOCATION/tuhh_day_04
            $DATASET_LOCATION/tuhh_night_07
            $DATASET_LOCATION/tuhh_night_08
            $DATASET_LOCATION/tuhh_night_09
          )

EPOC_DIR=/media/$USER/mySataSSD1/DATASETS/MCDVIRAL/Experiment/slict_11012024_hhs

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