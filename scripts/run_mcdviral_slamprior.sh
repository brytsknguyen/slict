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

export DATASET_PUBLISHED=/media/tmn/mySataSSD1/DATASETS/MCDVIRAL/PublishedSequencesUnzipped
export DATASET_UNPUBLISHED=/media/tmn/mySataSSD1/DATASETS/MCDVIRAL/UnpublishedSequences
export LAUNCH_FILE=run_mcdviral_slamprior.launch

# Find the available sequences
SEQUENCES=( 
            $DATASET_PUBLISHED/ntu_day_01
            $DATASET_PUBLISHED/ntu_day_02
            $DATASET_PUBLISHED/ntu_day_10
            $DATASET_PUBLISHED/ntu_night_04
            $DATASET_PUBLISHED/ntu_night_08
            $DATASET_PUBLISHED/ntu_night_13
            $DATASET_UNPUBLISHED/ntu_day_03
            $DATASET_UNPUBLISHED/ntu_day_04
            $DATASET_UNPUBLISHED/ntu_day_05
            $DATASET_UNPUBLISHED/ntu_day_06
            $DATASET_UNPUBLISHED/ntu_day_07
            $DATASET_UNPUBLISHED/ntu_day_08
            $DATASET_UNPUBLISHED/ntu_day_09
            $DATASET_UNPUBLISHED/ntu_night_01
            $DATASET_UNPUBLISHED/ntu_night_02
            $DATASET_UNPUBLISHED/ntu_night_03
            $DATASET_UNPUBLISHED/ntu_night_05
            $DATASET_UNPUBLISHED/ntu_night_06
            $DATASET_UNPUBLISHED/ntu_night_07
            $DATASET_UNPUBLISHED/ntu_night_09
            $DATASET_UNPUBLISHED/ntu_night_10
            $DATASET_UNPUBLISHED/ntu_night_11
            $DATASET_UNPUBLISHED/ntu_night_12
          )

EPOC_DIR=/media/$USER/mySataSSD1/DATASETS/MCDVIRAL/Experiment/slict_slamprior_2

for n in {1..3}; do

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