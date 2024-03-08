#!bin/bash

ROS_PKG=slict

# catkin build $ROS_PKG
# source /home/$USER/ros_ws/slict_ws/devel/setup.bash

# Get the current directory
CURR_DIR=$(pwd)
echo CURRENT DIR: $CURR_DIR

# Get the location of the package
PACKAGE_DIR=$(rospack find $ROS_PKG)
echo $ROS_PKG DIR: $PACKAGE_DIR

CAPTURE_SCREEN=false;
LOG_DATA=true;

DATASET_LOCATION=/media/$USER/mySataSSD21/DATASETS/NTU_VIRAL/DATA/
LAUNCH_FILE=run_ntuviral.launch

# Find the available sequences
SEQUENCES=( 
            $DATASET_LOCATION/spms_01
            $DATASET_LOCATION/spms_02
            $DATASET_LOCATION/spms_03
            $DATASET_LOCATION/eee_01
            $DATASET_LOCATION/eee_02
            $DATASET_LOCATION/eee_03
            $DATASET_LOCATION/nya_01
            $DATASET_LOCATION/nya_02
            $DATASET_LOCATION/nya_03
            $DATASET_LOCATION/rtp_01
            $DATASET_LOCATION/rtp_02
            $DATASET_LOCATION/rtp_03
            $DATASET_LOCATION/tnp_01
            $DATASET_LOCATION/tnp_02
            $DATASET_LOCATION/tnp_03
            $DATASET_LOCATION/sbs_01
            $DATASET_LOCATION/sbs_02
            $DATASET_LOCATION/sbs_03
          )

EPOC_DIR=/media/$USER/mySataSSD21/DATASETS/NTU_VIRAL/Experiment/slict_30012024

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
                       _ROSBAG_PATH="$seq/${seq_basename}_mod.bag" \
                       _CAPTURE_SCREEN=$CAPTURE_SCREEN \
                       _LOG_DATA=$LOG_DATA \
                       _LOG_PATH="$EPOC_DIR_N/result_$seq_basename" \
                       _LOOP_EN=0 \
                      #  _IMU_TOPIC=/vn200/imu \

      printf "\n"
  )
  done

done