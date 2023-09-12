#!bin/bash

ROS_PKG=slict

catkin build $ROS_PKG
source /home/$USER/dev_ws/devel/setup.bash

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

export DATASET_LOCATION=/media/$USER/mySataSSD1/DATASETS/Helmet
export LAUNCH_FILE=run_helmet.launch

# Find the available sequences
SEQUENCES=( 
            $DATASET_LOCATION/helmet_01
            $DATASET_LOCATION/helmet_02
            $DATASET_LOCATION/helmet_03
          )

#region -------------------------------------------------------------------------------------------------------------#

EPOC_DIR=/media/$USER/mySataSSD1/DATASETS/Helmet/Experiment/slict/slict_noloop

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
                     _LOG_PATH="$EPOC_DIR/result_$seq_basename" \
                     _LOOP_EN=0 \
                    #  _IMU_TOPIC=/vn200/imu \
    printf "\n"
)
done

#endregion ----------------------------------------------------------------------------------------------------------#