#!/bin/bash
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"

   printf "$KEY: \t%s\n" "$VALUE"
done


# if [ -z "${_IMU_TOPIC+x}" ];
# then
#     _IMU_TOPIC=/imu/imu
#     echo echo "_IMU_TOPIC is not set. Defaulted to '$_IMU_TOPIC'";
# # else
#     # echo "_IMU_TOPIC is set: '$_IMU_TOPIC'";
# fi

# if [ -z "${_CONFIG_SUITE+x}" ];
# then
#     _CONFIG_SUITE=''
#     echo echo "_CONFIG_SUITE is not set. Defaulted to '$_CONFIG_SUITE'";
# # else
#     # echo "_CONFIG_SUITE is set: '$_CONFIG_SUITE'";
# fi

# Reassigning the arguments
export LAUNCH_FILE=$_LAUNCH_FILE;
export ROS_PKG=$_ROS_PKG;
export ROSBAG_PATH=$_ROSBAG_PATH;
export CAPTURE_SCREEN=$_CAPTURE_SCREEN;
export LOG_DATA=$_LOG_DATA;
export LOG_PATH=$_LOG_PATH;
export LOOP_EN=$_LOOP_EN;
# export IMU_TOPIC=$_IMU_TOPIC
# export CONFIG_SUITE=$_CONFIG_SUITE

# Begin the processing

# Find the path to the package
ROS_PKG_DIR=$(ros2 pkg prefix $ROS_PKG)

# Notify the bag file
echo BAG FILE: "${ROSBAG_PATH}";

if $LOG_DATA
then

# Create the log director
mkdir -p $LOG_PATH/ ;
# Copy the config folders for later references
cp -R $ROS_PKG_DIR/config  $LOG_PATH;
cp -R $ROS_PKG_DIR/launch  $LOG_PATH;
cp -R $ROS_PKG_DIR/scripts $LOG_PATH;
# Create folder for BA output
mkdir -p $LOG_PATH/ba;

fi

#Notify the log file
echo LOG DIR: $LOG_PATH;


# Turn on the screen capture if selected
if $CAPTURE_SCREEN
then

echo CAPTURING SCREEN ON;
(
ffmpeg -video_size 1920x1080 -framerate 0.5 -f x11grab -i $DISPLAY+0,0 \
-loglevel quiet -y $LOG_PATH/screen.mp4
) &
FFMPEG_PID=$!

else

echo CAPTURING SCREEN OFF;
sleep 1;

fi

echo FFMPEG PID $FFMPEG_PID

if $LOG_DATA
then

echo LOGGING ON;

# Start the process
(
# xterm -T "GTGEN ${AFFIX}" -geometry 138x40+1075+-8 -e \
# "
ros2 launch $ROS_PKG $LAUNCH_FILE \
autoexit:=1 \
loop_en:=$LOOP_EN \
bag_file:=$ROSBAG_PATH \
log_dir:=$LOG_PATH/ba \
# 2>&1 | tee -a $LOG_PATH/terminal_log.txt
# "
)&

MAIN_PID=$!

# Log the topics
( sleep 1; ros2 topic echo --csv /pred_odom \
> $LOG_PATH/predict_odom.csv ) \
& \
( sleep 1; ros2 topic echo --csv /opt_odom \
> $LOG_PATH/opt_odom.csv ) \
& \
# ( sleep 1; rostopic echo -b $ROSBAG_PATH --csv /leica/pose/relative \
# > $LOG_PATH/leica_pose.csv ) \
# & \
# ( sleep 1; rostopic echo -b $ROSBAG_PATH --csv /dji_sdk/imu \
# > $LOG_PATH/dji_sdk_imu.csv ) \
# & \
# ( sleep 1; rostopic echo -b $ROSBAG_PATH --csv $IMU_TOPIC \
# > $LOG_PATH/vn100_imu.csv ) \
# & \
( sleep 1; ros2 topic echo --csv /opt_stat \
> $LOG_PATH/opt_stat.csv ) \
& \

else

echo LOGGING OFF;
sleep 1;

fi

wait $MAIN_PID;

# Close the screen recorder
kill $FFMPEG_PID;

exit;