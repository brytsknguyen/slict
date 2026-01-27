import os
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.actions import RegisterEventHandler, LogInfo, EmitEvent, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# Experiment data path
datapath = '/home/tmn/DATASETS/MCDVIRAL/'
bag_file = datapath + '/ntu_day_01/'

# Session level experiment
exp_log_dir   = "/home/tmn/slict_logs/mcdviral"             # Directory to log data
autoexit      = 1                                           # Set to 1 so that slict_estimator exits when does not receive data for a while
loop_en       = 0                                           # Set the 1 to enable loop closure in slict
use_prior_map = 1                                           # Set to 1 to load and localize on the prior map directly
prior_map_dir = ""
prior_map_dir = "/home/tmn/DATASETS/MCDVIRAL/PriorMap/NTU"  if 'ntu_'  in bag_file else prior_map_dir
prior_map_dir = "/home/tmn/DATASETS/MCDVIRAL/PriorMap/KTH"  if 'kth_'  in bag_file else prior_map_dir
prior_map_dir = "/home/tmn/DATASETS/MCDVIRAL/PriorMap/TUHH" if 'tuhh_' in bag_file else prior_map_dir
if use_prior_map != 0:
    assert prior_map_dir != "", "a path to prior map must be set when use_prior_map is set!"
    assert loop_en == 0, "loop closure must be disabled when using prior map!"

# List of transform to align with prior map
tf_Lprior_L0 = {
    "ntu_day_01":    [  49.28,  107.38,   7.58,  -41,    0,   0],
    "ntu_day_02":    [  61.99,  119.58,   7.69, -134,    0,   0],
    "ntu_day_03":    [  62.82,  119.51,   7.70,   39,    0,   0],
    "ntu_day_04":    [  55.52,  110.70,   7.72,  -40,    0,   0],
    "ntu_day_05":    [  59.18,  116.06,   7.72,   42,    0,   0],
    "ntu_day_06":    [  48.67,  109.16,   7.64,  -28,    0,   0],
    "ntu_day_07":    [ -27.11,   -1.57,   8.73,   -8,    0,   0],
    "ntu_day_08":    [  40.58,   15.90,   6.56,   48,    0,   0],
    "ntu_day_09":    [  71.90,   57.99,   7.67,   80,    0,   0],
    "ntu_day_10":    [  39.49,   23.48,   6.54,   36,    0,   0],

    "ntu_night_01":  [  59.69,  108.43,   7.82,  -36,    0,   0],
    "ntu_night_02":  [  55.78,  108.37,   7.78,  -32,    0,   0],
    "ntu_night_03":  [ 143.28,   36.80,   8.97, -136,    0,   0],
    "ntu_night_04":  [ 244.20,  -99.86,   5.97,  -32,    0,   0],
    "ntu_night_05":  [  85.37,   73.99,   7.77, -132,    0,   0],
    "ntu_night_06":  [  46.02,   21.03,   6.60, -135,    0,   0],
    "ntu_night_07":  [  55.97,  112.70,   7.75,  -36,    0,   0],
    "ntu_night_08":  [ 195.74,   -8.57,   7.18,  135,    0,   0],
    "ntu_night_09":  [ 234.26,  -41.31,   6.69, -107,    0,   0],
    "ntu_night_10":  [ 194.55, -216.91,  -3.69,  176,    0,   0],
    "ntu_night_11":  [  15.34, -197.79,  -4.99,  124,    0,   0],
    "ntu_night_12":  [  60.77,  -45.23,   2.20, -139,    0,   0],
    "ntu_night_13":  [  81.38,  -18.45,   3.43,   42,    0,   0],

    "kth_day_06":    [  64.41,   66.48,  38.50,  144,    0,   0],
    "kth_day_09":    [  70.40,   63.12,  38.30,  -26,    0,   0],
    "kth_day_10":    [  69.13,   63.57,  38.38,  145,    0,   0],
    "kth_night_01":  [  68.84,  -64.10,  38.43,  138,    0,   0],
    "kth_night_04":  [  71.47,   63.56,  38.37,  -52,    0,   0],
    "kth_night_05":  [  43.81, -131.66,  29.42,   76,    0,   0],

    "tuhh_day_02":   [  45.73,  447.10,  14.69, -157,    0,   0],
    "tuhh_day_03":   [  43.62,  446.53,  14.59, -162,    0,   0],
    "tuhh_day_04":   [  35.15,  114.94,  -1.30, -130,    0,   0],
    "tuhh_night_07": [  43.87,  447.10,  14.59, -138,    0,   0],
    "tuhh_night_08": [  42.93,  446.72,  14.63, -143,    0,   0],
    "tuhh_night_09": [  32.47,  111.61,  -1.42, -149,    0,   0],
}
# Select the init based on the bag file
for key, value in tf_Lprior_L0.items():
    if key in bag_file:
        tf_Lprior_L0_init = value
        break

# Create nodes to launch
launch_nodes = {}
def generate_launch_description():

    # Load the config file
    config = os.path.join(
        get_package_share_directory("slict"),
        'config',
        'mcdviral_atv.yaml'
    )

    # Play the livox to ouster converter
    launch_nodes['livox_to_ouster_node'] = Node(
        package     = 'slict',
        executable  = 'slict_livox_to_ouster',          # Name of the executable built by your package
        name        = 'slict_livox_to_ouster_node',     # Optional: gives the node instance a name
        output      = 'log',                            # Print the node output to the screen
        parameters  = []
    )

    # Run the sensor sync
    launch_nodes['sensorsync_node'] = Node(
        package     = 'slict',
        executable  = 'slict_sensorsync',               # Name of the executable built by your package
        name        = 'slict_sensorsync_node',          # Optional: gives the node instance a name
        output      = 'log',                         # Print the node output to the screen
        parameters  = [config]
    )

    # Run the estimator node
    launch_nodes['estimator_node'] = Node(
        package     = 'slict',
        executable  = 'slict_estimator',                # Name of the executable built by your package
        name        = 'slict_estimator_node',           # Optional: gives the node instance a name
        output      = 'screen',                         # Print the node output to the screen
        parameters  = [
            config,
            {"autoexit"          :  autoexit},
            {"use_prior_map"     :  use_prior_map},
            {"prior_map_dir"     :  prior_map_dir},
            {"tf_Lprior_L0_init" :  tf_Lprior_L0_init},
            {"loop_en"           :  loop_en},
            {"log_dir"           :  exp_log_dir}
        ]
    )

    # Run the IMU propagation node
    launch_nodes['imu_odom_node'] = Node(
        package     = 'slict',
        executable  = 'slict_imu_odom',                # Name of the executable built by your package
        name        = 'slict_imu_odom_node',           # Optional: gives the node instance a name
        output      = 'screen',                         # Print the node output to the screen
        parameters  = [config]
    )

    # Play all the bag files under the path
    launch_nodes['rosbag_play_node'] = ExecuteProcess(
        cmd=['bash', '-c',
            f'''
                # trap "kill 0" INT TERM EXIT
                for b in "{bag_file}"/*_os1_mid70_vn100_jazzy; do
                echo "Playing $b"
                ros2 bag play "$b" --read-ahead-queue-size 5000 &
                done
                wait
            '''
        ],
        output='screen'
    )

    # Visualize
    launch_nodes["rviz2"] = Node(
        package     = "rviz2",
        executable  = "rviz2",
        name        = "rviz2",
        output      = "screen",
        # emulate_tty = True,
        arguments   = ["-d", 'launch/mcdviral.rviz'],          # omit this line if you don't have a config
        parameters  = [],                               # optional: passes use_sim_time etc. (from your global yaml)
        # additional_env={
        #     # wipe anything that might inject snap libs
        #     "LD_LIBRARY_PATH": "",
        # },
    )

    return LaunchDescription(list(launch_nodes.values()))