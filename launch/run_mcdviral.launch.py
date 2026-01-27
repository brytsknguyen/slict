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
loop_en       = 1                                           # Set the 1 to enable loop closure in slict
use_prior_map = 0                                           # Set to 1 to load and localize on the prior map directly

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
            {"autoexit"      :  autoexit},
            {"use_prior_map" :  use_prior_map},
            {"loop_en"       :  loop_en},
            {"log_dir"       :  exp_log_dir}
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