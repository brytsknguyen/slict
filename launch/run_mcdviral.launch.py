import os
from launch import LaunchDescription
from launch.event_handlers import OnProcessExit
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.actions import RegisterEventHandler, LogInfo, EmitEvent, DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

datapath = '/home/tmn/DATASETS/MCDVIRAL/'
bag_file = datapath + '/ntu_day_01/'

launch_nodes = {}

def generate_launch_description():

    # Load the config file
    config = os.path.join(
        get_package_share_directory("slict"),
        'config',
        'mcdviral_atv.yaml'
    )

    # Run the estimator node
    launch_nodes['estimator_node'] = Node(
        package     = 'slict',
        executable  = 'slict_estimator',                # Name of the executable built by your package
        name        = 'slict_estimator_node',           # Optional: gives the node instance a name
        output      = 'screen',                         # Print the node output to the screen
        parameters  = [
            config,
            {"use_prior_map" : 0}
        ]
    )

    # Run the sensor sync
    launch_nodes['sensorsync_node'] = Node(
        package     = 'slict',
        executable  = 'slict_sensorsync',               # Name of the executable built by your package
        name        = 'slict_sensorsync_node',          # Optional: gives the node instance a name
        output      = 'log',                         # Print the node output to the screen
        parameters  = [config]
    )

    # Play the livox to ouster converter
    launch_nodes['livox_to_ouster_node'] = Node(
        package     = 'slict',
        executable  = 'slict_livox_to_ouster',          # Name of the executable built by your package
        name        = 'slict_livox_to_ouster_node',     # Optional: gives the node instance a name
        output      = 'log',                            # Print the node output to the screen
        parameters  = [
            {"intensityConvCoef"   : 1.0},
        ]
    )

    # Play all the bag files under the path
    launch_nodes['rosbag_play_node'] = ExecuteProcess(
        cmd=['bash', '-c',
            f'''
                trap "kill 0" INT TERM EXIT
                for b in "{bag_file}"/*_os1_mid70_vn100_jazzy; do
                echo "Playing $b"
                ros2 bag play "$b" --read-ahead-queue-size 5000 &
                done
                wait
            '''
        ],
        output='screen'
    )

    launch_nodes["rviz2"] = Node(
        package     = "rviz2",
        executable  = "rviz2",
        name        = "rviz2",
        output      = "screen",
        # emulate_tty = True,
        arguments   = ["-d", 'mcdviral.rviz'],          # omit this line if you don't have a config
        parameters  = [],                               # optional: passes use_sim_time etc. (from your global yaml)
        # additional_env={
        #     # wipe anything that might inject snap libs
        #     "LD_LIBRARY_PATH": "",
        # },
    )

    return LaunchDescription(list(launch_nodes.values()))