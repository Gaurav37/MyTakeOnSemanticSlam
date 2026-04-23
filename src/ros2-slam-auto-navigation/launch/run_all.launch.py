# save as: ~/ros2_ws/src/ros2-slam-auto-navigation/launch/semantic_slam.launch.py

import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# ── Config ────────────────────────────────────────────────────────────────
WORLD_FILE      = '/home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/worlds/warehouse_new.world'
SLAM_CONFIG     = '/home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/config/mapper_params_online_async.yaml'
ROBOT_SPAWN_X   = 0.0
ROBOT_SPAWN_Y   = 0.0
ROBOT_SPAWN_YAW = 0.0


def generate_launch_description():

    # ── Paths ─────────────────────────────────────────────────────────────
    slam_toolbox_dir  = get_package_share_directory('slam_toolbox')
    nav2_bringup_dir  = get_package_share_directory('nav2_bringup')
    pkg_dir           = get_package_share_directory('ros2_slam_auto_navigation')

    # ── 1. Gazebo Simulation (t=0s) ───────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'launch_sim.launch.py')
        ),
        launch_arguments={'world_file': WORLD_FILE}.items()
    )

    # ── 2. SLAM Toolbox (t=10s — wait for Gazebo) ────────────────────────
    slam = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')
                ),
                launch_arguments={
                    'slam_params_file': SLAM_CONFIG,
                    'use_sim_time': 'true'
                }.items()
            )
        ]
    )

    # ── 3. Nav2 (t=20s — wait for SLAM to initialize) ────────────────────
    nav2 = TimerAction(
        period=20.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
                ),
                launch_arguments={
                    'use_sim_time': 'True'
                }.items()
            )
        ]
    )

    # ── 4. RViz (t=25s) ──────────────────────────────────────────────────
    rviz = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=[
                    '-d', os.path.join(
                        nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz'
                    )
                ],
                parameters=[{'use_sim_time': True}],
                output='screen'
            )
        ]
    )

    # ── 5. Initial Pose (t=30s — wait for Nav2 + map to be ready) ────────
    initial_pose = TimerAction(
        period=30.0,
        actions=[
            Node(
                package='ros2_slam_auto_navigation',
                executable='initial_pose_setter',
                name='initial_pose_setter',
                parameters=[{
                    'x':   ROBOT_SPAWN_X,
                    'y':   ROBOT_SPAWN_Y,
                    'yaw': ROBOT_SPAWN_YAW,
                    'use_sim_time': True
                }],
                output='screen'
            )
        ]
    )

    # ── 6. Semantic SAM Node (t=35s) ─────────────────────────────────────
    semantic_node = TimerAction(
        period=35.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash', '-c',
                    'source /home/rupesh/ultralytics_env/bin/activate && '
                    'python /home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/scripts/run_model.py'
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        slam,
        nav2,
        rviz,
        initial_pose,
        semantic_node,
    ])