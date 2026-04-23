#!/bin/bash
# save as ~/ros2_ws/launch_all.sh

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

SLAM_CONFIG="/home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/config/mapper_params_online_async.yaml"
WORLD_FILE="/home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/worlds/warehouse_new.world"

# Window 1: Gazebo
xterm -T "Gazebo" -geometry 120x30 -e "
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  ros2 launch ros2_slam_auto_navigation launch_sim.launch.py world_file:=$WORLD_FILE
" &

# Window 2: SLAM Toolbox
xterm -T "SLAM Toolbox" -geometry 120x30 -e "
  sleep 10 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  ros2 launch slam_toolbox online_async_launch.py slam_params_file:=$SLAM_CONFIG use_sim_time:=true
" &

# Window 3: Nav2
xterm -T "Nav2" -geometry 120x30 -e "
  sleep 20 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True
" &

# Window 4: RViz
xterm -T "RViz" -geometry 120x30 -e "
  sleep 25 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  #   ros2 run rviz2 rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz
  ros2 run rviz2 rviz2 -d /home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/rviz_view/custom_view.rviz
" &

# Window 5: Initial Pose
xterm -T "Initial Pose" -geometry 80x10 -e "
  sleep 30 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  python3 /home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/scripts/initial_pose_setter.py
" &

# Window 6: Semantic SAM
xterm -T "Semantic SAM" -geometry 120x30 -e "
  sleep 35 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  source /home/rupesh/ultralytics_env/bin/activate &&
  python /home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/scripts/run_model.py
" &

# Window 7: Object Tracker
xterm -T "Object Tracker" -geometry 120x30 -e "
  sleep 60 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  python /home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/scripts/object_tracker2.py
" &
# behavior tree
xterm -T "Orchestration" -geometry 120x30 -e "
  sleep 70 &&
  source /opt/ros/humble/setup.bash &&
  source ~/ros2_ws/install/setup.bash &&
  python /home/rupesh/ros2_ws/src/ros2-slam-auto-navigation/scripts/pallet_bt.py
" &

echo "All windows launching... waiting for everything to start"
wait