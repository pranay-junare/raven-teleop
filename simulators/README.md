# raven-teleop

## ROS2 and Ignition Gazebo:
- Robot Tested: Tugbot
- ```$ ros2 run ros_gz_bridge parameter_bridge /model/tugbot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist ```
- ```$ ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/model/tugbot/cmd_vel ```
- Robot Tested: Turtlebot
- `$ ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py`
- `$ ros2 param set /motion_control sagfety_override full`
- `$ python scripts/landmark_detection.py`
- `$ python ./simulators/ros2/turtlebot_control.py`

## Kineval



## Mujoco



## Genesis
