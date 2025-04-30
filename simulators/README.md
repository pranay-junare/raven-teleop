# raven-teleop

## ROS2 and Ignition Gazebo:
- Robot Tested: Tugbot
- ```$ ros2 run ros_gz_bridge parameter_bridge /model/tugbot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist ```
- ```$ ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/model/tugbot/cmd_vel ```
- Robot Tested: Turtlebot
- ```$ python3 ./simulators/ros2/turtlebot_control.py```

## Kineval



## Mujoco



## Genesis
