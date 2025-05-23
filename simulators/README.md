# raven-teleop

Client Interfaces with the RAVEN-TELEOP

## ROS2 and Ignition Gazebo:
- Robot Tested: Tugbot
    - ```$ ros2 run ros_gz_bridge parameter_bridge /model/tugbot/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist ```
    - ```$ ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/model/tugbot/cmd_vel ```

- Robot Tested: Turtlebot
    - `$ ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py`
    - `$ ros2 param set /motion_control sagfety_override full`
    - `$ python scripts/landmark_detection.py`
    - `$ python ./simulators/ros2/turtlebot_control.py`

## ROS and RealRobot:
- Robot Tested: Turtlebot3
    - `$ roslaunch turtlebot3_bringup turtlebot3_robot.launch`
    - `$ python scripts/landmark_detection.py`
    - `$ python ./simulators/ros_realrobot/turtlebot_control.py`

## Kineval
- Robot Tested: MR2
    - Install ZMQ `$ npm install ZMQ `
    - run the subscriber.js file in a node server `$ node subscriber.js `
      
## PyBullet
- Robot Tested: Unitree's A1
    - Clone the repository `$ git@github.com:wupanhao/quadruped_simulation.git`
    - Replace the `JoyStickInterface.py` with `a1_control.py`
    - Run the simulation using `python3 run_robot.py`
