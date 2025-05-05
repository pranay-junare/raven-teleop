import rospy
from geometry_msgs.msg import Twist
import zmq
import json
from utils import map_range

class TurtleBot4ZMQController:
    def __init__(self):
        # ROS1 node initialization
        rospy.init_node('turtlebot4_zmq_controller', anonymous=True)
        self.publisher_ = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.loginfo("ZMQ ROS1 Controller Initialized")

        # Setup ZeroMQ subscriber
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:5555") 
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Initialize twist to zero velocities
        self.current_twist = Twist()

        # Timers
        self.poll_timer = rospy.Timer(rospy.Duration(0.1), self.poll_zmq)  # Check for new commands
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.publish_cmd)  # Publish current Twist at 10 Hz

    def publish_cmd(self, event):
        # Publish only if current_twist is updated from zero
        self.publisher_.publish(self.current_twist)

    def poll_zmq(self, event):
        try:
            if self.socket.poll(timeout=100): 
                msg = self.socket.recv_string()
                data = json.loads(msg)
                print(f"Received ZMQ message: {data}")
                robot_speed = float(data.get("robot_speed", 0)) # default speed
                robot_yaw = float(data.get("robot_yaw", 0)) # default speed
        
                twist = Twist()
                
                # Linear speed
                if robot_speed < 10 and robot_speed > -10:
                    robot_speed = 0.0
                else:
                    robot_speed = map_range(robot_speed, -100.0, 100.0, -0.5, 0.5) 
                twist.linear.x = robot_speed
                
                # Yaw speed
                robot_yaw = -robot_yaw 
                if robot_yaw < 5 and robot_yaw > -5:  
                    robot_yaw = 0.0
                else:
                    robot_yaw = map_range(robot_yaw, -180.0, 180.0, -2.0, 2.0)
                    # if robot_yaw >5:
                    #     robot_yaw = 0.5
                    # elif robot_yaw < -5:
                    #     robot_yaw = -0.5

                twist.angular.z = robot_yaw
                self.current_twist = twist
                rospy.loginfo(f"Received ZMQ command: {robot_speed}, {robot_yaw}")

        except Exception as e:
            rospy.logwarn(f"ZMQ error: {e}")

if __name__ == '__main__':
    try:
        controller = TurtleBot4ZMQController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
