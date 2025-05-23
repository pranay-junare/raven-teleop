import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import zmq
import json
from utils import map_range

class TurtleBot4ZMQController(Node):
    def __init__(self):
        super().__init__('turtlebot4_zmq_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("ZMQ ROS2 Controller Initialized")

        # Setup ZeroMQ subscriber
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:5555") 
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Initialize twist to zero velocities
        self.current_twist = Twist()

        # Timers
        self.create_timer(0.1, self.poll_zmq)   # Check for new commands
        self.create_timer(0.1, self.publish_cmd)  # Publish current Twist at 10 Hz

    def publish_cmd(self):
        # Publish only if current_twist is updated from zero
        if self.current_twist.linear.x != 0 or self.current_twist.angular.z != 0:
            self.publisher_.publish(self.current_twist)

    def poll_zmq(self):
        try:
            if self.socket.poll(timeout=100):  # 10ms
                msg = self.socket.recv_string()
                # self.get_logger().info(f"ZMQ Received: {msg}")
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
                twist.angular.z = robot_yaw
                
                self.current_twist = twist
            else:
                # No data received, keep the robot stopped
                self.current_twist = Twist()

        except Exception as e:
            self.get_logger().warn(f"ZMQ error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot4ZMQController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
