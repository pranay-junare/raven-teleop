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

                action = data.get("action")
                speed = float(data.get("speed", 0)) # default speed
                print("Action is: ", action, " Speed is: ", speed)

                twist = Twist()
                if action == "forward":
                    if speed < 10 and speed > -10:
                        speed = 0.0
                    else:
                        speed = map_range(speed, -100.0, 100.0, -4, 4) 
                    twist.linear.x = speed
                elif action == "yaw":
                    speed = -speed 
                    if speed < 10 and speed > -10:  
                        speed = 0.0
                    else:
                        speed = map_range(speed, -180.0, 180.0, -4.0, 4.0)
                    twist.angular.z = speed
                elif action == "stop":
                    twist = Twist()  # zero velocities
                else:
                    self.get_logger().warn(f"Unknown action: {action}")
                    return

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
