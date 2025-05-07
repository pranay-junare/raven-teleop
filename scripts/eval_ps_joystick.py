import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame
import time

class PSControllerPublisher(Node):
    def __init__(self):
        super().__init__('ps_controller_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('PS Controller ROS2 publisher initialized.')

        # Initialize pygame joystick
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            self.get_logger().error("No joystick detected.")
            exit()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.get_logger().info(f"Joystick connected: {self.joystick.get_name()}")

        # Main timer for polling joystick
        self.timer = self.create_timer(0.1, self.publish_twist)

    def publish_twist(self):
        pygame.event.pump()
        axis_x = self.joystick.get_axis(0)  # Left stick horizontal
        axis_y = self.joystick.get_axis(1)  # Left stick vertical

        linear = -axis_y if abs(axis_y) > 0.2 else 0.0
        angular = -axis_x if abs(axis_x) > 0.2 else 0.0

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular

        self.publisher_.publish(twist)
        self.get_logger().info(f"Publishing: linear={linear:.2f}, angular={angular:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PSControllerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
