#!/usr/bin/env python3
import rclpy
import math
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


class InitialPoseSetter(Node):
    def __init__(self):
        super().__init__('initial_pose_setter')

        # Declare parameters so they can be set from launch file
        self.declare_parameter('x',   0.0)
        self.declare_parameter('y',   0.0)
        self.declare_parameter('yaw', 0.0)

        x   = self.get_parameter('x').value
        y   = self.get_parameter('y').value
        yaw = self.get_parameter('yaw').value
        yaw=-0.5

        pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Give publisher time to connect
        time.sleep(2.0)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp    = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Standard covariance values
        msg.pose.covariance[0]  = 0.25   # x
        msg.pose.covariance[7]  = 0.25   # y
        msg.pose.covariance[35] = 0.06853  # yaw

        pub.publish(msg)
        self.get_logger().info(
            f"Initial pose set → x={x:.2f} y={y:.2f} yaw={yaw:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = InitialPoseSetter()
    rclpy.shutdown()


if __name__ == '__main__':
    main()