#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        self.bridge = CvBridge()
        self.saved = False

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )

        self.get_logger().info("Waiting for image...")

    def callback(self, msg):
        if self.saved:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            save_path = '/home/rupesh/py_ws/test_image.png'
            cv2.imwrite(save_path, image)

            self.get_logger().info(f"Image saved at {save_path}")

            self.saved = True

            # Shutdown after saving one image
            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main():
    rclpy.init()
    node = ImageSaver()
    rclpy.spin(node)


if __name__ == '__main__':
    main()