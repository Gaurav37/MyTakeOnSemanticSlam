#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray

import numpy as np
import tf2_ros
import math
import tf_transformations



class SemanticMapper(Node):
    def __init__(self):
        super().__init__('semantic_mapper')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.marker_sub = self.create_subscription(
            MarkerArray, '/semantic_markers', self.marker_callback, 10)

        # Publisher
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/semantic_map', 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Map params
        self.resolution = 0.05
        self.width = 400
        self.height = 400
        self.origin_x = -10.0
        self.origin_y = -10.0

        # Map storage
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.semantic_map = {} 

        self.get_logger().info("Semantic Mapper Node Ready")
    

    # -----------------------------
    # LiDAR → occupancy
    # -----------------------------
    def scan_callback(self, msg):
        angle = msg.angle_min
        for r in msg.ranges:
            if np.isfinite(r):
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                world = self.transform_point(x, y)
                if world is None:
                    angle += msg.angle_increment
                    continue
                wx, wy = world
                mx = int((wx - self.origin_x) / self.resolution)
                my = int((wy - self.origin_y) / self.resolution)
                if 0 <= mx < self.width and 0 <= my < self.height:
                    self.grid[my, mx] = 100
            angle += msg.angle_increment
    # -----------------------------
    # SAM → semantics
    # -----------------------------
    def marker_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9

        for marker in msg.markers:
            # Markers are already in map frame from run_model.py — use directly
            wx = marker.pose.position.x
            wy = marker.pose.position.y

            mx = int((wx - self.origin_x) / self.resolution)
            my = int((wy - self.origin_y) / self.resolution)

            if not (0 <= mx < self.width and 0 <= my < self.height):
                continue

            label = marker.text.split()[0]  # strip confidence if present

            key = (mx, my)
            if key in self.semantic_map:
                existing = self.semantic_map[key]
                if existing['label'] == label:
                    existing['count'] = min(existing['count'] + 1, 10)
                    existing['time']  = now
                else:
                    if existing['count'] <= 1:
                        self.semantic_map[key] = {'label': label, 'count': 1, 'time': now}
                    else:
                        existing['count'] -= 1
            else:
                self.semantic_map[key] = {'label': label, 'count': 1, 'time': now}

        self.forget_old_objects(now)
        self.publish_map()

    def transform_point(self, x, y):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time()
            )

            tx = trans.transform.translation.x
            ty = trans.transform.translation.y

            q = trans.transform.rotation
            yaw = tf_transformations.euler_from_quaternion(
                [q.x, q.y, q.z, q.w]
            )[2]

            mx = x * math.cos(yaw) - y * math.sin(yaw) + tx
            my = x * math.sin(yaw) + y * math.cos(yaw) + ty

            return mx, my
        except:
            return None

    # -----------------------------
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y

        data = np.zeros((self.height, self.width), dtype=np.int8)

        # occupancy
        data[self.grid == 100] = 100

        # semantic overlay (soft encoding)
        for (mx, my), label in self.semantic_map.items():
            if 0 <= mx < self.width and 0 <= my < self.height:
                data[my, mx] = 50

        msg.data = data.flatten().tolist()

        self.map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapper()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()