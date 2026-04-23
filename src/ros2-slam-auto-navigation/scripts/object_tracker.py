#!/usr/bin/env python3
import rclpy
import math
import json
import os
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker

MAP_SAVE_PATH    = '/home/rupesh/ros2_ws/semantic_objects.json'
MATCH_THRESHOLD  = 1.0   # meters — same object if closer than this

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')

        # {obj_id: {label, x, y, count, last_seen, first_seen}}
        self.objects        = {}
        self.next_object_id = 0

        # Load previously saved objects
        self.load_objects()

        # Subscribe to raw semantic markers from run_model.py
        self.create_subscription(
            MarkerArray, '/semantic_markers',
            self.marker_callback, 10)

        # Publish tracked objects with stable IDs
        self.tracked_pub = self.create_publisher(
            MarkerArray, '/tracked_objects', 10)

        # Save to disk every 30 seconds
        self.create_timer(30.0, self.save_objects)

        self.get_logger().info(
            f"Object Tracker Ready — "
            f"loaded {len(self.objects)} objects from disk"
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save_objects(self):
        try:
            with open(MAP_SAVE_PATH, 'w') as f:
                json.dump(self.objects, f, indent=2)
            self.get_logger().info(
                f"Saved {len(self.objects)} objects to {MAP_SAVE_PATH}")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")

    def load_objects(self):
        if not os.path.exists(MAP_SAVE_PATH):
            self.get_logger().info("No saved objects found — starting fresh")
            return
        try:
            with open(MAP_SAVE_PATH) as f:
                data = json.load(f)
            # JSON keys are strings — convert back to int
            self.objects        = {int(k): v for k, v in data.items()}
            if self.objects:
                self.next_object_id = max(self.objects.keys()) + 1
            self.get_logger().info(
                f"Loaded {len(self.objects)} objects from disk")
        except Exception as e:
            self.get_logger().error(f"Load failed: {e}")

    # ── Association ───────────────────────────────────────────────────────

    def find_match(self, label, wx, wy):
        """
        Find closest existing object with same label within threshold.
        Returns obj_id or None.
        """
        best_id   = None
        best_dist = float('inf')

        for obj_id, obj in self.objects.items():
            if obj['label'] != label:
                continue
            dist = math.sqrt(
                (obj['x'] - wx) ** 2 +
                (obj['y'] - wy) ** 2
            )
            if dist < MATCH_THRESHOLD and dist < best_dist:
                best_dist = dist
                best_id   = obj_id

        return best_id

    # ── Main callback ─────────────────────────────────────────────────────

    def marker_callback(self, msg):
        now = self.get_clock().now().nanoseconds / 1e9

        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue

            label = marker.text.split()[0]
            wx    = marker.pose.position.x
            wy    = marker.pose.position.y

            match_id = self.find_match(label, wx, wy)

            if match_id is not None:
                # ── Known object — update position ────────────────────────
                obj = self.objects[match_id]

                # Alpha decreases as count grows — position stabilizes over time
                alpha    = max(0.05, 0.3 / (1 + obj['count'] * 0.1))
                obj['x'] = (1 - alpha) * obj['x'] + alpha * wx
                obj['y'] = (1 - alpha) * obj['y'] + alpha * wy
                obj['count']     += 1
                obj['last_seen']  = now

                self.get_logger().debug(
                    f"Updated {label} id={match_id} "
                    f"count={obj['count']} "
                    f"pos=({obj['x']:.2f}, {obj['y']:.2f})"
                )

            else:
                # ── New object — register it ──────────────────────────────
                obj_id              = self.next_object_id
                self.next_object_id += 1

                self.objects[obj_id] = {
                    'label':      label,
                    'x':          wx,
                    'y':          wy,
                    'count':      1,
                    'last_seen':  now,
                    'first_seen': now,
                }
                self.get_logger().info(
                    f"New object: {label} id={obj_id} "
                    f"at ({wx:.2f}, {wy:.2f})"
                )

        self.publish_tracked_objects()

    # ── Publish ───────────────────────────────────────────────────────────

    def publish_tracked_objects(self):
        marker_array = MarkerArray()

        # Clear previous markers in RViz
        delete_all              = Marker()
        delete_all.header.frame_id = 'map'
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.action       = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for obj_id, obj in self.objects.items():
            # Only show objects seen at least twice — filters noise
            if obj['count'] < 2:
                continue

            m                    = Marker()
            m.header.frame_id    = 'map'
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = 'tracked'
            m.id                 = obj_id
            m.type               = Marker.TEXT_VIEW_FACING
            m.action             = Marker.ADD
            m.text               = f"{obj['label']} [{obj_id}]"
            m.scale.z            = 0.4
            m.color.a            = 1.0
            m.color.r            = 1.0
            m.color.g            = 1.0
            m.color.b            = 0.0
            m.pose.position.x    = obj['x']
            m.pose.position.y    = obj['y']
            m.pose.position.z    = 0.5
            m.pose.orientation.w = 1.0
            marker_array.markers.append(m)

        self.tracked_pub.publish(marker_array)
        self.get_logger().info(
            f"Tracking {len(self.objects)} objects total "
            f"({sum(1 for o in self.objects.values() if o['count'] >= 2)} confirmed)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()