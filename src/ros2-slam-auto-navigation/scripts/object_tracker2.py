#!/usr/bin/env python3
import rclpy
import math
import json
import os
import numpy as np
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker


MAP_SAVE_PATH   = '/home/rupesh/ros2_ws/semantic_objects.json'
MATCH_THRESHOLD = 1.5

PROCESS_NOISE_STATIC  = 0.001  # ← reduced — static objects barely move
PROCESS_NOISE_DYNAMIC = 0.5
MEASUREMENT_NOISE     = 0.3
INITIAL_UNCERTAINTY   = 2.0

# Cap uncertainty so it doesn't grow unbounded when object not seen
MAX_UNCERTAINTY = 4.0          # ← new — caps at 2m std dev

STATIC_CLASSES  = ["pallet", "box with barcode", "industrial shelving system"]
DYNAMIC_CLASSES = ["person", "forklift"]


class KalmanObject:
    def __init__(self, x, y, label):
        self.label      = label
        self.count      = 1
        self.first_seen = None
        self.last_seen  = None

        self.x_est = np.array([x, y], dtype=float)
        self.P     = np.eye(2) * INITIAL_UNCERTAINTY

        q       = PROCESS_NOISE_STATIC if label in STATIC_CLASSES \
                  else PROCESS_NOISE_DYNAMIC
        self.Q  = np.eye(2) * q
        self.R  = np.eye(2) * MEASUREMENT_NOISE
        self.H  = np.eye(2)

    def predict(self):
        """
        Grow uncertainty slightly.
        Capped at MAX_UNCERTAINTY so it never explodes.
        """
        self.P = self.P + self.Q

        # ← Cap uncertainty — static objects don't move
        self.P = np.minimum(self.P, np.eye(2) * MAX_UNCERTAINTY)

    def update(self, measured_x, measured_y):
        z       = np.array([measured_x, measured_y])
        y_innov = z - self.H @ self.x_est
        S       = self.H @ self.P @ self.H.T + self.R

        # Safe inversion — fallback to identity if singular
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Singular S matrix — skipping update")
            return

        K          = self.P @ self.H.T @ S_inv
        self.x_est = self.x_est + K @ y_innov
        self.P     = (np.eye(2) - K @ self.H) @ self.P

        # Ensure P stays symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        self.P = np.maximum(self.P, np.eye(2) * 1e-6)

        self.count += 1

    @property
    def x(self):
        return float(self.x_est[0])

    @property
    def y(self):
        return float(self.x_est[1])

    @property
    def uncertainty(self):
        return float(np.trace(self.P))

    @property
    def std_dev(self):
        # Clamp to avoid sqrt of negative from numerical errors
        sx = float(np.sqrt(max(self.P[0, 0], 0.0)))
        sy = float(np.sqrt(max(self.P[1, 1], 0.0)))
        return sx, sy

    @property
    def is_confident(self):
        return self.uncertainty < 0.1

    def to_dict(self):
        return {
            'label':      self.label,
            'x':          self.x,
            'y':          self.y,
            'P':          self.P.tolist(),
            'count':      self.count,
            'first_seen': self.first_seen,
            'last_seen':  self.last_seen,
        }

    @classmethod
    def from_dict(cls, d):
        obj            = cls(d['x'], d['y'], d['label'])
        obj.P          = np.array(d['P'])
        obj.count      = d['count']
        obj.first_seen = d.get('first_seen')
        obj.last_seen  = d.get('last_seen')
        return obj


class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')

        self.objects        = {}
        self.next_object_id = 0

        self.load_objects()

        self.create_subscription(
            MarkerArray, '/semantic_markers',
            self.marker_callback, 10)

        self.tracked_pub = self.create_publisher(
            MarkerArray, '/tracked_objects', 10)

        self.create_timer(30.0, self.save_objects)
        self.create_timer(1.0,  self.predict_all)

        # ← Publish tracked objects at fixed rate even if no markers arrive
        self.create_timer(0.5,  self.publish_tracked_objects)

        self.get_logger().info(
            f"Object Tracker Ready — "
            f"loaded {len(self.objects)} objects from disk"
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def save_objects(self):
        try:
            data = {str(k): v.to_dict() for k, v in self.objects.items()}
            with open(MAP_SAVE_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f"Saved {len(self.objects)} objects")
        except Exception as e:
            self.get_logger().error(f"Save failed: {e}")

    def load_objects(self):
        if not os.path.exists(MAP_SAVE_PATH):
            self.get_logger().info("No saved objects — starting fresh")
            return
        try:
            with open(MAP_SAVE_PATH) as f:
                data = json.load(f)
            self.objects = {
                int(k): KalmanObject.from_dict(v)
                for k, v in data.items()
            }
            if self.objects:
                self.next_object_id = max(self.objects.keys()) + 1
            self.get_logger().info(f"Loaded {len(self.objects)} objects")
        except Exception as e:
            self.get_logger().error(f"Load failed: {e}")

    # ── Kalman predict ────────────────────────────────────────────────────

    def predict_all(self):
        for obj in self.objects.values():
            obj.predict()

    # ── Association ───────────────────────────────────────────────────────

    def find_match(self, label, wx, wy):
        best_id   = None
        best_dist = float('inf')

        for obj_id, obj in self.objects.items():
            if obj.label != label:
                continue

            # Hard Euclidean limit first — fast rejection
            euclidean = math.sqrt((wx - obj.x)**2 + (wy - obj.y)**2)
            if euclidean > MATCH_THRESHOLD:
                continue

            # Mahalanobis distance for ranking
            diff = np.array([wx - obj.x, wy - obj.y])
            try:
                S     = obj.P + np.eye(2) * MEASUREMENT_NOISE
                S_inv = np.linalg.inv(S)
                dist  = float(diff.T @ S_inv @ diff)
            except np.linalg.LinAlgError:
                dist  = euclidean  # fallback

            if dist < best_dist:
                best_dist = dist
                best_id   = obj_id

        return best_id

    # ── Main callback ─────────────────────────────────────────────────────

    def marker_callback(self, msg):
        # ← Handle empty marker array gracefully
        if not msg.markers:
            self.get_logger().debug("Empty marker array received")
            return

        now = self.get_clock().now().nanoseconds / 1e9

        for marker in msg.markers:
            # ← Check action first before anything else
            if marker.action == Marker.DELETEALL:
                continue

            # ← Safe text parsing — skip if empty
            if not marker.text:
                continue
            parts = marker.text.split()
            if not parts:
                continue
            label = parts[0]

            # ← Validate position — skip NaN/inf
            wx = marker.pose.position.x
            wy = marker.pose.position.y
            if not (math.isfinite(wx) and math.isfinite(wy)):
                self.get_logger().warn(f"Invalid position for {label} — skipping")
                continue

            match_id = self.find_match(label, wx, wy)

            if match_id is not None:
                obj           = self.objects[match_id]
                obj.update(wx, wy)
                obj.last_seen = now

                std_x, std_y = obj.std_dev
                self.get_logger().debug(
                    f"Updated {label} id={match_id} "
                    f"pos=({obj.x:.2f},{obj.y:.2f}) "
                    f"σ=({std_x:.3f},{std_y:.3f}) "
                    f"confident={obj.is_confident}"
                )
            else:
                obj_id              = self.next_object_id
                self.next_object_id += 1

                new_obj             = KalmanObject(wx, wy, label)
                new_obj.first_seen  = now
                new_obj.last_seen   = now
                self.objects[obj_id] = new_obj

                self.get_logger().info(
                    f"New object: {label} id={obj_id} "
                    f"at ({wx:.2f},{wy:.2f}) "
                    f"uncertainty={new_obj.uncertainty:.3f}"
                )

    # ── Publish at fixed rate ─────────────────────────────────────────────

    def publish_tracked_objects(self):
        """
        Publishes at 2Hz regardless of whether markers arrived.
        This keeps RViz updated and lets pallet_bt.py always
        have fresh data even when objects are out of camera view.
        """
        marker_array = MarkerArray()

        delete_all                 = Marker()
        delete_all.header.frame_id = 'map'
        delete_all.header.stamp    = self.get_clock().now().to_msg()
        delete_all.action          = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for obj_id, obj in self.objects.items():
            if obj.count < 2:
                continue

            std_x, std_y = obj.std_dev

            # ── Text label ────────────────────────────────────────────────
            m                    = Marker()
            m.header.frame_id    = 'map'
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = 'tracked'
            m.id                 = obj_id
            m.type               = Marker.TEXT_VIEW_FACING
            m.action             = Marker.ADD
            m.text               = (
                f"{obj.label} [{obj_id}]\n"
                f"n={obj.count} σ=({std_x:.2f},{std_y:.2f})m"
            )
            m.scale.z            = 0.3
            m.pose.position.x    = obj.x
            m.pose.position.y    = obj.y
            m.pose.position.z    = 0.6
            m.pose.orientation.w = 1.0

            # Color by confidence
            if obj.is_confident:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            elif obj.uncertainty < 0.5:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            m.color.a = 1.0
            marker_array.markers.append(m)

            # ── Uncertainty ellipse ───────────────────────────────────────
            ellipse                    = Marker()
            ellipse.header.frame_id    = 'map'
            ellipse.header.stamp       = self.get_clock().now().to_msg()
            ellipse.ns                 = 'uncertainty'
            ellipse.id                 = obj_id + 10000
            ellipse.type               = Marker.CYLINDER
            ellipse.action             = Marker.ADD
            ellipse.pose.position.x    = obj.x
            ellipse.pose.position.y    = obj.y
            ellipse.pose.position.z    = 0.0
            ellipse.pose.orientation.w = 1.0
            ellipse.scale.x            = max(2 * std_x, 0.05)
            ellipse.scale.y            = max(2 * std_y, 0.05)
            ellipse.scale.z            = 0.02
            ellipse.color.r            = m.color.r
            ellipse.color.g            = m.color.g
            ellipse.color.b            = m.color.b
            ellipse.color.a            = 0.25
            marker_array.markers.append(ellipse)

        self.tracked_pub.publish(marker_array)

        confirmed = sum(1 for o in self.objects.values() if o.is_confident)
        self.get_logger().info(
            f"Tracking {len(self.objects)} objects | "
            f"confirmed: {confirmed}",
            throttle_duration_sec=5.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()