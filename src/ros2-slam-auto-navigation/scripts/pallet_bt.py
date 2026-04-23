#!/usr/bin/env python3
import rclpy
import math
import time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from visualization_msgs.msg import MarkerArray, Marker
from nav2_msgs.action import NavigateToPose
from nav2_msgs.action import Spin
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from enum import Enum, auto


# ── Constants ─────────────────────────────────────────────────────────────
ROTATION_SPEED    = 0.025       # rad/s (~11 deg/s — slow enough to detect)
ROTATION_DURATION = 2 * math.pi / ROTATION_SPEED  # time for full 360°
STOP_DISTANCE     = 1.0        # meters in front of pallet
WAIT_AT_PALLET    = 6.0        # seconds to wait at each pallet
POSITION_TOLERANCE = 0.3       # meters — close enough to goal
CORRECTION_THRESHOLD = 0.5     # meters — re-navigate if pallet moved this much


# ── Behavior Tree States ──────────────────────────────────────────────────
class BTState(Enum):
    ROTATING            = auto()
    ROTATION_COMPLETE   = auto()
    SELECTING_PALLET    = auto()
    NAVIGATING          = auto()
    CORRECTING          = auto()
    WAITING_AT_PALLET   = auto()
    ALL_PALLETS_DONE    = auto()


class PalletBT(Node):
    def __init__(self):
        super().__init__('pallet_bt')

        # ── State ─────────────────────────────────────────────────────────
        self.state               = BTState.ROTATING
        self.robot_x             = 0.0
        self.robot_y             = 0.0
        self.robot_yaw           = 0.0

        # Pallet tracking
        self.known_pallets       = {}   # {obj_id: {x, y, label}}
        self.pallet_visit_queue  = []   # ordered list of obj_ids to visit
        self.visited_pallets     = set()
        self.current_pallet_id   = None
        self.current_goal_x      = None
        self.current_goal_y      = None

        # Timing
        self.rotation_start_time = None
        self.wait_start_time     = None
        self.navigation_start    = None

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            MarkerArray, '/tracked_objects',
            self.tracked_objects_callback, 10)
        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Action Clients ────────────────────────────────────────────────
        self.nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── State machine timer — runs at 10Hz ────────────────────────────
        self.create_timer(0.1, self.tick)

        self.get_logger().info("Pallet BT Ready — starting rotation...")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def tracked_objects_callback(self, msg):
        """
        Continuously update known pallet positions from object_tracker.
        This runs even during navigation — positions get corrected in real time.
        """
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                continue
            if 'pallet' not in marker.text.lower():
                continue

            obj_id = marker.id
            new_x  = marker.pose.position.x
            new_y  = marker.pose.position.y

            if obj_id not in self.known_pallets:
                # New pallet discovered
                self.known_pallets[obj_id] = {
                    'x': new_x,
                    'y': new_y,
                    'label': marker.text
                }
                self.get_logger().info(
                    f"Discovered pallet id={obj_id} "
                    f"at ({new_x:.2f}, {new_y:.2f})"
                )
                # Add to queue if not visited
                if obj_id not in self.visited_pallets and \
                   obj_id not in self.pallet_visit_queue:
                    self.pallet_visit_queue.append(obj_id)
            else:
                # Update existing pallet position
                old_x = self.known_pallets[obj_id]['x']
                old_y = self.known_pallets[obj_id]['y']
                self.known_pallets[obj_id]['x'] = new_x
                self.known_pallets[obj_id]['y'] = new_y

                # If currently navigating to this pallet and position shifted
                # significantly — trigger correction
                if self.current_pallet_id == obj_id and \
                   self.state == BTState.NAVIGATING:
                    shift = math.sqrt(
                        (new_x - old_x)**2 + (new_y - old_y)**2)
                    if shift > CORRECTION_THRESHOLD:
                        self.get_logger().info(
                            f"Pallet id={obj_id} moved {shift:.2f}m "
                            f"— triggering correction"
                        )
                        self.state = BTState.CORRECTING

    # ── Main tick — called at 10Hz ────────────────────────────────────────

    def tick(self):
        if self.state == BTState.ROTATING:
            self.tick_rotating()

        elif self.state == BTState.ROTATION_COMPLETE:
            self.tick_rotation_complete()

        elif self.state == BTState.SELECTING_PALLET:
            self.tick_selecting_pallet()

        elif self.state == BTState.NAVIGATING:
            self.tick_navigating()

        elif self.state == BTState.CORRECTING:
            self.tick_correcting()

        elif self.state == BTState.WAITING_AT_PALLET:
            self.tick_waiting()

        elif self.state == BTState.ALL_PALLETS_DONE:
            self.tick_done()

    # ── State handlers ────────────────────────────────────────────────────

    def tick_rotating(self):
        """Rotate slowly 360° to scan environment."""
        if self.rotation_start_time is None:
            self.rotation_start_time = self.get_clock().now()
            self.get_logger().info(
                f"Starting 360° rotation at {math.degrees(ROTATION_SPEED):.1f} deg/s "
                f"(will take {ROTATION_DURATION:.1f}s)"
            )

        # Publish rotation command
        twist         = Twist()
        twist.angular.z = ROTATION_SPEED
        self.cmd_vel_pub.publish(twist)

        # Check if full rotation complete
        elapsed = (self.get_clock().now() - \
                   self.rotation_start_time).nanoseconds / 1e9

        self.get_logger().info(
            f"Rotating... {elapsed:.1f}s / {ROTATION_DURATION:.1f}s | "
            f"Pallets found so far: {len(self.known_pallets)}",
            throttle_duration_sec=2.0
        )

        if elapsed >= ROTATION_DURATION:
            # Stop rotation
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(
                f"Rotation complete — found {len(self.known_pallets)} pallets"
            )
            self.state = BTState.ROTATION_COMPLETE

    def tick_rotation_complete(self):
        """Brief pause after rotation then start selecting pallets."""
        self.get_logger().info(
            f"Pallets in queue: {self.pallet_visit_queue}",
            once=True
        )

        if not self.known_pallets:
            self.get_logger().warn("No pallets found during rotation!")
            self.state = BTState.ALL_PALLETS_DONE
            return

        self.state = BTState.SELECTING_PALLET

    def tick_selecting_pallet(self):
        """Pick next pallet from queue."""
        # Remove already visited from queue
        self.pallet_visit_queue = [
            pid for pid in self.pallet_visit_queue
            if pid not in self.visited_pallets
        ]

        if not self.pallet_visit_queue:
            self.get_logger().info("All pallets visited!")
            self.state = BTState.ALL_PALLETS_DONE
            return

        # Select first in queue
        self.current_pallet_id = self.pallet_visit_queue[0]
        pallet = self.known_pallets.get(self.current_pallet_id)

        if pallet is None:
            self.get_logger().warn(
                f"Pallet id={self.current_pallet_id} not in known_pallets!")
            self.pallet_visit_queue.pop(0)
            return

        self.get_logger().info(
            f"Selected pallet id={self.current_pallet_id} "
            f"at ({pallet['x']:.2f}, {pallet['y']:.2f})"
        )

        self.state = BTState.NAVIGATING
        self.send_nav_goal_to_pallet(self.current_pallet_id)

    def tick_navigating(self):
        """Monitor navigation — position correction handled in callback."""
        elapsed = (self.get_clock().now() - \
                   self.navigation_start).nanoseconds / 1e9 \
                  if self.navigation_start else 0

        dist = self.distance_to_current_goal()

        self.get_logger().info(
            f"Navigating to pallet id={self.current_pallet_id} | "
            f"distance to goal: {dist:.2f}m | "
            f"elapsed: {elapsed:.1f}s",
            throttle_duration_sec=2.0
        )

    def tick_correcting(self):
        """Pallet position updated — resend corrected goal."""
        self.get_logger().info(
            f"Correcting goal for pallet id={self.current_pallet_id}")
        self.send_nav_goal_to_pallet(self.current_pallet_id)
        self.state = BTState.NAVIGATING

    def tick_waiting(self):
        """Wait 6 seconds at pallet then move to next."""
        if self.wait_start_time is None:
            self.wait_start_time = self.get_clock().now()
            self.get_logger().info(
                f"Arrived at pallet id={self.current_pallet_id}! "
                f"Waiting {WAIT_AT_PALLET}s..."
            )

        elapsed   = (self.get_clock().now() - \
                     self.wait_start_time).nanoseconds / 1e9
        remaining = WAIT_AT_PALLET - elapsed

        self.get_logger().info(
            f"Waiting at pallet id={self.current_pallet_id} — "
            f"{remaining:.1f}s remaining",
            throttle_duration_sec=1.0
        )

        if elapsed >= WAIT_AT_PALLET:
            # Mark visited and move to next
            self.visited_pallets.add(self.current_pallet_id)
            self.pallet_visit_queue.remove(self.current_pallet_id)
            self.get_logger().info(
                f"Done waiting at pallet id={self.current_pallet_id} — "
                f"moving to next"
            )
            self.current_pallet_id = None
            self.wait_start_time   = None
            self.state             = BTState.SELECTING_PALLET

    def tick_done(self):
        self.get_logger().info(
            f"All done! Visited {len(self.visited_pallets)} pallets: "
            f"{self.visited_pallets}",
            once=True
        )

    # ── Navigation helpers ────────────────────────────────────────────────

    def compute_goal_in_front_of_pallet(self, pallet_x, pallet_y):
        """
        Compute goal pose STOP_DISTANCE meters in front of pallet,
        facing the pallet.
        """
        dx   = pallet_x - self.robot_x
        dy   = pallet_y - self.robot_y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.01:
            return pallet_x, pallet_y, 0.0

        # Unit vector toward pallet
        nx = dx / dist
        ny = dy / dist

        # Goal is STOP_DISTANCE before the pallet
        goal_x = pallet_x - nx * STOP_DISTANCE
        goal_y = pallet_y - ny * STOP_DISTANCE

        # Face toward pallet
        yaw = math.atan2(dy, dx)

        return goal_x, goal_y, yaw

    def send_nav_goal_to_pallet(self, pallet_id):
        """Send Nav2 goal to position 1m in front of given pallet."""
        pallet = self.known_pallets.get(pallet_id)
        if pallet is None:
            self.get_logger().error(f"Pallet id={pallet_id} not found!")
            return

        goal_x, goal_y, yaw = self.compute_goal_in_front_of_pallet(
            pallet['x'], pallet['y'])

        self.current_goal_x  = goal_x
        self.current_goal_y  = goal_y
        self.navigation_start = self.get_clock().now()

        self.get_logger().info(
            f"Sending nav goal for pallet id={pallet_id} → "
            f"goal=({goal_x:.2f}, {goal_y:.2f}) "
            f"yaw={math.degrees(yaw):.1f}°"
        )

        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        pose                      = PoseStamped()
        pose.header.frame_id      = 'map'
        pose.header.stamp         = self.get_clock().now().to_msg()
        pose.pose.position.x      = goal_x
        pose.pose.position.y      = goal_y
        pose.pose.position.z      = 0.0
        pose.pose.orientation.z   = math.sin(yaw / 2.0)
        pose.pose.orientation.w   = math.cos(yaw / 2.0)

        goal      = NavigateToPose.Goal()
        goal.pose = pose

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self.nav_response_callback)

    def nav_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected — retrying...")
            self.state = BTState.CORRECTING
            return
        self.get_logger().info("Nav2 goal accepted!")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        self.get_logger().info(
            f"Nav2 reached goal for pallet id={self.current_pallet_id}")

        # Verify we're actually close enough — correct if not
        dist = self.distance_to_current_goal()
        if dist > POSITION_TOLERANCE:
            self.get_logger().warn(
                f"Too far from goal ({dist:.2f}m > {POSITION_TOLERANCE}m) "
                f"— correcting"
            )
            self.state = BTState.CORRECTING
        else:
            self.state           = BTState.WAITING_AT_PALLET
            self.wait_start_time = None

    def distance_to_current_goal(self):
        if self.current_goal_x is None:
            return float('inf')
        return math.sqrt(
            (self.robot_x - self.current_goal_x)**2 +
            (self.robot_y - self.current_goal_y)**2
        )


def main(args=None):
    rclpy.init(args=args)
    node = PalletBT()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()