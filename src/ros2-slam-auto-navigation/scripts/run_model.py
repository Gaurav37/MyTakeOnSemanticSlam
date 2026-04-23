#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from rclpy.time import Time


GROUNDING_DINO_CONFIG = "/home/rupesh/py_ws/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CKPT   = "/home/rupesh/py_ws/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
SAM_CKPT              = "/home/rupesh/py_ws/Grounded-Segment-Anything/sam_vit_b_01ec64.pth"
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = [
    "man",
    "wood pallet",
    "cardboard shipping box",
    "cabinets",
    "pallet-jack"
]

BOX_THRESHOLD  = 0.3
TEXT_THRESHOLD = 0.3

CLASS_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255)
]


class GroundedSAMNode(Node):
    def __init__(self):
        super().__init__('grounded_sam_node')

        self.bridge = CvBridge()
        self.latest_scan = None

        # self.marker_id = 0
        marker_array = MarkerArray()

        # First add a DELETE_ALL marker to clear previous frame's markers
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # Reset ID counter each frame
        self.marker_id = 0

        # Camera intrinsics (adjust if needed)
        # Replace hardcoded values in run_model.py
        self.fx = 528.433756558705
        self.fy = 528.433756558705
        self.cx = 320.5
        self.cy = 240.5

        # Models
        self.get_logger().info("Loading GroundingDINO...")
        self.dino = Model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CKPT
        )

        self.get_logger().info("Loading SAM...")
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
        sam.to(DEVICE)
        self.sam = SamPredictor(sam)

        # TF
        # self.tf_buffer = Buffer()
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # Publishers
        self.det_pub = self.create_publisher(Detection2DArray, '/semantic_detections', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_markers', 10)
        self.annotated_pub = self.create_publisher(Image, '/semantic_image', 10)

        self.get_logger().info("Semantic LiDAR + SAM Node Ready")
    
    def transform_to_map(self, x, y, stamp):
        """Transform a point from base_link to map frame using message timestamp."""
        try:
            point = PointStamped()
            point.header.frame_id = 'base_link'
            point.header.stamp    = rclpy.time.Time().to_msg()   # ← use original message time not now()
            point.point.x = float(x)
            point.point.y = float(y)
            point.point.z = 0.0

            transformed = self.tf_buffer.transform(
                point,
                'map',
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return transformed.point.x, transformed.point.y, transformed.point.z

        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}", throttle_duration_sec=2.0)
            return None
    # -----------------------
    # LiDAR callback
    # -----------------------
    def lidar_callback(self, msg):
        self.latest_scan = msg

    # -----------------------
    # Project LiDAR into image
    # -----------------------
    def project_lidar(self, scan, width, height):
        if scan is None:
            return []

        points = []

        angle = scan.angle_min

        for r in scan.ranges:
            if r > scan.range_min and r < scan.range_max:

                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = 0.0

                # simple pinhole projection assumption (camera aligned front)
                if x <= 0:
                    angle += scan.angle_increment
                    continue

                u = int(self.fx * (y / x) + self.cx)
                v = int(self.fy * (z / x) + self.cy)

                if 0 <= u < width and 0 <= v < height:
                    points.append((u, v, x, y, z))

            angle += scan.angle_increment

        return points

    # -----------------------
    # Main callback
    # -----------------------
    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = image.shape[:2]

        # -----------------------
        # GroundingDINO
        # -----------------------
        detections = self.dino.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        if len(detections) == 0:
            out = self.bridge.cv2_to_imgmsg(image, 'bgr8')
            out.header = msg.header
            self.annotated_pub.publish(out)
            return

        # -----------------------
        # SAM segmentation
        # -----------------------
        self.sam.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        boxes = torch.tensor(detections.xyxy, device=DEVICE, dtype=torch.float32)
        boxes = self.sam.transform.apply_boxes_torch(boxes, image.shape[:2])

        with torch.no_grad():
            masks, _, _ = self.sam.predict_torch(
                boxes=boxes,
                point_coords=None,
                point_labels=None,
                multimask_output=False
            )

        # -----------------------
        # LiDAR projection
        # -----------------------
        lidar_points = self.project_lidar(self.latest_scan, w, h)

        lidar_map = {}
        for u, v, x, y, z in lidar_points:
            lidar_map[(u, v)] = (x, y, z)

        # -----------------------
        # Outputs
        # -----------------------
        annotated = image.copy()
        det_array = Detection2DArray()
        marker_array = MarkerArray()
        
        det_array.header = msg.header

        # ← Add this block — clears all previous markers before publishing new ones
        delete_all = Marker()
        delete_all.header.frame_id = 'map'
        delete_all.header.stamp = self.get_clock().now().to_msg()
        delete_all.ns = "semantic"
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        # Reset ID each frame so markers replace instead of accumulate
        self.marker_id = 0
        det_array.header = msg.header

        # -----------------------
        # For each object
        # -----------------------
        for i, (box, class_id, conf) in enumerate(zip(
            detections.xyxy,
            detections.class_id,
            detections.confidence
        )):

            if class_id is None:
                continue

            label = CLASSES[class_id]
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

            mask = masks[i][0].cpu().numpy().astype(bool)

            pts = []

            for v in range(h):
                for u in range(w):
                    if mask[v, u] and (u, v) in lidar_map:
                        pts.append(lidar_map[(u, v)])

            if len(pts) < 10:
                continue

            pts = np.array(pts)
            x, y, z = np.median(pts, axis=0)

            # -----------------------
            # visualization
            # -----------------------
            # -----------------------
            # visualization — mask contours instead of boxes
            # -----------------------
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Semi-transparent filled mask
                mask_colored = np.zeros_like(image)
                cv2.fillPoly(mask_colored, contours, color)
                annotated = cv2.addWeighted(annotated, 1.0, mask_colored, 0.4, 0)

                # Draw actual object boundary
                cv2.drawContours(annotated, contours, -1, color, thickness=2)

                # Label above the contour
                bx, by, bw, bh = cv2.boundingRect(contours[0])
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (bx, max(by - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # -----------------------
            # marker
            # -----------------------
            marker = Marker()
            marker.header = msg.header
            marker.header.frame_id = "map"

            marker.ns = "semantic"
            marker.id = self.marker_id
            self.marker_id += 1

            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.text = label

            map_pos=self.transform_to_map(x,y,msg.header.stamp)
            if map_pos is None:
                continue
            marker.pose.position.x = map_pos[0]
            marker.pose.position.y = map_pos[1]
            marker.pose.position.z = map_pos[2] +0.5
            marker.pose.orientation.w = 1.0

            marker.scale.z = 0.4
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        # -----------------------
        # publish
        # -----------------------
        out_img = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_img.header = msg.header

        self.annotated_pub.publish(out_img)
        self.marker_pub.publish(marker_array)

        self.get_logger().info(f"Markers: {len(marker_array.markers)}")


def main():
    rclpy.init()
    node = GroundedSAMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()