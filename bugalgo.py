import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, ReliabilityPolicy

class Bug2Controller(Node):
    def __init__(self):
        super().__init__('bug2_controller')

        # Define QoS profile
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.status_pub = self.create_publisher(String, '/robot_status', qos)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', qos)

        # Subscribers
        qos_sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_sensor)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_sensor)
        self.goal_sub = self.create_subscription(Pose, '/goal_pose', self.goal_callback, qos_sensor)

        # Robot parameters
        self.forward_speed = 0.1
        self.turning_speed = 1.7  # เพิ่มจาก 0.5 เป็น 0.7 เพื่อตอบสนองเร็วขึ้น
        self.dist_too_close_to_wall = 0.27  # ปรับจาก 0.3 เป็น 0.4 (40 ซม.)
        self.dist_thresh_wf = 0.35          # ปรับจาก 0.4 เป็น 0.6 (60 ซม.)

        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.goal_x = 4.3
        self.goal_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.robot_mode = "go to goal mode"
        self.go_to_goal_state = "adjust heading"
        self.wall_following_state = "turn left"

        # Bug2 parameters
        self.bug2_switch = "ON"
        self.start_goal_line_calculated = False
        self.start_goal_line_slope_m = 0.0
        self.start_goal_line_y_intercept = 0.0
        self.hit_point_x = 0.0
        self.hit_point_y = 0.0
        self.leave_point_x = 0.0
        self.leave_point_y = 0.0
        self.distance_to_goal_from_hit_point = 0.0
        self.distance_to_goal_from_leave_point = 0.0
        self.dist_thresh_bug2 = 0.3  # ปรับจาก 0.3 เป็น 0.5 (50 ซม.) เพื่อตอบสนองไกลขึ้น
        self.distance_to_start_goal_line_precision = 0.1 #สำคัณเลยอันนี้
        self.leave_point_to_hit_point_diff = 0.05
        self.yaw_precision = math.radians(5.0)
        self.dist_precision = 0.1

        # Laser scan readings
        self.left_dist = 999.9
        self.leftfront_dist = 999.9
        self.front_dist = 999.9
        self.rightfront_dist = 999.9
        self.right_dist = 999.9

        self.get_logger().info("Bug2Controller initialized with default goal (4.0, 0.0)")

    def euler_from_quaternion(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def goal_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.start_x = self.current_x
        self.start_y = self.current_y
        self.start_goal_line_calculated = False
        status_msg = String()
        status_msg.data = f"Goal set: ({self.goal_x}, {self.goal_y})"
        self.status_pub.publish(status_msg)
        self.publish_start_goal_line_marker()

    def scan_callback(self, msg):
        num_ranges = len(msg.ranges)
        if num_ranges == 0:
            self.get_logger().warn("No laser scan data received!")
            return

        angle_increment = (msg.angle_max - msg.angle_min) / (num_ranges - 1) if num_ranges > 1 else 0
        if angle_increment == 0:
            self.get_logger().warn("Invalid angle increment, skipping scan data")
            return

        def get_index(angle_deg):
            angle_rad = math.radians(angle_deg)
            index = int((angle_rad - msg.angle_min) / angle_increment)
            return min(max(index, 0), num_ranges - 1)

        self.front_dist = msg.ranges[get_index(0)]
        self.leftfront_dist = msg.ranges[get_index(45)]
        self.left_dist = msg.ranges[get_index(90)]
        self.right_dist = msg.ranges[get_index(270)]
        self.rightfront_dist = msg.ranges[get_index(315)]

        self.get_logger().info(f"Scan: F={self.front_dist}, LF={self.leftfront_dist}, RF={self.rightfront_dist}, num_ranges={num_ranges}")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        _, _, self.current_yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        self.get_logger().info(f"Odometry: x={self.current_x}, y={self.current_y}, yaw={self.current_yaw}")
        self.bug2()

    def bug2(self):
        if not self.start_goal_line_calculated:
            self.robot_mode = "go to goal mode"
            self.start_goal_line_slope_m = (self.goal_y - self.start_y) / (self.goal_x - self.start_x + 1e-6)
            self.start_goal_line_y_intercept = self.goal_y - (self.start_goal_line_slope_m * self.goal_x)
            self.start_goal_line_calculated = True
            self.publish_start_goal_line_marker()

        if self.robot_mode == "go to goal mode":
            self.go_to_goal()
        elif self.robot_mode == "wall following mode":
            self.follow_wall()

    def go_to_goal(self):
        msg = Twist()
        self.get_logger().info(f"Go to goal - State: {self.go_to_goal_state}")

        # ตรวจจับสิ่งกีดขวางในระยะไกลขึ้น (0.5 m)
        if (self.front_dist < self.dist_thresh_bug2 or 
            self.leftfront_dist < self.dist_thresh_bug2 or 
            self.rightfront_dist < self.dist_thresh_bug2):
            if self.front_dist < self.dist_thresh_bug2:
                status_msg = String()
                status_msg.data = f"Stopped 50 cm before obstacle at ({self.current_x}, {self.current_y})"
                self.status_pub.publish(status_msg)
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.cmd_vel_pub.publish(msg)
                self.robot_mode = "wall following mode"
                self.hit_point_x = self.current_x
                self.hit_point_y = self.current_y
                self.distance_to_goal_from_hit_point = math.sqrt(
                    (self.goal_x - self.hit_point_x) ** 2 + (self.goal_y - self.hit_point_y) ** 2
                )
                self.publish_hit_point_marker()
                return
            else:
                self.robot_mode = "wall following mode"
                self.hit_point_x = self.current_x
                self.hit_point_y = self.current_y
                self.distance_to_goal_from_hit_point = math.sqrt(
                    (self.goal_x - self.hit_point_x) ** 2 + (self.goal_y - self.hit_point_y) ** 2
                )
                self.publish_hit_point_marker()
                status_msg = String()
                status_msg.data = f"Hit obstacle at ({self.hit_point_x}, {self.hit_point_y})"
                self.status_pub.publish(status_msg)
                msg.angular.z = self.turning_speed  # 0.7 เพื่อหมุนเร็วขึ้น
                self.cmd_vel_pub.publish(msg)
                return

        desired_yaw = math.atan2(self.goal_y - self.current_y, self.goal_x - self.current_x)
        yaw_error = desired_yaw - self.current_yaw
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        if abs(yaw_error) > self.yaw_precision:
            msg.angular.z = self.turning_speed if yaw_error > 0 else -self.turning_speed  # 0.7
            self.go_to_goal_state = "adjust heading"
            status_msg = String()
            status_msg.data = "Adjusting heading"
            self.status_pub.publish(status_msg)
        else:
            distance_to_goal = math.sqrt((self.goal_x - self.current_x) ** 2 + (self.goal_y - self.current_y) ** 2)
            if distance_to_goal > self.dist_precision:
                msg.linear.x = self.forward_speed  # 0.1
                self.go_to_goal_state = "go straight"
                status_msg = String()
                status_msg.data = "Going straight to goal"
                self.status_pub.publish(status_msg)
            else:
                self.get_logger().info("Goal achieved!")
                status_msg = String()
                status_msg.data = "Goal achieved!"
                self.status_pub.publish(status_msg)
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.go_to_goal_state = "goal achieved"

        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f"Cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}")

    def follow_wall(self):
        msg = Twist()
        d = self.dist_thresh_wf  # 60 ซม.

        # ตรวจสอบการกลับมาที่เส้น start-goal line
        y_on_line = self.start_goal_line_slope_m * self.current_x + self.start_goal_line_y_intercept
        distance_to_line = abs(y_on_line - self.current_y)
        if distance_to_line < self.distance_to_start_goal_line_precision:  # 5 ซม.
            self.leave_point_x = self.current_x
            self.leave_point_y = self.current_y
            self.distance_to_goal_from_leave_point = math.sqrt(
                (self.goal_x - self.leave_point_x) ** 2 + (self.goal_y - self.leave_point_y) ** 2
            )
            if self.distance_to_goal_from_hit_point - self.distance_to_goal_from_leave_point > self.leave_point_to_hit_point_diff:  # 10 ซม.
                self.robot_mode = "go to goal mode"
                self.publish_leave_point_marker()
                status_msg = String()
                status_msg.data = f"Leaving wall at ({self.leave_point_x}, {self.leave_point_y})"
                self.status_pub.publish(status_msg)
                return

        # ปรับพฤติกรรมให้ตอบสนองสิ่งกีดขวางเร็วขึ้น
        if self.front_dist > d and self.rightfront_dist > d and self.leftfront_dist > d:
            self.wall_following_state = "search for wall"
            msg.linear.x = self.forward_speed * 0.8  # 0.08
            msg.angular.z = -self.turning_speed * 0.7  # -0.49
        elif self.front_dist < d:
            self.wall_following_state = "turn left"
            msg.angular.z = self.turning_speed  # 0.7 เพื่อหมุนเร็วขึ้น
        elif self.rightfront_dist < d and self.rightfront_dist > self.dist_too_close_to_wall:
            self.wall_following_state = "follow wall"
            msg.linear.x = self.forward_speed  # 0.1
        elif self.rightfront_dist < self.dist_too_close_to_wall:
            self.wall_following_state = "turn left"
            msg.angular.z = self.turning_speed * 0.8  # 0.56 เพื่อหมุนเร็วขึ้น
        else:
            self.wall_following_state = "go straight"
            msg.linear.x = self.forward_speed  # 0.1

        status_msg = String()
        status_msg.data = f"Wall following: {self.wall_following_state}, RF={self.rightfront_dist:.2f}, Dist_to_line={distance_to_line:.2f}"
        self.status_pub.publish(status_msg)
        self.cmd_vel_pub.publish(msg)

    def publish_start_goal_line_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bug2"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        start_point = Pose().position
        start_point.x = self.start_x
        start_point.y = self.start_y
        goal_point = Pose().position
        goal_point.x = self.goal_x
        goal_point.y = self.goal_y
        marker.points = [start_point, goal_point]

        self.marker_pub.publish(marker)

    def publish_hit_point_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bug2"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.pose.position.x = self.hit_point_x
        marker.pose.position.y = self.hit_point_y
        marker.pose.position.z = 0.0

        self.marker_pub.publish(marker)

    def publish_leave_point_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bug2"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.position.x = self.leave_point_x
        marker.pose.position.y = self.leave_point_y
        marker.pose.position.z = 0.0

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = Bug2Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()