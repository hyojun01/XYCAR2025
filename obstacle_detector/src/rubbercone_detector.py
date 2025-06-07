#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from obstacle_detector.msg import Waypoint
from scipy.interpolate import CubicSpline
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class LidarLeftOffsetExtractor:
    def __init__(self):
        rospy.init_node('rubbercone_left_offset')
        self.min_range     = rospy.get_param('~min_range',    1.0)
        self.max_range     = rospy.get_param('~max_range',   35.0)
        self.left_min_deg  = rospy.get_param('~left_min_deg', 5.0)
        self.left_max_deg  = rospy.get_param('~left_max_deg', 85.0)
        self.detect_dist   = rospy.get_param('~detect_dist',   10.0)
        self.offset_dist   = rospy.get_param('~offset_dist',   1.6) 
        self.num_ctrl_pts  = rospy.get_param('~num_ctrl_pts',  3)
        self.num_waypts    = rospy.get_param('~num_waypts',    1000)
        self.min_dx        = rospy.get_param('~min_dx',        0.1)

        self.camera_trigger = False  # 카메라가 트리거했는지 여부
        rospy.Subscriber('/usb_cam/image_raw', Image, self.image_cb)
        self.bridge = CvBridge() # cv_bridge 초기화
        # control points buffer
        self.raw_pts = []  # list of (x, y)

        # Counter for consecutive frames with insufficient points
        self.consecutive_insufficient_points = 0
        self.INSUFFICIENT_POINTS_THRESHOLD = 28 # Threshold for consecutive frames

        # Publisher / Subscriber
        self.wp_pub = rospy.Publisher('/rubbercone_waypoints', Waypoint, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        # rospy.loginfo("Rubbercone Left-Offset: publishing smoothed offset waypoints")
        rospy.spin()

    def scan_cb(self, scan: LaserScan):
        if not self.camera_trigger:
            return
        # Convert ranges and angles
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges), dtype=np.float32)
        rel_deg = (np.degrees(angles) + 180) % 360 - 180

        # Filter left side cones
        mask_l = (
            (ranges >= self.min_range) & (ranges <= self.max_range) &
            (rel_deg >= self.left_min_deg) & (rel_deg <= self.left_max_deg) &
            (ranges <= self.detect_dist)
        )
        left_pts = [(ranges[i]*math.cos(angles[i]), ranges[i]*math.sin(angles[i]))
                    for i in np.where(mask_l)[0]]
        if len(left_pts) < 2:
            self.consecutive_insufficient_points += 1
            # rospy.loginfo(f"Not enough left-side points detected. Consecutive count: {self.consecutive_insufficient_points}/{self.INSUFFICIENT_POINTS_THRESHOLD}")
            if self.consecutive_insufficient_points >= self.INSUFFICIENT_POINTS_THRESHOLD:
                # rospy.logwarn("Threshold for insufficient points reached. Publishing Waypoint with cnt=0.")
                wp = Waypoint()
                wp.cnt = 0
                wp.x_arr = [0.0] * 2000 # Waypoint message expects 2000 elements
                wp.y_arr = [0.0] * 2000
                self.wp_pub.publish(wp)

                self.camera_trigger = False
                # rospy.loginfo("웨이포인트 중단 → camera_trigger OFF")
            return
        else:
            # Reset counter if sufficient points are found
            # if self.consecutive_insufficient_points > 0:
                # rospy.loginfo(f"Sufficient left-side points detected. Resetting insufficient points counter from {self.consecutive_insufficient_points}.")
            self.consecutive_insufficient_points = 0

        # Sort by forward x
        left_pts.sort(key=lambda p: p[0])
        # Sample at fixed offset intervals
        selected = []
        last_x = -float('inf')
        for lx, ly in left_pts:
            if lx >= last_x + self.offset_dist:
                selected.append((lx, ly))
                last_x = lx
        if not selected:
            return

        # Compute offset points (to the right of heading)
        offset_pts = []
        for i, (lx, ly) in enumerate(selected):
            # heading based on next or previous point
            if i < len(selected) - 1:
                nx, ny = selected[i+1][0] - lx, selected[i+1][1] - ly
            else:
                nx, ny = lx - selected[i-1][0], ly - selected[i-1][1]
            theta = math.atan2(ny, nx)
            # right-normal vector = (sin(theta), -cos(theta))
            ox = math.sin(theta) * self.offset_dist
            oy = -math.cos(theta) * self.offset_dist
            wx = lx + ox
            wy = ly + oy
            offset_pts.append((wx, wy))

        # Take first offset point as new control point
        wx, wy = offset_pts[0]
        if not self.raw_pts or abs(wx - self.raw_pts[-1][0]) > self.min_dx:
            self.raw_pts.append((wx, wy))
            if len(self.raw_pts) > self.num_ctrl_pts:
                self.raw_pts.pop(0)

        # Publish spline or fallback
        if len(self.raw_pts) >= self.num_ctrl_pts:
            self._publish_spline()
        else:
            # fallback single point
            pt = self.raw_pts[-1]
            wp = Waypoint(); wp.cnt = 1
            wp.x_arr = [pt[0]] + [0.0]*1999
            wp.y_arr = [pt[1]] + [0.0]*1999
            self.wp_pub.publish(wp)
            # rospy.loginfo(f"Fallback WP @ ({pt[0]:.2f},{pt[1]:.2f})")


    def image_cb(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        # 이미지 크기 정보
        h, w = cv_image.shape[:2]

        # 하단 영역만 잘라내기 (위쪽 25% 제외)
        cropped = cv_image[int(h * 0.25):, :]

        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # 주황색 범위 설정
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # 주황색 비율 계산
        orange_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
        if orange_ratio > 0.01:
            # rospy.loginfo("주황색 라바콘 감지됨 → 웨이포인트 생성")
            self.camera_trigger = True  # ✅ 트리거 설정



    def _publish_spline(self):
        pts = np.array(self.raw_pts)
        # parameterize by cumulative distance
        deltas = np.diff(pts, axis=0)
        dists = np.hypot(deltas[:,0], deltas[:,1])
        cum = np.concatenate(([0], np.cumsum(dists)))
        t = cum / cum[-1]
        # cubic splines
        csx = CubicSpline(t, pts[:,0], bc_type='clamped')
        csy = CubicSpline(t, pts[:,1], bc_type='clamped')
        samples = np.linspace(0, 1, self.num_waypts)
        xs, ys = csx(samples), csy(samples)
        wp = Waypoint(); wp.cnt = self.num_waypts
        wp.x_arr = list(xs) + [0.0]*(2000-self.num_waypts)
        wp.y_arr = list(ys) + [0.0]*(2000-self.num_waypts)
        self.wp_pub.publish(wp)
        # rospy.loginfo(f"Published {self.num_waypts} spline waypoints")

if __name__ == '__main__':
    try:
        LidarLeftOffsetExtractor()
    except rospy.ROSInterruptException:
        pass