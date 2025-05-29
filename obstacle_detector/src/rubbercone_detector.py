#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from obstacle_detector.msg import Waypoint
from scipy.interpolate import CubicSpline

class LidarLeftOffsetExtractor:
    def __init__(self):
        rospy.init_node('rubbercone_left_offset')
        self.min_range     = rospy.get_param('~min_range',    1.0)
        self.max_range     = rospy.get_param('~max_range',   35.0)
        self.left_min_deg  = rospy.get_param('~left_min_deg', 5.0)
        self.left_max_deg  = rospy.get_param('~left_max_deg', 75.0)
        self.detect_dist   = rospy.get_param('~detect_dist',   10.0)
        self.offset_dist   = rospy.get_param('~offset_dist',   2.0) 
        self.num_ctrl_pts  = rospy.get_param('~num_ctrl_pts',  3)
        self.num_waypts    = rospy.get_param('~num_waypts',    100)
        self.min_dx        = rospy.get_param('~min_dx',        0.1)

        # control points buffer
        self.raw_pts = []  # list of (x, y)

        # Publisher / Subscriber
        self.wp_pub = rospy.Publisher('/rubbercone_waypoints', Waypoint, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        rospy.loginfo("Rubbercone Left-Offset: publishing smoothed offset waypoints")
        rospy.spin()

    def scan_cb(self, scan: LaserScan):
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
            return

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
            wp.x_arr = [pt[0]] + [0.0]*199
            wp.y_arr = [pt[1]] + [0.0]*199
            self.wp_pub.publish(wp)
            rospy.loginfo(f"Fallback WP @ ({pt[0]:.2f},{pt[1]:.2f})")

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
        wp.x_arr = list(xs) + [0.0]*(200-self.num_waypts)
        wp.y_arr = list(ys) + [0.0]*(200-self.num_waypts)
        self.wp_pub.publish(wp)
        rospy.loginfo(f"Published {self.num_waypts} spline waypoints")

if __name__ == '__main__':
    try:
        LidarLeftOffsetExtractor()
    except rospy.ROSInterruptException:
        pass








# import rospy
# import numpy as np
# import math
# from sensor_msgs.msg import LaserScan
# from obstacle_detector.msg import Waypoint
# from scipy.interpolate import CubicSpline

# class LidarRubberconeExtractor:
#     def __init__(self):
#         rospy.init_node('rubbercone_detector')

#         # ─── 파라미터 ─────────────────────────────────────
#         self.min_range     = rospy.get_param('~min_range',    1.0)
#         self.max_range     = rospy.get_param('~max_range',   35.0)
#         self.left_min_deg  = rospy.get_param('~left_min_deg', 20.0)
#         self.left_max_deg  = rospy.get_param('~left_max_deg', 60.0)
#         self.right_min_deg = rospy.get_param('~right_min_deg', -60.0)
#         self.right_max_deg = rospy.get_param('~right_max_deg', -20.0)
#         self.detect_dist   = rospy.get_param('~detect_dist',   8.0)
#         self.timeout       = rospy.get_param('~pair_timeout',  4.0)
#         self.min_width     = rospy.get_param('~min_width',     0.2)
#         self.y_tol         = rospy.get_param('~y_tolerance',   0.1)

#         # spline 관련
#         self.num_ctrl_pts  = rospy.get_param('~num_ctrl_pts',  3)
#         self.num_waypts    = rospy.get_param('~num_waypts',    6)

#         # 상태 플래그
#         self.clear_timeout     = rospy.get_param('~clear_timeout', 8.0)
#         self.last_pair_time    = rospy.Time.now()
#         self.raw_pts           = []
#         self.first_pair_done   = False
#         self.waiting_left_pt   = None
#         self.waiting_left_time = None

#         # 퍼블리셔/구독자
#         self.wp_pub = rospy.Publisher('/rubbercone_waypoints', Waypoint, queue_size=1)
#         rospy.Subscriber('/scan', LaserScan, self.scan_cb)

#         rospy.loginfo("Rubbercone Detector: initial y-match pairing then sequential L-R")
#         rospy.spin()

#     def scan_cb(self, scan: LaserScan):
#         now = rospy.Time.now()
#         ranges = np.array(scan.ranges, dtype=np.float32)
#         angles = np.linspace(scan.angle_min, scan.angle_max, ranges.shape[0], dtype=np.float32)
#         rel_deg = (np.degrees(angles) + 180) % 360 - 180

#         # ROI 필터링
#         mask_l = ((ranges >= self.min_range) & (ranges <= self.max_range) &
#                   (rel_deg >= self.left_min_deg) & (rel_deg <= self.left_max_deg) &
#                   (ranges <= self.detect_dist))
#         mask_r = ((ranges >= self.min_range) & (ranges <= self.max_range) &
#                   (rel_deg >= self.right_min_deg) & (rel_deg <= self.right_max_deg) &
#                   (ranges <= self.detect_dist))

#         # 좌/우 후보점 리스트
#         left_pts  = [(ranges[i]*math.cos(angles[i]), ranges[i]*math.sin(angles[i])) for i in np.where(mask_l)[0]]
#         right_pts = [(ranges[i]*math.cos(angles[i]), ranges[i]*math.sin(angles[i])) for i in np.where(mask_r)[0]]

#         # 1) 초기 한 번, y좌표 매칭 페어링
#         if not self.first_pair_done and left_pts and right_pts:
#             for lx, ly in left_pts:
#                 for rx, ry in right_pts:
#                     if abs(ly - ry) < self.y_tol:
#                         # 초기 페어링
#                         cx, cy = 0.5*(lx + rx), 0.5*(ly + ry)
#                         self._add_midpoint(cx, cy)
#                         self.first_pair_done = True
#                         rospy.loginfo(f"Initial y-pair WP @ ({cx:.2f},{cy:.2f})")
#                         return

#         # 2) 이후 sequential L->R 페어링
#         # 왼쪽 먼저 기다림
#         if left_pts and self.waiting_left_pt is None:
#             lx, ly = min(left_pts, key=lambda p: p[0])  # 가장 전방의 왼쪽 콘
#             self.waiting_left_pt   = (lx, ly)
#             self.waiting_left_time = now
#             return

#         # 오른쪽 감지 시 페어링
#         if self.waiting_left_pt and right_pts:
#             if (now - self.waiting_left_time).to_sec() <= self.timeout:
#                 # 좌우 페어링
#                 lx, ly = self.waiting_left_pt
#                 rx, ry = min(right_pts, key=lambda p: p[0])  # 가장 전방의 오른쪽 콘
#                 cx, cy = 0.5*(lx + rx), 0.5*(ly + ry)
#                 self._add_midpoint(cx, cy)
#                 rospy.loginfo(f"Seq L-R WP @ ({cx:.2f},{cy:.2f})")
#             # reset
#             self.waiting_left_pt = None
#             self.waiting_left_time = None
#             return

#         # 3) 오래 기다리면 reset
#         if self.waiting_left_time and (now - self.waiting_left_time).to_sec() > self.timeout:
#             self.waiting_left_pt = None
#             self.waiting_left_time = None

#     def _add_midpoint(self, cx, cy):
#         # forward-progress 체크
#         if not self.raw_pts or cx > self.raw_pts[-1][0] + 0.03:
#             self.raw_pts.append((cx, cy))
#             if len(self.raw_pts) > self.num_ctrl_pts:
#                 self.raw_pts.pop(0)
#         # spline 또는 fallback
#         if len(self.raw_pts) >= self.num_ctrl_pts:
#             self._publish_spline()
#         elif len(self.raw_pts) >= 1:
#             fx, fy = self.raw_pts[-1]
#             wp = Waypoint(); wp.cnt = 1
#             wp.x_arr = [fx] + [0.0]*199
#             wp.y_arr = [fy] + [0.0]*199
#             self.wp_pub.publish(wp)
#             rospy.loginfo(f"Fallback WP @ ({fx:.2f},{fy:.2f})")
#         self.last_pair_time = rospy.Time.now()

#     def _publish_spline(self):
#         pts = np.array(self.raw_pts)
#         deltas = np.diff(pts, axis=0)
#         dists = np.hypot(deltas[:,0], deltas[:,1])
#         cum = np.concatenate(([0], np.cumsum(dists)))
#         t = cum / cum[-1]
#         csx = CubicSpline(t, pts[:,0], bc_type='clamped')
#         csy = CubicSpline(t, pts[:,1], bc_type='clamped')
#         samples = np.linspace(0, 1, self.num_waypts)
#         xs, ys = csx(samples), csy(samples)
#         wp = Waypoint(); wp.cnt = self.num_waypts
#         wp.x_arr = list(xs) + [0.0]*(200-self.num_waypts)
#         wp.y_arr = list(ys) + [0.0]*(200-self.num_waypts)
#         self.wp_pub.publish(wp)

# if __name__ == '__main__':
#     try:
#         LidarRubberconeExtractor()
#     except rospy.ROSInterruptException:
#         pass



    
# import rospy
# import numpy as np
# from sensor_msgs.msg import LaserScan
# from obstacle_detector.msg import Obstacles
# import math

# class LidarRubberconeExtractor:
#     def __init__(self):
#         rospy.init_node('rubbercone_detector')

#         # parameters
#         self.min_range   = rospy.get_param('~min_range',   0.2)
#         self.max_range   = rospy.get_param('~max_range',   5.0)
#         # parameters: 좌/우 ROI 각도 범위 (degree)
#         #   left: rel_deg ∈ [ left_min, left_max ]
#         #   right: rel_deg ∈ [ right_min, right_max ]
#         self.left_min_deg  = rospy.get_param('~left_min_deg',  20.0)
#         self.left_max_deg  = rospy.get_param('~left_max_deg',  60.0)
#         self.right_min_deg = rospy.get_param('~right_min_deg', -60.0)
#         self.right_max_deg = rospy.get_param('~right_max_deg', -20.0)
#         # detect_dist 이하 점만 추출
#         self.detect_dist = rospy.get_param('~detect_dist', 0.5)
#         # publisher
#         self.pub = rospy.Publisher(
#             '/raw_obstacles_rubbercone',
#             Obstacles,
#             queue_size=1
#         )
#         # subscriber
#         rospy.Subscriber('/scan', LaserScan, self.scan_cb)
#         rospy.loginfo("LiDAR-only rubbercone extractor started")
#         rospy.spin()

#     def scan_cb(self, scan: LaserScan):
#         # 1) ROI filtering
#         ranges_all = np.array(scan.ranges, dtype=np.float32)
#         angles_all = np.linspace(
#             scan.angle_min,
#             scan.angle_max,
#             ranges_all.shape[0],
#             dtype=np.float32
#         )
#         rel_deg = (np.degrees(angles_all) + 180) % 360 - 180
#        # 2) 좌/우 ROI 각각 필터링
#         mask_left = (
#             (ranges_all >= self.min_range) &
#             (ranges_all <= self.max_range) &
#             (rel_deg >= self.left_min_deg) &
#             (rel_deg <= self.left_max_deg)
#         )
#         mask_right = (
#             (ranges_all >= self.min_range) &
#             (ranges_all <= self.max_range) &
#             (rel_deg >= self.right_min_deg) &
#             (rel_deg <= self.right_max_deg)
#         )
#         # 좌우 각각의 유효한 (angle, range) 리스트
#         angles_l = angles_all[mask_left];   ranges_l = ranges_all[mask_left]
#         angles_r = angles_all[mask_right];  ranges_r = ranges_all[mask_right]

#         # 2) detect_dist 이하 점 인덱스 모두 찾기
#         idxs = np.where(ranges <= self.detect_dist)[0]

#         # 3) 각 점마다 별도 메시지 생성·발행
#         for i in idxs:
#             x = ranges[i] * np.cos(angles[i])
#             y = ranges[i] * np.sin(angles[i])

#             obs_msg = Obstacles()
#             obs_msg.header.stamp    = scan.header.stamp
#             obs_msg.header.frame_id = scan.header.frame_id

#             # Point+true_radius 메시지 필드에 값 채우기
#             obs_msg.point.x     = x
#             obs_msg.point.y     = y
#             obs_msg.point.z     = 0.0
#             obs_msg.true_radius = float(ranges[i])

#             self.pub.publish(obs_msg)
#         # (감지된 점이 없으면 아무 메시지도 발행되지 않습니다)

# if __name__ == '__main__':
#     try:
#         LidarRubberconeExtractor()
#     except rospy.ROSInterruptException:
#         pass

