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
        self.left_max_deg  = rospy.get_param('~left_max_deg', 85.0)
        self.detect_dist   = rospy.get_param('~detect_dist',   10.0)
        self.offset_dist   = rospy.get_param('~offset_dist',   1.6) 
        self.num_ctrl_pts  = rospy.get_param('~num_ctrl_pts',  3)
        self.num_waypts    = rospy.get_param('~num_waypts',    1000)
        self.min_dx        = rospy.get_param('~min_dx',        0.1)

        # control points buffer
        self.raw_pts = []  # list of (x, y)

        # Counter for consecutive frames with insufficient points
        self.consecutive_insufficient_points = 0
        self.INSUFFICIENT_POINTS_THRESHOLD = 40 # Threshold for consecutive frames

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
            self.consecutive_insufficient_points += 1
            # rospy.loginfo(f"Not enough left-side points detected. Consecutive count: {self.consecutive_insufficient_points}/{self.INSUFFICIENT_POINTS_THRESHOLD}")
            if self.consecutive_insufficient_points >= self.INSUFFICIENT_POINTS_THRESHOLD:
                # rospy.logwarn("Threshold for insufficient points reached. Publishing Waypoint with cnt=0.")
                wp = Waypoint()
                wp.cnt = 0
                wp.x_arr = [0.0] * 2000 # Waypoint message expects 2000 elements
                wp.y_arr = [0.0] * 2000
                self.wp_pub.publish(wp)
            return
        else:
            # Reset counter if sufficient points are found
            if self.consecutive_insufficient_points > 0:
                rospy.loginfo(f"Sufficient left-side points detected. Resetting insufficient points counter from {self.consecutive_insufficient_points}.")
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
        wp.x_arr = list(xs) + [0.0]*(2000-self.num_waypts)
        wp.y_arr = list(ys) + [0.0]*(2000-self.num_waypts)
        self.wp_pub.publish(wp)
        rospy.loginfo(f"Published {self.num_waypts} spline waypoints")

if __name__ == '__main__':
    try:
        LidarLeftOffsetExtractor()
    except rospy.ROSInterruptException:
        pass