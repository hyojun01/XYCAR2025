#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from obstacle_detector.msg import Obstacles, CircleObstacle, Waypoint #Waypoint 메시지 임포트
from xycar_msgs.msg import XycarMotor
from scipy.spatial import Delaunay

class RubberconeWaypointNode:
    def __init__(self):
        rospy.init_node('rubbercone_waypoint_node')

        # === 파라미터 ===
        self.min_x = rospy.get_param("~min_x", 0.0)
        self.max_x = rospy.get_param("~max_x", 3.0)
        self.min_y = rospy.get_param("~min_y", -1.5)
        self.max_y = rospy.get_param("~max_y", 1.5)

        # === 퍼블리셔 & 서브스크라이버 ===
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.waypoint_pub = rospy.Publisher("/waypoints", Waypoint, queue_size=1)
        self.motor_pub = rospy.Publisher("/xycar_motor", XycarMotor, queue_size=1)

        self.rate = rospy.Rate(10)
        rospy.spin()

    def scan_callback(self, msg):
        points = []
        angle = msg.angle_min

        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                if self.min_x < x < self.max_x and self.min_y < y < self.max_y:
                    points.append([x, y])
            angle += msg.angle_increment

        if len(points) < 3:
            return

        points_np = np.array(points)

        try:
            tri = Delaunay(points_np)
        except:
            rospy.logwarn("Delaunay triangulation failed")
            return

        # === 삼각형 중심점 계산 ===
        centers = []
        for simplex in tri.simplices:
            triangle = points_np[simplex]
            center = np.mean(triangle, axis=0)
    
            # y축 오프셋 적용 (중앙선 회피)
            offset = 0.5 if center[1] >= 0 else -0.5
            center[1] += offset

            centers.append(center)
        
        # === Waypoint 메시지 생성 ===
        waypoint_msg = Waypoint()
        cnt = min(len(centers), 200)
        waypoint_msg.cnt = cnt
        x_arr = np.zeros(200, dtype=np.float32)
        y_arr = np.zeros(200, dtype=np.float32)

        centers_sorted = sorted(centers, key=lambda p: p[0])  # x 기준 정렬
        for i in range(cnt):
            x_arr[i] = centers_sorted[i][0]
            y_arr[i] = centers_sorted[i][1]

        waypoint_msg.x_arr = x_arr.tolist()
        waypoint_msg.y_arr = y_arr.tolist()

        self.waypoint_pub.publish(waypoint_msg)

        # === 가장 가까운 장애물에 반응해 속도 조정 ===
        if cnt > 0:
            closest = centers_sorted[0]
            dist = np.linalg.norm(closest)

            motor_msg = XycarMotor()
            if dist < 1.5:
                motor_msg.speed = 4.0
            else:
                motor_msg.speed = 7.0

            motor_msg.angle = self.compute_steering(closest)
            self.motor_pub.publish(motor_msg)

    def compute_steering(self, target_point):
        """간단한 조향 계산 함수 (직선 경로 따라가기용)"""
        x, y = target_point
        angle_deg = -np.degrees(np.arctan2(y, x))  # 좌우 반영
        return np.clip(angle_deg, -50.0, 50.0)

if __name__ == '__main__':
    try:
        node = RubberconeWaypointNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass