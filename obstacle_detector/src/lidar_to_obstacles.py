#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiDAR → Obstacles 변환 노드
  • 전방(FOV) + 같은 차선 폭 내 점만 사용
  • 1-D 연속거리(adaptive) 클러스터링 → CircleObstacle[]
  • /raw_obstacles_static  : 정적 장애물 목록
  • /front_obstacle_distance : 전방 최단 거리 (Float64)
"""

import rospy, math
import numpy as np
from sensor_msgs.msg import LaserScan
from obstacle_detector.msg import Obstacles, CircleObstacle
from geometry_msgs.msg    import Point, Vector3
from std_msgs.msg         import Float64

# ────────── 파라미터 (필드에서 튜닝) ──────────
FRONT_FOV_DEG      = 60          # 전방 ±30°
LANE_HALF_WIDTH    = 3.0         # 차선 중심선 기준 반폭 [m]
K_ADAPT            = 1.8         # adaptive 계수 (1.3~2.0)
MIN_CLUSTER_SIZE   = 3           # ≥3 점이면 장애물로 인정
PUB_HZ             = 15          # 퍼블리시 주기 [Hz]


MIN_RADIUS         = 0.3         # 장애물 최소 반지름 [m] (라바콘 등 필터링)
class Lidar2Obstacles:
    def __init__(self):
        rospy.init_node("lidar_to_obstacles", anonymous=True)

        self.scan_msg = None
        self.pub_obs  = rospy.Publisher("/raw_obstacles_static",
                                        Obstacles, queue_size=1)
        self.pub_dist = rospy.Publisher("/front_obstacle_distance",
                                        Float64,   queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self.scan_cb, queue_size=1)

        rate = rospy.Rate(PUB_HZ)
        while not rospy.is_shutdown():
            if self.scan_msg:
                obs_msg, d_min = self.process_scan(self.scan_msg)
                self.pub_obs.publish(obs_msg)
                self.pub_dist.publish(Float64(d_min))
            rate.sleep()

    # ────────── 스캔 수신 ──────────
    def scan_cb(self, msg):        # 콜백: 최신 스캔 캐시
        self.scan_msg = msg

    # ────────── 핵심 처리 ──────────
    def process_scan(self, scan):
        # 1) 배열 생성
        ranges = np.asarray(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        # 2) 유효 + 전방 FOV 필터
        finite_mask = np.isfinite(ranges)
        fov_mask    = np.abs(np.rad2deg(angles)) <= (FRONT_FOV_DEG/2)
        mask        = finite_mask & fov_mask
        ranges, angles = ranges[mask], angles[mask]

        # 스캔에 값이 없으면 빈 메시지 반환
        if len(ranges) == 0:
            return Obstacles(header=scan.header), float('inf')

        # 3) 극 → 직교 좌표
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pts = np.column_stack((xs, ys))

        # 4) 1-D 연속거리 클러스터링 (adaptive threshold)
        clusters, cur = [], [pts[0]]
        for i in range(1, len(pts)):
            thr = K_ADAPT * 2.0 * ranges[i] * math.sin(scan.angle_increment/2.0)
            if np.linalg.norm(pts[i] - pts[i-1]) < thr:
                cur.append(pts[i])
            else:
                if len(cur) >= MIN_CLUSTER_SIZE:
                    clusters.append(np.asarray(cur))
                cur = [pts[i]]
        if len(cur) >= MIN_CLUSTER_SIZE:
            clusters.append(np.asarray(cur))

        # 5) 클러스터 → CircleObstacle[]
        circles = []
        for c in clusters:
            center = c.mean(axis=0)          # (x,y)
            # 같은 차선폭 내? (y=좌+, –우)
            if abs(center[1]) > LANE_HALF_WIDTH:
                continue
            rad = np.max(np.linalg.norm(c - center, axis=1))
            if rad < MIN_RADIUS:  #  라바콘 등 작은 물체 필터링
                continue

            circ = CircleObstacle()
            circ.center      = Point(center[0], center[1], 0.0)
            circ.velocity    = Vector3()     # 정적 (0,0,0)
            circ.radius      = rad
            circ.true_radius = rad
            circles.append(circ)

        # 6) 전방 최단 거리 계산 (여전히 FOV 내부)
        d_min = float(np.min(ranges)) if len(ranges) else float('inf')

        # 7) Obstacles 메시지 작성
        obs_msg = Obstacles()
        obs_msg.header.stamp    = rospy.Time.now()
        obs_msg.header.frame_id = scan.header.frame_id
        obs_msg.circles         = circles
        return obs_msg, d_min


# ────────── 노드 실행 ──────────
if __name__ == "__main__":
    try:
        Lidar2Obstacles()
    except rospy.ROSInterruptException:
        pass