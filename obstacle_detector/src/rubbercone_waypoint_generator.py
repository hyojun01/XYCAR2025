#!/usr/bin/env python3

from obstacle_detector.msg import Waypoint  # 위에 import 추가

class XycarPlanner:
    def __init__(self):
        ...
        self.waypoints = []
        self.wp_idx = 0
        self.lookahead_dist = 1.0

        rospy.Subscriber("/waypoints", Waypoint, self.waypointCB)

    def waypointCB(self, msg):
        self.waypoints = []
        for i in range(msg.cnt):
            self.waypoints.append((msg.x_arr[i], msg.y_arr[i]))
        self.wp_idx = 0

    def followWaypoints(self):
        if not self.waypoints:
            return

        # 현재 목표 지점 찾기
        while self.wp_idx < len(self.waypoints):
            target = self.waypoints[self.wp_idx]
            dist = math.hypot(target[0], target[1])
            if dist > self.lookahead_dist:
                break
            self.wp_idx += 1

        if self.wp_idx >= len(self.waypoints):
            self.rubbercone_mode_flag = False  # 경로 끝났으면 종료
            return

        # 목표 포인트
        tx, ty = self.waypoints[self.wp_idx]
        angle_to_target = math.atan2(ty, tx) * 180 / math.pi
        self.ctrl_rubbercone.angle = self.pid.pid_control(angle_to_target)
        self.ctrl_rubbercone.speed = 5  # 회피 중 속도 고정
