#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import rospy, math
import time
from xycar_msgs.msg import xycar_motor, XycarMotor
from std_msgs.msg import String
from obstacle_detector.msg import Waypoint, Obstacles
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge
import numpy as np

class XycarPlanner:
    def __init__(self):
        rospy.init_node('xycar_planner', anonymous=True)

        # 퍼블리셔: 실제 모터 제어 토픽
        self.ctrl_cmd_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
        self.mode_pub     = rospy.Publisher('/mode', String, queue_size=1)
        self.cmd_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)

        # 서브스크라이버: 차선·정적 장애물·라바콘 웨이포인트
        rospy.Subscriber('/xycar_motor_lane',   xycar_motor, self.ctrlLaneCB)
        rospy.Subscriber('/xycar_motor_static', xycar_motor, self.ctrlStaticCB)
        rospy.Subscriber('/rubbercone_waypoints', Waypoint,  self.ctrlRubberconeCB)
        rospy.Subscriber('/raw_obstacles_static', Obstacles, self.obstacleCB)

        # 내부 상태
        self.ctrl_lane   = xycar_motor()
        self.ctrl_static = xycar_motor()
        self.ctrl_cmd    = XycarMotor()
        
        # 라바콘 상태 머신 플래그 & 카운터
        self.rubber_mode        = False
        self.detected_frames    = 0
        self.lost_frames        = 0
        self.prev_wp_cnt        = 0
        self.wp_cnt             = 0
        self.latest_wp          = None   # (x,y)
        self.target_wp          = None   # 고정된 목표점
        self.rubber_start_time  = None

        # 파라미터
        self.rcon_speed   = rospy.get_param('~rcon_speed', 10.0)
        self.entry_thresh = 3          # 연속 검출 프레임 수
        self.exit_thresh  = 5          # 연속 미검출 프레임 수
        self.reach_dist   = 0.3        # 목표 도달 임계거리[m]
        self.max_duration = 30.0       # 최대 라바콘 모드 지속 시간[s]

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            # — 상태 머신 업데이트 —
            # 1) 진입 조건
            if not self.rubber_mode:
                if self.wp_cnt >= 1:
                    self.detected_frames += 1
                else:
                    self.detected_frames = 0
                if  self.detected_frames >= self.entry_thresh:
                    self.rubber_mode       = True
                    self.rubber_start_time = rospy.Time.now()
                    self.target_wp         = self.latest_wp
                    rospy.loginfo("=== Enter Rubbercone Mode ===")
            else:
                # 2) 모드 중: 목표 업데이트 및 종료 체크
                if  self.wp_cnt >= 1:
                    # 웨이포인트가 계속 들어오면 타겟 갱신
                    self.target_wp = self.latest_wp
                    self.lost_frames = 0
                else:
                    self.lost_frames += 1

                # 종료 기준: 도달, 미검출 지속, 시간 초과
                dist_to_wp = math.hypot(*(self.target_wp or (float('inf'),0)))
                elapsed    = (rospy.Time.now() - self.rubber_start_time).to_sec()
                if (dist_to_wp < self.reach_dist
                    or self.lost_frames >= self.exit_thresh
                    or elapsed >= self.max_duration):
                    self.rubber_mode     = False
                    self.detected_frames = 0
                    self.lost_frames     = 0
                    rospy.loginfo("=== Exit Rubbercone Mode ===")

            # — 제어 명령 결정 —
            if self.rubber_mode and self.target_wp:
                # 라바콘 모드
                x, y = self.target_wp
                ang = math.degrees(math.atan2(y, x))
                speed, steer = self.rcon_speed, -ang
                mode_str = "Rubbercone"
            elif self.ctrl_static.flag:
                # 정적 장애물 회피 모드
                speed = self.ctrl_static.speed
                steer = self.ctrl_static.angle
                mode_str = "StaticObs"
            else:
                # 차선 주행 모드
                speed = self.ctrl_lane.speed
                steer = self.ctrl_lane.angle
                mode_str = "LaneFollowing"

            # 퍼블리시
            self.ctrl_cmd.speed = speed
            self.ctrl_cmd.angle = steer
            self.ctrl_cmd_pub.publish(self.ctrl_cmd)
            self.mode_pub.publish(mode_str)
            rospy.loginfo("[Mode:%s] speed=%.1f angle=%.1f", mode_str, speed, steer)

            rate.sleep()

    # 콜백들
    def ctrlLaneCB(self, msg):
        self.ctrl_lane = msg

    def ctrlStaticCB(self, msg):
        self.ctrl_static = msg

    def ctrlRubberconeCB(self, msg):
        # Waypoint 메시지에서 cnt, x_arr, y_arr를 읽어옵니다.
        self.wp_cnt = msg.cnt
        rospy.loginfo(f"[CTRL_CB] got waypoint cnt={msg.cnt}, x0={msg.x_arr[0]:.2f}, y0={msg.y_arr[0]:.2f}")
        if msg.cnt >= 1 and len(msg.x_arr) > 0 and len(msg.y_arr) > 0:
            # 첫 번째 웨이포인트만 사용
            self.latest_wp = (msg.x_arr[0], msg.y_arr[0])
        else:
            # 웨이포인트가 없으면 None
            self.latest_wp = None

    def obstacleCB(self, msg):
        # (기존 정적 장애물 처리 로직)
        pass

if __name__ == '__main__':
    try:
        XycarPlanner()
    except rospy.ROSInterruptException:
        pass
