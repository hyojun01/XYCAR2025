#! /usr/bin/env python3

# 기본 Python3 인터프리터 설정

from __future__ import print_function

from xycar_msgs.msg import xycar_motor, XycarMotor  # xycar 모터 메시지 모듈 임포트
from sensor_msgs.msg import Imu  # IMU 데이터 메시지 모듈 임포트
from std_msgs.msg import Float32, String, Int64 # Float32 메시지 모듈 임포트
from obstacle_detector.msg import Waypoint, Obstacles, CarObstacles

from math import radians, pi  # 각도를 라디안으로 변환하는 함수 임포트

import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트
import math

from cv_bridge import CvBridge, CvBridgeError  # CV-Bridge 라이브러리 임포트

import time, math
import rospy  # ROS 파이썬 라이브러리 임포트
from sensor_msgs.msg import Image, CompressedImage  # 이미지 데이터 메시지 모듈 임포트



import tf


import tkinter as tk

class Obstacle:
    def __init__(self, x=None, y=None, distance=None):
        self.x = x
        self.y = y
        self.distance = distance

def nothing(x):
    pass

# PID 클래스 정의
class PID():
    def __init__(self, kp, ki, kd):
        self.kp = kp  # 비례 이득 설정
        self.ki = ki  # 적분 이득 설정
        self.kd = kd  # 미분 이득 설정
        self.p_error = 0.0  # 이전 비례 오차 초기화
        self.i_error = 0.0  # 적분 오차 초기화
        self.d_error = 0.0  # 미분 오차 초기화

    def pid_control(self, cte):
        self.d_error = cte - self.p_error  # 미분 오차 계산
        self.p_error = cte  # 비례 오차 갱신
        self.i_error += cte  # 적분 오차 갱신

        # PID 제어 계산
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error



class XycarPlanner:
    def __init__(self):
        rospy.init_node('xycar_planner', anonymous=True)  # ROS 노드 초기화
        
        try:
            # 카메라와 IMU 데이터 구독
            rospy.Subscriber("/xycar_motor_lane", xycar_motor, self.ctrlLaneCB)
            rospy.Subscriber("/xycar_motor_static", xycar_motor, self.ctrlStaticCB)
            rospy.Subscriber("/traffic_light", Int64, self.trafficLightCB)
            rospy.Subscriber("/raw_obstacles_static", CarObstacles, self.obstacleCB)
            rospy.Subscriber('/rubbercone_waypoints', Waypoint,  self.ctrlRubberconeCB)


            # 모터 제어 명령과 현재 속도 퍼블리셔 설정
            self.ctrl_cmd_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
            self.mode_pub = rospy.Publisher('/mode', String, queue_size=1)


            self.bridge = CvBridge()  # CV-Bridge 초기화


            self.steer = 0.0  # 조향각 초기화
            self.motor = 0.0  # 모터 속도 초기화
            self.traffic_light = 1  # 신호등 상태 초기화##############
            self.ctrl_cmd_msg = XycarMotor()

            self.ctrl_cmd    = XycarMotor()
            self.ctrl_lane = xycar_motor()  # 모터 제어 메시지 초기화
            self.ctrl_static = xycar_motor()
            self.ctrl_rubbercone = xycar_motor()
            self.ctrl_ar = xycar_motor()

            self.static_mode_flag = False
            self.lane_mode_flag = False

            self.pid = PID(0.7, 0.0008, 0.15)

            self.obstacles = []

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

            rate = rospy.Rate(30)  # 루프 주기 설정
            while not rospy.is_shutdown():  # ROS 노드가 종료될 때까지 반복

                # MODE 판별
                # 루프 내부에 속도 제어
                
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
                elif self.static_mode_flag:
                    # 정적 장애물 회피 모드
                    speed = self.ctrl_static.speed
                    steer = self.ctrl_static.angle
                    mode_str = "StaticObs"
                else:
                    # 차선 주행 모드
                    speed = self.ctrl_lane.speed
                    steer = self.ctrl_lane.angle
                    mode_str = "LaneFollowing"

                # MODE에 따른 motor, steer 설정
                if self.traffic_light == 1:
                    speed = 0
                elif self.traffic_light == 2:
                    pass  # 그대로 유지

                # 퍼블리시
                self.ctrl_cmd.speed = speed
                self.ctrl_cmd.angle = steer
                self.ctrl_cmd_pub.publish(self.ctrl_cmd)
                self.mode_pub.publish(mode_str)
                rospy.loginfo("[Mode:%s] speed=%.1f angle=%.1f", mode_str, speed, steer)

                rate.sleep()
                
        finally:
            cv2.destroyAllWindows()  # 창 닫기

        
    def publishCtrlCmd(self, motor_msg, servo_msg):
        self.ctrl_cmd_msg.speed = motor_msg  # 모터 속도 설정
        self.ctrl_cmd_msg.angle = servo_msg  # 조향각 설정
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)  # 명령 퍼블리시

    def ctrlLaneCB(self, msg):
        self.ctrl_lane.speed = msg.speed
        self.ctrl_lane.angle = msg.angle
        self.lane_mode_flag = msg.flag

    def ctrlStaticCB(self, msg):
        self.ctrl_static.speed = msg.speed
        self.ctrl_static.angle = msg.angle
        self.static_mode_flag = msg.flag
##########################################        
    def trafficLightCB(self, msg):
        self.traffic_light= msg.data

################################################

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
        self.obstacles = []
        for circle in msg.circles:
            x = circle.center.x
            y = circle.center.y
            distance = (x**2 + y**2) ** 0.5  # 유클리드 거리 계산
            obstacle = Obstacle(x, y, distance)
            self.obstacles.append(obstacle)
        
        self.obstacles.sort(key=lambda obs: obs.distance)

        if len(self.obstacles) > 0:
            self.closest_obstacle = self.obstacles[0]
        else:
            self.closest_obstacle = Obstacle()

    # def ctrlRubberconeCB(self, msg):
    #     self.ctrl_rubbercone.speed = msg.speed
    #     self.ctrl_rubbercone.angle = msg.angle
    #     self.rubbercone_mode_flag = msg.flag

    # def ctrlARCB(self, msg):
    #     self.ctrl_ar.speed = msg.speed
    #     self.ctrl_ar.angle = msg.angle
    #     self.ar_mode_flag = msg.flag



if __name__ == '__main__':
    try:
        autopilot_control = XycarPlanner()  # AutopilotControl 객체 생성
    except rospy.ROSInterruptException:
        pass  # 예외 발생 시 무시하고 종료