#! /usr/bin/env python3


from __future__ import print_function
from xycar_msgs.msg import xycar_motor, XycarMotor  
from sensor_msgs.msg import Imu 
from std_msgs.msg import Float32, String, Int64 
from obstacle_detector.msg import Waypoint, Obstacles, CarObstacles # 장애물 및 웨이포인트 관련 메시지
from math import radians, pi 
import cv2 
import numpy as np 
import math
from cv_bridge import CvBridge, CvBridgeError  
import time, math 
import rospy  
from sensor_msgs.msg import Image, CompressedImage  
import tf 


import tkinter as tk 

# 장애물 정보를 저장하기 위한 클래스
class Obstacle:
    def __init__(self, x=None, y=None, distance=None):
        self.x = x 
        self.y = y 
        self.distance = distance 

def nothing(x):
    pass

# PID 제어기 클래스
class PID():
    def __init__(self, kp, ki, kd):
        self.kp = kp  
        self.ki = ki  
        self.kd = kd  
        self.p_error = 0.0 
        self.i_error = 0.0  
        self.d_error = 0.0 

    # PID 제어 연산을 수행하는 메소드
    def pid_control(self, cte): 
        self.d_error = cte - self.p_error  
        self.p_error = cte  
        self.i_error += cte  

        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error


# Xycar의 주행 계획 및 제어를 담당하는 메인 클래스
class XycarPlanner:
    def __init__(self):
        
        rospy.init_node('xycar_planner', anonymous=True)  
        
        try:
           
            rospy.Subscriber("/xycar_motor_lane", xycar_motor, self.ctrlLaneCB) # 차선 주행 제어 명령 구독
            rospy.Subscriber("/xycar_motor_static", xycar_motor, self.ctrlStaticCB) # 정적 장애물 회피 제어 명령 구독
            rospy.Subscriber("/traffic_light", Int64, self.trafficLightCB) # 신호등 상태 구독
            rospy.Subscriber("/raw_obstacles_static", CarObstacles, self.obstacleCB) # 정적 장애물 정보 구독
            rospy.Subscriber('/rubbercone_waypoints', Waypoint,  self.ctrlRubberconeCB) # 라바콘 회피 웨이포인트 구독

           
            self.ctrl_cmd_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1) # 최종 차량 제어 명령 발행
            self.mode_pub = rospy.Publisher('/mode', String, queue_size=1) # 현재 주행 모드 발행

            self.bridge = CvBridge()  

            # 주행 제어 관련 변수 초기화
            self.steer = 0.0  
            self.motor = 0.0  
            self.traffic_light = 1  # 신호등 상태 (1: 정지, 2: 직진)
            self.ctrl_cmd_msg = XycarMotor() 

            # 각 모드별 제어 명령 저장 변수
            self.ctrl_cmd    = XycarMotor() # 최종 제어 명령
            self.ctrl_lane = xycar_motor()  # 차선 주행 제어 명령
            self.ctrl_static = xycar_motor() # 정적 장애물 회피 제어 명령
            self.ctrl_rubbercone = xycar_motor() # 라바콘 회피 제어 명령
            self.ctrl_ar = xycar_motor() # AR 주차 제어 명령 (현재 코드에서는 사용되지 않음)

            # 주행 모드 플래그
            self.static_mode_flag = False # 정적 장애물 회피 모드 활성화 여부
            self.lane_mode_flag = False # 차선 주행 모드 활성화 여부 (현재 코드에서는 직접적인 제어 흐름에 사용되지 않음)

            # PID 제어기 인스턴스 생성 (라바콘 회피 등에 사용될 수 있음)
            self.pid = PID(0.7, 0.0008, 0.15)

            self.obstacles = [] # 감지된 장애물 리스트

            # 라바콘 회피 모드 관련 변수 초기화
            self.rubber_mode        = False # 라바콘 회피 모드 활성화 여부
            self.detected_frames    = 0 # 라바콘 웨이포인트 연속 감지 프레임 수
            self.lost_frames        = 0 # 라바콘 웨이포인트 연속 손실 프레임 수
            self.prev_wp_cnt        = 0 # 이전 웨이포인트 개수
            self.wp_cnt             = 0 # 현재 웨이포인트 개수
            self.latest_wp          = None   # 가장 최근에 수신된 라바콘 웨이포인트 (x, y)
            self.target_wp          = None   # 현재 목표로 하는 라바콘 웨이포인트
            self.rubber_start_time  = None # 라바콘 모드 시작 시간
            self.no_points_frames   = 0  # 웨이포인트가 없는 프레임 수 (현재 코드에서는 사용되지 않음)

            # 라바콘 회피 관련 파라미터 (ROS 파라미터 서버 또는 기본값 사용)
            self.rcon_speed   = rospy.get_param('~rcon_speed', 10.0) # 라바콘 모드 주행 속도
            self.entry_thresh = 3  # 라바콘 모드 진입을 위한 최소 연속 감지 프레임
            self.exit_thresh  = 5  # 라바콘 모드 이탈을 위한 최대 연속 손실 프레임
            self.reach_dist   = 0.3 # 웨이포인트 도달 판단 거리 (현재 코드에서는 사용되지 않음)
            self.max_duration = 30.0 # 라바콘 모드 최대 지속 시간 (현재 코드에서는 사용되지 않음)

            rate = rospy.Rate(30)  # 루프 실행 빈도 (30Hz)
            # 메인 제어 루프
            while not rospy.is_shutdown():  
                current_wp_available = (self.wp_cnt >= 1 and self.latest_wp is not None) # 현재 라바콘 웨이포인트 유효성 검사

                # 라바콘 모드 진입/유지/이탈 로직
                if not self.rubber_mode: # 현재 라바콘 모드가 아닐 경우
                    if current_wp_available:
                        self.detected_frames += 1 # 유효 웨이포인트 감지 시 카운트 증가
                    else:
                        self.detected_frames = 0 # 감지 못하면 초기화
                    
                    # 연속 감지 프레임이 진입 임계값을 넘으면 라바콘 모드 시작
                    if  self.detected_frames >= self.entry_thresh:
                        self.rubber_mode       = True
                        self.rubber_start_time = rospy.Time.now()
                        self.target_wp         = self.latest_wp 
                        self.lost_frames = 0 
                else: # 현재 라바콘 모드일 경우
                    if current_wp_available:
                        self.target_wp = self.latest_wp # 유효 웨이포인트 감지 시 목표 업데이트
                        self.lost_frames = 0 
                    else:
                        self.lost_frames += 1 # 감지 못하면 손실 카운트 증가
                        rospy.loginfo(f"Rubbercone Mode: No new waypoints, lost_frames = {self.lost_frames}/{self.exit_thresh}")

                    # 연속 손실 프레임이 이탈 임계값을 넘으면 라바콘 모드 종료
                    if self.lost_frames >= self.exit_thresh:
                        rospy.loginfo(f"=== Exit Rubbercone Mode (Lost Points: {self.lost_frames} frames) ===")
                        self.rubber_mode = False
                        # 라바콘 모드 관련 변수 초기화
                        self.lost_frames = 0       
                        self.detected_frames = 0   
                        self.target_wp = None      
                        self.rubber_start_time = None 

                        # 라바콘 모드 종료 후 정적 장애물 회피 모드로 전환하기 위한 임시 제어 명령 발행
                        speed = 30 
                        steer = 48 # 특정 조향각으로 설정 (우회전 또는 좌회전 유도 가능성)
                        mode_str = "StaticObs" # 모드를 정적 장애물 회피로 설정

                        for _ in range(33):  # 일정 시간 동안 해당 제어 명령 유지
                            self.ctrl_cmd.speed = speed
                            self.ctrl_cmd.angle = steer
                            self.ctrl_cmd_pub.publish(self.ctrl_cmd)
                            self.mode_pub.publish(mode_str)
                            rate.sleep()    
                        continue # 다음 루프 반복으로 바로 넘어감

                # 현재 주행 모드에 따른 속도 및 조향각 결정
                if self.static_mode_flag: # 정적 장애물 회피 모드가 활성화된 경우
                    speed = self.ctrl_static.speed
                    steer = self.ctrl_static.angle
                    mode_str = "StaticObs"
                elif self.rubber_mode and self.target_wp: # 라바콘 모드이고 목표 웨이포인트가 있는 경우
                    x, y = self.target_wp
                    ang = 2.2 * math.degrees(math.atan2(y, x)) # 목표 웨이포인트 방향각 계산
                    speed, steer = self.rcon_speed, -ang # 라바콘 모드 속도 및 계산된 조향각 사용
                    mode_str = "Rubbercone"
                else: # 기본 모드 (차선 주행)
                    speed = self.ctrl_lane.speed
                    steer = self.ctrl_lane.angle
                    mode_str = "LaneFollowing"

                # 신호등 상태에 따른 속도 제어
                if self.traffic_light == 1: # 빨간불이면 정지
                    speed = 0
                elif self.traffic_light == 2: # 초록불이면 기존 속도 유지
                    pass  

                # 최종 제어 명령 설정 및 발행
                self.ctrl_cmd.speed = speed
                self.ctrl_cmd.angle = steer
                self.ctrl_cmd_pub.publish(self.ctrl_cmd)
                self.mode_pub.publish(mode_str) # 현재 모드 발행

                rate.sleep() # 루프 주기 유지
                
        finally:
            cv2.destroyAllWindows()  

        
    # 최종 제어 명령을 발행하는 함수
    def publishCtrlCmd(self, motor_msg, servo_msg):
        self.ctrl_cmd_msg.speed = motor_msg  
        self.ctrl_cmd_msg.angle = servo_msg  
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)  

    # 차선 주행 제어 명령 콜백 함수
    def ctrlLaneCB(self, msg):
        self.ctrl_lane.speed = msg.speed
        self.ctrl_lane.angle = msg.angle
        self.lane_mode_flag = msg.flag # 차선 주행 모드 플래그 업데이트

    # 정적 장애물 회피 제어 명령 콜백 함수
    def ctrlStaticCB(self, msg):
        self.ctrl_static.speed = msg.speed
        self.ctrl_static.angle = msg.angle
        self.static_mode_flag = msg.flag # 정적 장애물 회피 모드 플래그 업데이트

    # 신호등 상태 콜백 함수
    def trafficLightCB(self, msg):
        self.traffic_light= msg.data # 신호등 상태 업데이트

    # 라바콘 웨이포인트 콜백 함수
    def ctrlRubberconeCB(self, msg):
        self.wp_cnt = msg.cnt # 수신된 웨이포인트 개수 업데이트
        if msg.cnt >= 1 and len(msg.x_arr) > 0 and len(msg.y_arr) > 0:
            self.latest_wp = (msg.x_arr[0], msg.y_arr[0]) # 첫 번째 웨이포인트를 최근 웨이포인트로 저장
        else:
            self.latest_wp = None # 웨이포인트가 없으면 None으로 설정

    # 정적 장애물 정보 콜백 함수
    def obstacleCB(self, msg):
        self.obstacles = [] # 장애물 리스트 초기화
        for circle in msg.circles: # 수신된 각 장애물(원형)에 대해
            x = circle.center.x
            y = circle.center.y
            distance = (x**2 + y**2) ** 0.5  # 원점으로부터의 거리 계산
            obstacle = Obstacle(x, y, distance) # Obstacle 객체 생성
            self.obstacles.append(obstacle) 
        
        # 장애물을 거리순으로 정렬
        self.obstacles.sort(key=lambda obs: obs.distance)

        # 가장 가까운 장애물 정보 저장
        if len(self.obstacles) > 0:
            self.closest_obstacle = self.obstacles[0]
        else:
            self.closest_obstacle = Obstacle()

if __name__ == '__main__':
    try:
        autopilot_control = XycarPlanner()  
    except rospy.ROSInterruptException:
        pass