#! /usr/bin/env python3

# 기본 Python3 인터프리터 설정

from __future__ import print_function

from xycar_msgs.msg import xycar_motor, XycarMotor  # xycar 모터 메시지 모듈 임포트
from sensor_msgs.msg import Imu  # IMU 데이터 메시지 모듈 임포트
from std_msgs.msg import Float32, String  # Float32 메시지 모듈 임포트

from math import radians, pi  # 각도를 라디안으로 변환하는 함수 임포트

import cv2  # OpenCV 라이브러리 임포트
import numpy as np  # NumPy 라이브러리 임포트
import math

from cv_bridge import CvBridge, CvBridgeError  # CV-Bridge 라이브러리 임포트

import rospy  # ROS 파이썬 라이브러리 임포트
from sensor_msgs.msg import Image, CompressedImage  # 이미지 데이터 메시지 모듈 임포트

from obstacle_detector.msg import Obstacles


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

            rospy.Subscriber("/raw_obstacles_static", Obstacles, self.obstacleCB)

            # 모터 제어 명령과 현재 속도 퍼블리셔 설정
            self.ctrl_cmd_pub = rospy.Publisher('/xycar_motor', XycarMotor, queue_size=1)
            self.mode_pub = rospy.Publisher('/mode', String, queue_size=1)


            self.bridge = CvBridge()  # CV-Bridge 초기화


            self.steer = 0.0  # 조향각 초기화
            self.motor = 0.0  # 모터 속도 초기화

            self.ctrl_cmd_msg = XycarMotor()

            self.ctrl_lane = xycar_motor()  # 모터 제어 메시지 초기화
            self.ctrl_static = xycar_motor()
            self.ctrl_rubbercone = xycar_motor()
            self.ctrl_ar = xycar_motor()

            self.static_mode_flag = False
            self.lane_mode_flag = False

            self.pid = PID(0.7, 0.0008, 0.15)

            self.obstacles = []

            rate = rospy.Rate(30)  # 루프 주기 설정
            while not rospy.is_shutdown():  # ROS 노드가 종료될 때까지 반복



                # MODE 판별
                if self.static_mode_flag == True:
                    self.motor = self.ctrl_static.speed
                    self.steer = self.ctrl_static.angle
                    print("Current mode : Obstacles")
                else:
                    self.motor = self.ctrl_lane.speed
                    self.steer = self.ctrl_lane.angle
                    print("Current mode : Lane")

                # MODE에 따른 motor, steer 설정


                if len(self.obstacles) > 0:
                    # 특정 roi에 인지가 들어오면 일단 감속
                    for obstacle in self.obstacles:
                        if (0 < obstacle.x < 2.0) and (-0.45 <= obstacle.y <= 0.45):
                            self.motor = 5
                            #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

                self.publishCtrlCmd(self.motor, self.steer)
                # print('self.lane_mode_flag', self.lane_mode_flag)

                cv2.waitKey(1)  # 키 입력 대기
                rate.sleep()  # 주기마다 대기
                
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