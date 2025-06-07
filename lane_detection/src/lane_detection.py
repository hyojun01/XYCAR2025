#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from xycar_msgs.msg import xycar_motor 
import cv2 # OpenCV 라이브러리
from slidewindow_both_lane import SlideWindow # 슬라이딩 윈도우 방식의 차선 감지 클래스
from cv_bridge import CvBridge, CvBridgeError 
import rospy 
from sensor_msgs.msg import Image 
from shared_function import *

# PID 제어기 클래스
class PID():
    def __init__(self, kp, ki, kd):
        # PID 제어 파라미터 초기화
        self.kp = kp   
        self.ki = ki  
        self.kd = kd  
        # 오차 값 초기화
        self.p_error = 0.0  
        self.i_error = 0.0  
        self.d_error = 0.0  

    # PID 제어 값을 계산하는 메소드
    def pid_control(self, cte): 
        self.d_error = cte - self.p_error  
        self.p_error = cte  
        self.i_error += cte  
        
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error

# 차선 감지 및 차량 제어를 위한 메인 클래스
class LaneDetection(object):
    def __init__(self):
        rospy.init_node('lane_detection', anonymous=True)  
        
        try:
            # 이미지 토픽 구독 설정
            rospy.Subscriber("/usb_cam/image_raw", Image, self.cameraCB)

            # 차량 제어 명령 발행 설정
            self.ctrl_cmd_pub = rospy.Publisher('/xycar_motor_lane', xycar_motor, queue_size=1)
            
            self.bridge = CvBridge()  
            self.ctrl_cmd_msg = xycar_motor()  
            self.slidewindow = SlideWindow()  

            # 원근 변환(Bird's-eye view)을 위한 원본 이미지 상의 4개 점 좌표
            src_pts = np.float32([[ 1, 479],    
                      [220, 260],    
                      [420, 260],    
                      [638, 479]])   
            # 원근 변환 후 결과 이미지 상의 4개 점 좌표
            dst_pts = np.float32([[ 70, 479],
                      [ 70,   0],
                      [570,   0],
                      [570, 479]])

            self.version = rospy.get_param('~version', 'safe')

            # 조향각 및 모터 속도 초기화
            self.steer = 0.0  
            self.motor = 0.0  
                       
            # 주행 모드('fast' 또는 'safe')에 따른 PID 파라미터 설정
            if self.version == 'fast':
                self.pid = PID(0.78, 0.0005, 0.405) 
            else:             
                self.pid = PID(0.78, 0.0005, 0.405) 

            self.cv_image = None  # 수신된 카메라 이미지를 저장할 변수
                       
            rate = rospy.Rate(30) 

            while not rospy.is_shutdown():  
                if self.cv_image is not None:  
                    # 1. 이미지 전처리 (그레이스케일, 블러, 이진화 등)
                    binary = process_image(self.cv_image)                    
                    # 2. 원근 변환 (Bird's-eye view)
                    warped = warper(binary, src_pts, dst_pts)                    
                    # 3. 차선 감지를 위한 관심 영역(ROI) 설정
                    cropped_img = roi_for_lane(warped)                    
                    # 4. 슬라이딩 윈도우를 이용한 차선 감지 및 차선 중심 x좌표 계산
                    out_img, x_location, _ = self.slidewindow.slidewindow(cropped_img)  

                    # 차선 중심 x좌표가 감지되지 않았을 경우 이전 값 사용
                    if x_location == None:  
                        x_location = last_x_location  # last_x_location 변수가 이전에 정의되어 있어야 함
                    else:
                        last_x_location = x_location  # 현재 감지된 x_location을 다음을 위해 저장
                                       
                    # PID 제어기를 사용하여 조향각 계산 (이미지 중앙(320)과의 차이를 오차로 사용)
                    self.steer = round(self.pid.pid_control(x_location - 320))  
                    self.steer = self.steer * 0.5  

                    if self.version == 'fast':
                        self.motor = 60 
                    else:
                        self.motor = 60
                    
                    # 계산된 모터 속도와 조향각을 차량에 전달
                    self.publishCtrlCmd(self.motor, self.steer)  

                    cv2.imshow('out_img', out_img)
            
                    cv2.waitKey(1)  

                rate.sleep()  
                
        finally:
            cv2.destroyAllWindows()  
        
    # 모터 속도와 조향각을 ROS 토픽으로 발행하는 메소드
    def publishCtrlCmd(self, motor_msg, servo_msg):
        self.ctrl_cmd_msg.speed = motor_msg  
        self.ctrl_cmd_msg.angle = servo_msg 
        self.ctrl_cmd_msg.flag = True 
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg) 
        
    # 카메라 이미지 토픽 콜백 함수
    def cameraCB(self, msg):
        try:
            # ROS Image 메시지를 OpenCV 이미지(bgr8)로 변환
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  
        except CvBridgeError as e:
            rospy.logwarn(e)

if __name__ == '__main__':
    try:
        autopilot_control = LaneDetection()  
    except rospy.ROSInterruptException:
        pass