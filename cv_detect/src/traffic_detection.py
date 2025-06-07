#!/usr/bin/env python3
from __future__ import print_function
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import numpy as np
from shared_function import *

from std_msgs.msg import Int64MultiArray
from std_msgs.msg import Int64
threshold_pixel_min = 5000 # 신호등으로 인식할 최소 픽셀 수
isgreen = False # 현재 신호등이 초록색인지 여부를 저장하는 전역 변수 (클래스 멤버로 이동하는 것이 좋음)

# TrafficDetection 클래스: 카메라 이미지를 받아 신호등을 감지하고 그 상태를 발행합니다.
class TrafficDetection:
    def __init__(self):
        
        rospy.init_node('traffic_detection', anonymous=True)
        self.isgreen = False # 현재 신호등 상태 (초록불 여부)
        try:
            self.bridge = CvBridge() 

            # ROS 구독자 설정: "/usb_cam/image_raw" 토픽에서 이미지 메시지를 받아 self.cameraCB 콜백 함수 호출
            rospy.Subscriber("/usb_cam/image_raw", Image, self.cameraCB)
            # ROS 발행자 설정: "/traffic_light" 토픽으로 신호등 상태 (Int64)를 발행
            self.traffic_light_pub = rospy.Publisher("/traffic_light", Int64, queue_size=1)

            
            self.cv_image = None # 수신된 이미지를 저장할 변수
            rate = rospy.Rate(30) 
            
            while not rospy.is_shutdown():
                if self.cv_image is not None:
                    # 이미지가 수신되면 신호등 감지 함수 호출
                    self.detect_traffic_light(self.cv_image)
                    cv2.waitKey(1) 
                rate.sleep() 

        finally:
            cv2.destroyAllWindows() 

    def nothing(self, x):
        pass

    # cameraCB 메소드: 이미지 메시지를 수신했을 때 호출되는 콜백 함수
    def cameraCB(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV bgr8 형식으로 변환
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(e) 


    # filter_circular_contours 메소드: 윤곽선들 중에서 원형에 가깝고 특정 크기 이상인 것들만 필터링합니다.
    def filter_circular_contours(self, contours, circularity_threshold=0.2, min_area=300):
        filtered_contours = []
        filtered_contours_circularity = []
        filtered_contours_area = []
        for contour in contours:
            area = cv2.contourArea(contour) # 윤곽선 면적 계산
            perimeter = cv2.arcLength(contour, True) # 윤곽선 둘레 계산
            if perimeter == 0:
                continue
            # 원형도 계산: (4 * pi * 면적) / (둘레^2)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            # 원형도와 면적이 임계값 이상인 경우 필터링된 윤곽선 목록에 추가
            if circularity > circularity_threshold and area > min_area: 
                filtered_contours.append(contour)
                filtered_contours_circularity.append(circularity)
                filtered_contours_area.append(area)

        return filtered_contours, filtered_contours_area, filtered_contours_circularity
    
    # get_contour_centers 메소드: 윤곽선들의 중심 좌표를 계산합니다.
    def get_contour_centers(self, contours):
        centers = []
        for contour in contours:
            M = cv2.moments(contour) # 윤곽선의 모멘트 계산
            if M["m00"] != 0:
                # 모멘트를 사용하여 중심 좌표 (cX, cY) 계산
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        return centers

    # detect_traffic_light 메소드: 입력 이미지에서 신호등을 감지하고 상태를 결정합니다.
    def detect_traffic_light(self, image):
        # cv2.imshow("src", image) # 원본 이미지 표시 (디버깅용)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR 이미지를 HSV 이미지로 변환
        
        h, s, v = cv2.split(hsv_image) # HSV 이미지를 H, S, V 채널로 분리


        # 빨간색 신호등 감지를 위한 HSV 색상 범위 정의
        h_min_red1 = 161
        h_max_red1 = 179
        s_min_red1 = 100
        s_max_red1 = 255
        v_min_red1 = 100
        v_max_red1 = 255

        # 초록색 신호등 감지를 위한 HSV 색상 범위 정의
        h_min_green1 = 41
        h_max_green1 = 84
        s_min_green1 = 100
        s_max_green1 = 255
        v_min_green1 = 100
        v_max_green1 = 255

        # 정의된 HSV 범위에 따라 빨간색과 초록색 마스크 생성
        lower_red1 = np.array([h_min_red1, s_min_red1, v_min_red1])
        upper_red1 = np.array([h_max_red1, s_max_red1, v_max_red1])
        lower_green1 = np.array([h_min_green1, s_min_green1, v_min_green1])
        upper_green1 = np.array([h_max_green1, s_max_green1, v_max_green1])

        red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1)
        green_mask = cv2.inRange(hsv_image, lower_green1, upper_green1)

        # 생성된 마스크에서 윤곽선 찾기
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 찾은 윤곽선들을 원형도와 면적 기준으로 필터링
        red_filtered_contours, red_contours_area, red_contours_circularity = self.filter_circular_contours(red_contours)
        green_filtered_contours, green_contours_area, green_contours_circularity= self.filter_circular_contours(green_contours)

        # 필터링된 윤곽선을 시각화하기 위한 빈 이미지 생성
        red_result = np.zeros_like(red_mask)
        green_result = np.zeros_like(green_mask)

        # 필터링된 윤곽선을 결과 이미지에 그리기
        cv2.drawContours(red_result, red_filtered_contours, -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(green_result, green_filtered_contours, -1, (255), thickness=cv2.FILLED)

        # 결과 이미지에서 각 색상의 픽셀 수 계산
        red_pixel_counts = np.count_nonzero(red_result)
        green_pixel_counts = np.count_nonzero(green_result)

        
        # 필터링된 윤곽선들의 중심 좌표 계산
        red_centers = self.get_contour_centers(red_filtered_contours)
        green_centers = self.get_contour_centers(green_filtered_contours)

        # rospy.loginfo(f"Red Centers: {red_centers}")
        # rospy.loginfo(f"Green Centers: {green_centers}") 
     
        
        red_centers_flattened = [coord for center in red_centers for coord in center]
        green_centers_flattened = [coord for center in green_centers for coord in center]

        # 각 색상의 픽셀 수가 임계값 이상이면 해당 색상 신호등이 켜진 것으로 간주
        red_on = red_pixel_counts > threshold_pixel_min
        green_on = green_pixel_counts > threshold_pixel_min
        
        # 감지된 신호등의 y 좌표가 특정 범위 내에 있는지 확인 (이미지 상단에 위치하는지)
        if len(green_centers) > 0:
            green_valid = any(center[1] <= 80 for center in green_centers) # 초록불 중심의 y좌표가 80 이하인지
            # rospy.loginfo(f"Green centers Y: {[c[1] for c in green_centers]}")
        else:
            green_valid = False

        if len(red_centers) > 0:
            red_valid = any(center[1] <= 80 for center in red_centers) # 빨간불 중심의 y좌표가 80 이하인지
            # rospy.loginfo(f"Red centers Y: {[c[1] for c in red_centers]}")
        else:
            red_valid = False

        # 신호등 상태 결정 로직
        if green_on and not red_on and green_valid: # 초록불만 켜지고 유효한 위치에 있을 경우
            state = 2 # 상태: 초록불
            self.isgreen = True
            # rospy.loginfo(f"green")
        elif red_on and not green_on and red_valid: # 빨간불만 켜지고 유효한 위치에 있을 경우
            state = 1 # 상태: 빨간불
            self.isgreen = False
            # rospy.loginfo(f"red")
        else: # 그 외의 경우 (둘 다 켜지거나, 둘 다 꺼지거나, 유효하지 않은 위치 등)
            # 이전 상태를 유지
            if self.isgreen:
                state = 2 # 이전이 초록불이었으면 초록불 유지
                # rospy.loginfo(f"go")
            else:
                state = 1 # 이전이 빨간불이었으면 빨간불 유지
                # rospy.loginfo(f"stop")
        
        # 결정된 신호등 상태를 ROS 메시지로 발행
        msg = Int64()
        msg.data = state
        self.traffic_light_pub.publish(msg)

if __name__ == '__main__':
    try:
        
        traffic_detection_node = TrafficDetection()
    except rospy.ROSInterruptException:
        pass