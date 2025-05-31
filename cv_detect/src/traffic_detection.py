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
threshold_pixel_min = 5000
isgreen = False
class TrafficDetection:
    def __init__(self):
        rospy.init_node('traffic_detection', anonymous=True)
        self.isgreen = False
        try:
            self.bridge = CvBridge()

            rospy.Subscriber("/usb_cam/image_raw", Image, self.cameraCB)
            self.traffic_light_pub = rospy.Publisher("/traffic_light", Int64, queue_size=1)

            
            self.cv_image = None
            rate = rospy.Rate(30)
            while not rospy.is_shutdown():
                if self.cv_image is not None:
                    self.detect_traffic_light(self.cv_image)
                    cv2.waitKey(1)
                rate.sleep()

        finally:
            cv2.destroyAllWindows()

    def nothing(self, x):
        pass

    def cameraCB(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(e)


    def filter_circular_contours(self, contours, circularity_threshold=0.2, min_area=300):
        filtered_contours = []
        filtered_contours_circularity = []
        filtered_contours_area = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > circularity_threshold and area > min_area:  # 원형 비율 임계값과 최소 면적
                filtered_contours.append(contour)
                filtered_contours_circularity.append(circularity)
                filtered_contours_area.append(area)

        return filtered_contours, filtered_contours_area, filtered_contours_circularity
    
    def get_contour_centers(self, contours):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        return centers

    def detect_traffic_light(self, image):
        cv2.imshow("src", image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        h, s, v = cv2.split(hsv_image)


        # Red trackbar values
        h_min_red1 = 161
        h_max_red1 = 179
        s_min_red1 = 100
        s_max_red1 = 255
        v_min_red1 = 100
        v_max_red1 = 255

        # Green trackbar values
        h_min_green1 = 41
        h_max_green1 = 84
        s_min_green1 = 100
        s_max_green1 = 255
        v_min_green1 = 100
        v_max_green1 = 255
        lower_red1 = np.array([h_min_red1, s_min_red1, v_min_red1])
        upper_red1 = np.array([h_max_red1, s_max_red1, v_max_red1])
        lower_green1 = np.array([h_min_green1, s_min_green1, v_min_green1])
        upper_green1 = np.array([h_max_green1, s_max_green1, v_max_green1])

        red_mask = cv2.inRange(hsv_image, lower_red1, upper_red1)
        green_mask = cv2.inRange(hsv_image, lower_green1, upper_green1)

        # 윤곽선 찾기
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        red_filtered_contours, red_contours_area, red_contours_circularity = self.filter_circular_contours(red_contours)
        green_filtered_contours, green_contours_area, green_contours_circularity= self.filter_circular_contours(green_contours)

        # 빈 이미지 생성 (배경 검은색)
        red_result = np.zeros_like(red_mask)
        green_result = np.zeros_like(green_mask)

        # 필터링된 윤곽선만 그림
        cv2.drawContours(red_result, red_filtered_contours, -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(green_result, green_filtered_contours, -1, (255), thickness=cv2.FILLED)

        red_pixel_counts = np.count_nonzero(red_result)
        green_pixel_counts = np.count_nonzero(green_result)

        
        # 무게중심 계산
        red_centers = self.get_contour_centers(red_filtered_contours)
        green_centers = self.get_contour_centers(green_filtered_contours)

        rospy.loginfo(f"Red Centers: {red_centers}")
        rospy.loginfo(f"Green Centers: {green_centers}")
     
        # 중심 좌표 배열로 변환
        red_centers_flattened = [coord for center in red_centers for coord in center]
        green_centers_flattened = [coord for center in green_centers for coord in center]
##########################################################
        red_on = red_pixel_counts > threshold_pixel_min
        green_on = green_pixel_counts > threshold_pixel_min
        
        if len(green_centers) > 0:
            green_valid = any(center[1] <= 80 for center in green_centers)
            rospy.loginfo(f"Green centers Y: {[c[1] for c in green_centers]}")
        else:
            green_valid = False

        if len(red_centers) > 0:
            red_valid = any(center[1] <= 80 for center in red_centers)
            rospy.loginfo(f"Red centers Y: {[c[1] for c in red_centers]}")
        else:
            red_valid = False

        if green_on and not red_on and green_valid:
            state = 2
            self.isgreen = True
            rospy.loginfo(f"green")
        elif red_on and not green_on and red_valid:
            state = 1
            self.isgreen = False
            rospy.loginfo(f"red")
        else:
            # 이전 상태 유지 판단
            if self.isgreen:
                state = 2
                rospy.loginfo(f"go")
            else:
                state = 1
                rospy.loginfo(f"stop")
        msg = Int64()
        msg.data = state
        self.traffic_light_pub.publish(msg)

if __name__ == '__main__':
    try:
        traffic_detection_node = TrafficDetection()
    except rospy.ROSInterruptException:
        pass