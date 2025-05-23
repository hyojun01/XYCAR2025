#!/usr/bin/env python3

from __future__ import print_function
import cv2
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import numpy as np
from shared_function import *

from std_msgs.msg import Int64MultiArray

class TrafficDetection:
    def __init__(self):
        rospy.init_node('traffic_detection', anonymous=True)

        try:
            self.bridge = CvBridge()

            rospy.Subscriber("/usb_cam/image_raw", Image, self.cameraCB)
            self.traffic_light_pub = rospy.Publisher("/traffic_light", Int64MultiArray, queue_size=1)
            self.red_center_pub = rospy.Publisher("/red_center", Int64MultiArray, queue_size=1)
            self.green_center_pub = rospy.Publisher("/green_center", Int64MultiArray, queue_size=1)


            self.cv_image = None

            # 트랙바 윈도우 생성
            cv2.namedWindow('Red Trackbars')
            cv2.createTrackbar('H_min_red1', 'Red Trackbars', 0, 179, self.nothing)
            cv2.createTrackbar('H_max_red1', 'Red Trackbars', 179, 179, self.nothing)
            cv2.createTrackbar('S_min_red1', 'Red Trackbars', 0, 255, self.nothing)
            cv2.createTrackbar('S_max_red1', 'Red Trackbars', 10, 255, self.nothing)
            cv2.createTrackbar('V_min_red1', 'Red Trackbars', 180, 255, self.nothing)
            cv2.createTrackbar('V_max_red1', 'Red Trackbars', 240, 255, self.nothing)
            
            cv2.namedWindow('Green Trackbars')
            cv2.createTrackbar('H_min_green1', 'Green Trackbars', 68, 179, self.nothing)
            cv2.createTrackbar('H_max_green1', 'Green Trackbars', 91, 179, self.nothing)
            cv2.createTrackbar('S_min_green1', 'Green Trackbars', 0, 255, self.nothing)
            cv2.createTrackbar('S_max_green1', 'Green Trackbars', 180, 255, self.nothing)
            cv2.createTrackbar('V_min_green1', 'Green Trackbars', 160, 255, self.nothing)
            cv2.createTrackbar('V_max_green1', 'Green Trackbars', 255, 255, self.nothing)

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


    def filter_circular_contours(self, contours, circularity_threshold=0.7, min_area=150):
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
        h_min_red1 = cv2.getTrackbarPos('H_min_red1', 'Red Trackbars')
        h_max_red1 = cv2.getTrackbarPos('H_max_red1', 'Red Trackbars')
        s_min_red1 = cv2.getTrackbarPos('S_min_red1', 'Red Trackbars')
        s_max_red1 = cv2.getTrackbarPos('S_max_red1', 'Red Trackbars')
        v_min_red1 = cv2.getTrackbarPos('V_min_red1', 'Red Trackbars')
        v_max_red1 = cv2.getTrackbarPos('V_max_red1', 'Red Trackbars')

        # Green trackbar values
        h_min_green1 = cv2.getTrackbarPos('H_min_green1', 'Green Trackbars')
        h_max_green1 = cv2.getTrackbarPos('H_max_green1', 'Green Trackbars')
        s_min_green1 = cv2.getTrackbarPos('S_min_green1', 'Green Trackbars')
        s_max_green1 = cv2.getTrackbarPos('S_max_green1', 'Green Trackbars')
        v_min_green1 = cv2.getTrackbarPos('V_min_green1', 'Green Trackbars')
        v_max_green1 = cv2.getTrackbarPos('V_max_green1', 'Green Trackbars')

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

        # rospy.loginfo(f"red_contours_area: {red_contours_area}")
        # rospy.loginfo(f"red_contours_circularity: {red_contours_circularity}")
        # rospy.loginfo("")

        # rospy.loginfo(f"green_contours_area: {green_contours_area}")
        # rospy.loginfo(f"green_contours_circularity: {green_contours_circularity}")
        # rospy.loginfo("")

        # 빈 이미지 생성 (배경 검은색)
        red_result = np.zeros_like(red_mask)
        green_result = np.zeros_like(green_mask)

        # 필터링된 윤곽선만 그림
        cv2.drawContours(red_result, red_filtered_contours, -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(green_result, green_filtered_contours, -1, (255), thickness=cv2.FILLED)

        red_pixel_counts = np.count_nonzero(red_result)
        green_pixel_counts = np.count_nonzero(green_result)

        # rospy.loginfo(f'red_pixel_counts: {red_pixel_counts}')
        # rospy.loginfo(f'green_pixel_counts: {green_pixel_counts}')

        cv2.imshow('red_mask', red_mask)
        cv2.imshow('green_mask', green_mask)
        cv2.imshow('red_result', red_result)
        cv2.imshow('green_result', green_result)
        cv2.imshow('h', h)
        cv2.imshow('s', s)
        cv2.imshow('v', v)

        # 무게중심 계산
        red_centers = self.get_contour_centers(red_filtered_contours)
        green_centers = self.get_contour_centers(green_filtered_contours)



        # 중심 좌표 배열로 변환
        red_centers_flattened = [coord for center in red_centers for coord in center]
        green_centers_flattened = [coord for center in green_centers for coord in center]


        msg = Int64MultiArray()
        msg.data = [red_pixel_counts, green_pixel_counts]

        self.traffic_light_pub.publish(msg)

        red_centers_msg = Int64MultiArray()
        red_centers_msg.data = red_centers_flattened
        self.red_center_pub.publish(red_centers_msg)
        
        green_centers_msg = Int64MultiArray()
        green_centers_msg.data = green_centers_flattened
        self.green_center_pub.publish(green_centers_msg)

if __name__ == '__main__':
    try:
        traffic_detection_node = TrafficDetection()
    except rospy.ROSInterruptException:
        pass