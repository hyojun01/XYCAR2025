#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan 
from obstacle_detector.msg import Waypoint # 사용자 정의 웨이포인트 메시지
from scipy.interpolate import CubicSpline # 스플라인 보간
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import cv2 

# LidarLeftOffsetExtractor 클래스: 라이다 데이터를 사용하여 좌측 장애물(라바콘)을 감지하고,
# 이를 기반으로 오프셋된 웨이포인트를 생성하여 발행합니다.
# 카메라 이미지를 사용하여 주황색 라바콘 감지 시에만 웨이포인트 생성을 활성화합니다.
class LidarLeftOffsetExtractor:
    def __init__(self):
        
        rospy.init_node('rubbercone_left_offset')
        
        self.min_range     = rospy.get_param('~min_range',    1.0) # 라이다 최소 인식 거리
        self.max_range     = rospy.get_param('~max_range',   35.0) # 라이다 최대 인식 거리
        self.left_min_deg  = rospy.get_param('~left_min_deg', 5.0)  # 좌측 감지 최소 각도
        self.left_max_deg  = rospy.get_param('~left_max_deg', 85.0) # 좌측 감지 최대 각도
        self.detect_dist   = rospy.get_param('~detect_dist',   10.0) # 장애물 감지 거리
        self.offset_dist   = rospy.get_param('~offset_dist',   1.6)  # 웨이포인트 오프셋 거리
        self.num_ctrl_pts  = rospy.get_param('~num_ctrl_pts',  3)   # 스플라인 제어점 개수
        self.num_waypts    = rospy.get_param('~num_waypts',    1000) # 생성할 웨이포인트 개수
        self.min_dx        = rospy.get_param('~min_dx',        0.1)  # 제어점 간 최소 x축 거리

        self.camera_trigger = False # 카메라 기반 라바콘 감지 트리거
        
        rospy.Subscriber('/usb_cam/image_raw', Image, self.image_cb)
        self.bridge = CvBridge() 
       
        self.raw_pts = []  # 원시 제어점 저장 리스트

        # 연속적으로 유효한 포인트가 부족한 경우를 카운트
        self.consecutive_insufficient_points = 0
        self.INSUFFICIENT_POINTS_THRESHOLD = 28 # 포인트 부족 판단 임계값

        
        self.wp_pub = rospy.Publisher('/rubbercone_waypoints', Waypoint, queue_size=1)
        
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        rospy.spin() 

    # scan_cb 메소드: 라이다 스캔 데이터를 받아 처리합니다.
    def scan_cb(self, scan: LaserScan):
        # 카메라 트리거가 활성화되지 않으면 라이다 데이터 처리 안 함
        if not self.camera_trigger:
            return
        
        ranges = np.array(scan.ranges, dtype=np.float32)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges), dtype=np.float32)
        # 각도를 -180 ~ 180 범위로 변환
        rel_deg = (np.degrees(angles) + 180) % 360 - 180

        # 좌측 감지 영역 마스크 생성
        mask_l = (
            (ranges >= self.min_range) & (ranges <= self.max_range) &
            (rel_deg >= self.left_min_deg) & (rel_deg <= self.left_max_deg) &
            (ranges <= self.detect_dist) # 설정된 감지 거리 내의 포인트만 사용
        )
        # 마스크에 해당하는 포인트들을 (x, y) 좌표로 변환
        left_pts = [(ranges[i]*math.cos(angles[i]), ranges[i]*math.sin(angles[i]))
                    for i in np.where(mask_l)[0]]
        
        # 유효한 좌측 포인트가 2개 미만일 경우 처리
        if len(left_pts) < 2:
            self.consecutive_insufficient_points += 1
            # 임계값 이상으로 포인트가 부족하면 웨이포인트 발행 중단 및 카메라 트리거 비활성화
            if self.consecutive_insufficient_points >= self.INSUFFICIENT_POINTS_THRESHOLD:
                wp = Waypoint()
                wp.cnt = 0 # 웨이포인트 개수를 0으로 설정하여 중단 신호 전달
                wp.x_arr = [0.0] * 2000 
                wp.y_arr = [0.0] * 2000
                self.wp_pub.publish(wp)
                self.camera_trigger = False # 카메라 트리거 비활성화
            return
        else:
            # 유효한 포인트가 감지되면 카운터 초기화
            self.consecutive_insufficient_points = 0

        # 좌측 포인트를 x좌표 기준으로 정렬
        left_pts.sort(key=lambda p: p[0])
        
        # 일정 간격(offset_dist)으로 포인트 선택
        selected = []
        last_x = -float('inf')
        for lx, ly in left_pts:
            if lx >= last_x + self.offset_dist: # x축으로 일정 거리 이상 떨어진 포인트만 선택
                selected.append((lx, ly))
                last_x = lx
        if not selected: # 선택된 포인트가 없으면 종료
            return

        # 선택된 포인트들로부터 오프셋된 웨이포인트 계산
        offset_pts = []
        for i, (lx, ly) in enumerate(selected):
            # 현재 포인트와 다음 포인트(또는 이전 포인트)를 사용하여 법선 방향 계산
            if i < len(selected) - 1: # 마지막 포인트가 아니면 다음 포인트 사용
                nx, ny = selected[i+1][0] - lx, selected[i+1][1] - ly
            else: # 마지막 포인트면 이전 포인트 사용
                nx, ny = lx - selected[i-1][0], ly - selected[i-1][1]
            theta = math.atan2(ny, nx) # 두 점을 잇는 선분의 각도
            
            # 법선 방향으로 오프셋 계산
            ox = math.sin(theta) * self.offset_dist
            oy = -math.cos(theta) * self.offset_dist
            wx = lx + ox # 오프셋된 x 좌표
            wy = ly + oy # 오프셋된 y 좌표
            offset_pts.append((wx, wy))

        # 계산된 첫 번째 오프셋 포인트를 원시 제어점 리스트에 추가
        wx, wy = offset_pts[0]
        # 이전 제어점과 x축 거리가 min_dx 이상 차이날 경우에만 추가
        if not self.raw_pts or abs(wx - self.raw_pts[-1][0]) > self.min_dx:
            self.raw_pts.append((wx, wy))
            # 제어점 리스트가 최대 개수를 초과하면 가장 오래된 점 제거
            if len(self.raw_pts) > self.num_ctrl_pts:
                self.raw_pts.pop(0)

        # 제어점 개수가 충분하면 스플라인 보간된 웨이포인트 발행
        if len(self.raw_pts) >= self.num_ctrl_pts:
            self._publish_spline()
        else:
            # 제어점이 부족하면 마지막 제어점을 단일 웨이포인트로 발행 (Fallback)
            pt = self.raw_pts[-1]
            wp = Waypoint(); wp.cnt = 1
            wp.x_arr = [pt[0]] + [0.0]*1999
            wp.y_arr = [pt[1]] + [0.0]*1999
            self.wp_pub.publish(wp)

    # image_cb 메소드: 카메라 이미지를 받아 주황색 라바콘을 감지합니다.
    def image_cb(self, msg: Image):
        try:
        
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        h, w = cv_image.shape[:2] # 이미지 높이, 너비

        # 이미지 하단 75% 영역만 사용 (상단 25% 제외)
        cropped = cv_image[int(h * 0.25):, :]

        # BGR 이미지를 HSV 색공간으로 변환
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # 주황색 검출을 위한 HSV 색상 범위 정의
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([20, 255, 255])
        # 마스크 생성 (주황색 영역은 흰색, 나머지는 검은색)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        # 전체 픽셀 대비 주황색 픽셀 비율 계산
        orange_ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
        # 주황색 비율이 임계값(0.01)을 넘으면 카메라 트리거 활성화
        if orange_ratio > 0.01:
            self.camera_trigger = True 

    # _publish_spline 메소드: 제어점들을 사용하여 스플라인 보간된 웨이포인트를 생성하고 발행합니다.
    def _publish_spline(self):
        pts = np.array(self.raw_pts) # 제어점들을 numpy 배열로 변환
        
        # 제어점 간의 거리 계산
        deltas = np.diff(pts, axis=0)
        dists = np.hypot(deltas[:,0], deltas[:,1])
        cum = np.concatenate(([0], np.cumsum(dists))) # 누적 거리
        t = cum / cum[-1] # 정규화된 누적 거리 (0~1)
        
        # 3차 스플라인 보간 (x, y 각각 수행)
        csx = CubicSpline(t, pts[:,0], bc_type='clamped') # x 좌표 스플라인
        csy = CubicSpline(t, pts[:,1], bc_type='clamped') # y 좌표 스플라인
        # 보간된 경로를 따라 샘플링할 지점 생성
        samples = np.linspace(0, 1, self.num_waypts)
        xs, ys = csx(samples), csy(samples) # 샘플링된 x, y 좌표
        
        # Waypoint 메시지 생성 및 발행
        wp = Waypoint(); wp.cnt = self.num_waypts
        wp.x_arr = list(xs) + [0.0]*(2000-self.num_waypts) # 남는 공간은 0으로 채움
        wp.y_arr = list(ys) + [0.0]*(2000-self.num_waypts) # 남는 공간은 0으로 채움
        self.wp_pub.publish(wp)

if __name__ == '__main__':
    try:
        LidarLeftOffsetExtractor()
    except rospy.ROSInterruptException:
        pass 