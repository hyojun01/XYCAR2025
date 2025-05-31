#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  ▸ /raw_obstacles_static  : 장애물(같은 차선) 수신
  ▸ /xycar_motor_static    : 차선 변경용 스티어/속도 퍼블리시
  ▸ 로직 :  L(주행) → C(차선 변경) → L
            C 상태에선 고정조향 ±STEER_CMD 를 CHANGE_FRAMES 만큼 유지
"""


import rospy, math
from obstacle_detector.msg import Obstacles, CarObstacles
from std_msgs.msg import Float64, String
from xycar_msgs.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# ────────── 차선·ROI 파라미터 ──────────
LANE_HALF      = 0.6   # [m] 차선 중앙에서 y±0.6 이내가 “현재 차선”
ROI_X_MAX      = 7.0   # [m] 전방 장애물 감지 범위

# ────────── 차선 변경 파라미터 ──────────
STEER_CMD      = 80.0  # [deg] 변경할 때 고정 스티어
CHANGE_FRAMES  = 100    # [loops] ≈ 15/30Hz = 0.5 s
SPEED_CRUISE   = 60     # [km/h] 평상시
SPEED_CHANGE   = 30     # [km/h] 차선 변경 시

class Obstacle:                    # 간편 구조체
    def __init__(self,x=None,y=None,dist=None):
        self.x,self.y,self.dist = x,y,dist

class StaticAvoidance:
    def __init__(self):
        # ── ROS I/O ──
        rospy.Subscriber("/raw_obstacles_static", CarObstacles, self.obstacleCB)
        rospy.Subscriber("/heading",  Float64,  self.headingCB)  # 유지-호환
        rospy.Subscriber("/usb_cam/image_raw", Image, self.imageCB)  # 카메라 이미지 추가
        self.pub = rospy.Publisher("/xycar_motor_static",xycar_motor, queue_size=1)

        self.bridge = CvBridge()
        self.img = None

        self.cmd   = xycar_motor()
        self.lane  = "LEFT"      # 시작 차선
        self.state = "L"         # FSM: L 주행 / C 변경
        self.dir   = None        # 이번에 꺾을 방향
        self.counter_change = 0  # C 단계 남은 frame 수
        self.obs = []
        self.roi_x_max = ROI_X_MAX

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.step()
            self.visualize()
            rate.sleep()

    # ────────── 매 루프 처리 ──────────
    def step(self):
        if self.state == "L":                 # 정상 주행
            if self.detect_in_lane():         # 같은 차선 장애물?
                self.start_change()
            self.publish(SPEED_CRUISE, 0.0, False)

        elif self.state == "C":               # 차선 변경 중
            steer = STEER_CMD if self.dir=="RIGHT" else -STEER_CMD
            steer = (self.counter_change - CHANGE_FRAMES / 2) / (CHANGE_FRAMES / 2) * steer  # 점진적 조향
            self.publish(SPEED_CHANGE, steer, True)
            self.counter_change -= 1
            if self.counter_change <= 0:      # 타이머 끝 → 직진
                self.finish_change()
                self.roi_x_max = 13.5  # 차선 변경 후 전방 범위 늘림

    # ────────── 차선 변경 개시 ──────────
    def start_change(self):
        self.dir = "RIGHT" if self.lane=="LEFT" else "LEFT"
        self.counter_change = CHANGE_FRAMES
        self.state = "C"
        rospy.loginfo(f"[AVOID] change to {self.dir}")

    # ────────── 차선 변경 완료 ──────────
    def finish_change(self):
        self.state = "L"
        self.lane  = "RIGHT" if self.lane=="LEFT" else "LEFT"
        self.dir   = None
        rospy.loginfo(f"[AVOID] lane={self.lane}  straight ahead")

    # ────────── 같은 차선 장애물 유무 ──────────
    def detect_in_lane(self):
        for ob in self.obs:
            if not (0 < ob.x < self.roi_x_max):          # 전방 범위
                continue
            if self.lane=="LEFT"  and -LANE_HALF <= ob.y <= LANE_HALF:
                rospy.loginfo("[STATIC] 장애물 인식! (LEFT 차선)")
                return True
            if self.lane=="RIGHT" and -LANE_HALF <= ob.y <= LANE_HALF:
                rospy.loginfo("[STATIC] 장애물 인식! (RIGHT 차선)")
                return True
        return False

    # ────────── 콜백 ──────────
    def obstacleCB(self,msg):
        self.obs=[Obstacle(c.center.x, c.center.y,
                           math.hypot(c.center.x,c.center.y))
                  for c in msg.circles]

    def headingCB(self,msg): pass                 # 호환용, 미사용

    def imageCB(self, msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logwarn(e)

    def visualize(self):
        if self.img is None:
            return

        img_copy = self.img.copy()
        h, w, _ = img_copy.shape

        for ob in self.obs:
            # LiDAR 기준 x: 전방거리, y: 좌우거리
            # 이미지 기준: x → 가로축, y → 세로축 (아래가 +)
            px = int(w / 2 + ob.y * 50)  # 좌우 1m당 100px
            py = int(h - ob.x * 50)      # 전후 1m당 100px, 아래가 +이므로 h에서 빼줌

            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img_copy, (px, py), 8, (0, 0, 255), -1)
                rospy.loginfo("[STATIC] 장애물 인식!")
                cv2.putText(img_copy, f"{ob.dist:.1f}m", (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Obstacle View", img_copy)
        cv2.waitKey(1)

    # ────────── 퍼블리시 ──────────
    def publish(self,spd,ang,flag):
        self.cmd.speed = round(spd)
        self.cmd.angle = round(ang)
        self.cmd.flag  = flag         # True: 회피 중
        self.pub.publish(self.cmd)

# ────────── 실행 ──────────
if __name__ == "__main__":
    rospy.init_node("static_obstacle_avoidance", anonymous=True)
    try:  StaticAvoidance()
    except rospy.ROSInterruptException:  pass
    finally: cv2.destroyAllWindows()