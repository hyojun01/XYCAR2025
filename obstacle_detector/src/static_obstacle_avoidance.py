#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  ▸ /raw_obstacles_static  : 장애물(같은 차선) 수신
  ▸ /xycar_motor_static    : 차선 변경용 스티어/속도 퍼블리시
  ▸ 로직 :  L(주행) → C(차선 변경) → L
            C 상태에선 고정조향 ±STEER_CMD 를 CHANGE_FRAMES 만큼 유지
"""

import rospy, math
from obstacle_detector.msg import Obstacles
from std_msgs.msg import Float64, String
from xycar_msgs.msg import xycar_motor

# ────────── 차선·ROI 파라미터 ──────────
LANE_HALF      = 0.6   # [m] 차선 중앙에서 y±0.6 이내가 “현재 차선”
ROI_X_MAX      = 8.0   # [m] 전방 장애물 감지 범위

# ────────── 차선 변경 파라미터 ──────────
STEER_CMD      = 25.0  # [deg] 변경할 때 고정 스티어
CHANGE_FRAMES  = 15    # [loops] ≈ 15/30Hz = 0.5 s
SPEED_CRUISE   = 30     # [km/h] 평상시
SPEED_CHANGE   = 5     # [km/h] 차선 변경 시

class Obstacle:                    # 간편 구조체
    def __init__(self,x=None,y=None,dist=None):
        self.x,self.y,self.dist = x,y,dist

class StaticAvoidance:
    def __init__(self):
        # ── ROS I/O ──
        rospy.Subscriber("/raw_obstacles_static", Obstacles, self.obstacleCB)
        rospy.Subscriber("/heading",  Float64,  self.headingCB)  # 유지-호환
        self.pub = rospy.Publisher("/xycar_motor_static",
                                   xycar_motor, queue_size=1)

        self.cmd   = xycar_motor()
        self.lane  = "LEFT"      # 시작 차선
        self.state = "L"         # FSM: L 주행 / C 변경
        self.dir   = None        # 이번에 꺾을 방향
        self.counter_change = 0  # C 단계 남은 frame 수
        self.obs = []

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()

    # ────────── 매 루프 처리 ──────────
    def step(self):
        if self.state == "L":                 # 정상 주행
            if self.detect_in_lane():         # 같은 차선 장애물?
                self.start_change()
            self.publish(SPEED_CRUISE, 0.0, False)

        elif self.state == "C":               # 차선 변경 중
            steer = -STEER_CMD if self.dir=="RIGHT" else STEER_CMD
            self.publish(SPEED_CHANGE, steer, True)
            self.counter_change -= 1
            if self.counter_change <= 0:      # 타이머 끝 → 직진
                self.finish_change()

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
            if not (0 < ob.x < ROI_X_MAX):          # 전방 범위
                continue
            if self.lane=="LEFT"  and 0.0 <= ob.y <= LANE_HALF:
                return True
            if self.lane=="RIGHT" and -LANE_HALF <= ob.y <= 0.0:
                return True
        return False

    # ────────── 콜백 ──────────
    def obstacleCB(self,msg):
        self.obs=[Obstacle(c.center.x, c.center.y,
                           math.hypot(c.center.x,c.center.y))
                  for c in msg.circles]

    def headingCB(self,msg): pass                 # 호환용, 미사용

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


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy, math
# from obstacle_detector.msg import Obstacles
# from std_msgs.msg import Float64, String
# from xycar_msgs.msg import xycar_motor

# LANE_HALF   = 0.6   # [m]  현 차선 y 범위 (센터±0.6)
# ROI_X_MAX   = 10.0   # [m]  전방 거리 한계

# class Obstacle:
#     def __init__(self, x=None, y=None, dist=None):
#         self.x, self.y, self.dist = x, y, dist

# class StaticAvoidance:
#     def __init__(self):
#         # ────── ROS I/O ──────
#         rospy.Subscriber("/raw_obstacles_static", Obstacles, self.obstacleCB)
#         rospy.Subscriber("/heading",              Float64,   self.headingCB)
#         rospy.Subscriber("/mode",                 String,    self.modeCB)

#         self.pub = rospy.Publisher("/xycar_motor_static",
#                                    xycar_motor, queue_size=1)
#         self.cmd = xycar_motor()

#         # ────── 상태 변수 ──────
#         self.state = "L"          # FSM: L 주행 / A 회피 / R 복귀
#         self.lane  = "LEFT"       # ← 내부에서만 관리, 초기값 LEFT
#         self.dir   = None         # 이번 회피 방향 "LEFT"/"RIGHT"

#         # 주행 파라미터
#         self.speed = 5
#         self.angle = 0

#         # 헤딩 관리
#         self.real_hd = None
#         self.gt_hd   = None
#         self.local_hd= None
#         self.avoid_hd= None
#         self.return_hd=None

#         # 장애물·카운터
#         self.obs = []
#         self.closest = Obstacle()
#         self.cnt, self.miss = 0, 0
#         self.CNT_MAX  = 10
#         self.MISS_MAX = 5

#         rate = rospy.Rate(30)
#         while not rospy.is_shutdown():
#             self.fsm_step()
#             self.publish(self.speed, self.angle, self.state != "L")
#             rate.sleep()

#     # ────────── FSM ──────────
#     def fsm_step(self):
#         self.update_counter()
#         if self.state == "L" and self.cnt >= self.CNT_MAX:
#             self.start_avoid()
#         elif self.state == "A":
#             self.do_avoid()
#         elif self.state == "R":
#             self.do_return()

#     # ────────── 회피 시작 ──────────
#     def start_avoid(self):
#         self.gt_hd = self.real_hd

#         if self.lane == "LEFT":           # 현재 왼쪽 → 오른쪽으로 피하기
#             self.dir   = "RIGHT"
#             self.avoid_hd, self.return_hd = -25.0, 10.0
#         else:                             # 현재 오른쪽 → 왼쪽으로 피하기
#             self.dir   = "LEFT"
#             self.avoid_hd, self.return_hd = 30.0, -20.0

#         self.state = "A"
#         rospy.loginfo(f"[STATIC] AVOID start, dir={self.dir}")

#     # ────────── A 상태 ──────────
#     def do_avoid(self):
#         if self.local_hd is None: return
#         if self.dir == "RIGHT":
#             if self.local_hd > self.avoid_hd:
#                 self.angle = 1.8 * (self.local_hd - self.avoid_hd)
#             else:
#                 self.state = "R"
#         else:  # LEFT
#             if self.local_hd < self.avoid_hd:
#                 self.angle = -13.5 * abs(self.local_hd - self.avoid_hd)
#             else:
#                 self.state = "R"

#     # ────────── R 상태 ──────────
#     def do_return(self):
#         if self.local_hd is None: return
#         if self.dir == "RIGHT":  # 복귀는 좌로
#             if self.local_hd < self.return_hd:
#                 self.angle = -1.0 * abs(self.local_hd - self.return_hd)
#             else:
#                 self.finish_avoid()
#         else:                    # 복귀는 우로
#             if self.local_hd > self.return_hd:
#                 self.angle = 1.0 * abs(self.local_hd - self.return_hd)
#             else:
#                 self.finish_avoid()

#     # ────────── 복귀 완료 ──────────
#     def finish_avoid(self):
#         self.state = "L"
#         # ▸ 차선 정보 토글
#         self.lane = "RIGHT" if self.lane == "LEFT" else "LEFT"
#         # 변수 초기화
#         self.dir = self.gt_hd = self.avoid_hd = self.return_hd = None
#         self.cnt = 0
#         rospy.loginfo(f"[STATIC] RETURN done → lane={self.lane}")

#     # # ────────── 카운터 관리 ──────────
#     # def update_counter(self):
#     #     in_roi = any(0 < ob.x < self.roi_x_max and abs(ob.y) <= self.roi_y
#     #                  for ob in self.obs)
#     #     if in_roi:
#     #         self.cnt = min(self.CNT_MAX, self.cnt + 1); self.miss = 0
#     #     else:
#     #         self.miss += 1
#     #         if self.miss > self.MISS_MAX:
#     #             self.cnt = max(0, self.cnt - 1)

#     def update_counter(self):
#         def in_my_lane(ob):
#             if not (0 < ob.x < ROI_X_MAX):          # 전방 거리 조건
#                 return False
#             # y 부호로 현재 차선 구분 (y+:좌, y-:우)
#             if self.lane == "LEFT":
#                 return 0.0 <= ob.y <= LANE_HALF
#             else:  # RIGHT
#                 return -LANE_HALF <= ob.y <= 0.0

#         in_roi = any(in_my_lane(ob) for ob in self.obs)

#         if in_roi:
#             self.cnt  = min(self.CNT_MAX, self.cnt + 1)
#             self.miss = 0
#         else:
#             self.miss += 1
#             if self.miss > self.MISS_MAX:
#                 self.cnt = max(0, self.cnt - 1)

#     # ────────── 콜백 ──────────
#     def obstacleCB(self, msg):
#         self.obs = [Obstacle(c.center.x, c.center.y,
#                              math.hypot(c.center.x, c.center.y))
#                     for c in msg.circles]
#         self.obs.sort(key=lambda o: o.dist)
#         self.closest = self.obs[0] if self.obs else Obstacle()

#     def headingCB(self, msg):
#         self.real_hd = msg.data
#         if self.gt_hd is not None:
#             self.local_hd = self.real_hd - self.gt_hd
#             if self.local_hd > 180:   self.local_hd -= 360
#             elif self.local_hd < -180:self.local_hd += 360
#         else:
#             self.local_hd = None

#     def modeCB(self, msg): self.mode = msg.data

#     # ────────── 모터 퍼블리시 ──────────
#     def publish(self, spd, ang, flag):
#         self.cmd.speed = round(spd)
#         self.cmd.angle = round(ang)
#         self.cmd.flag  = flag
#         self.pub.publish(self.cmd)

# # ────────── 실행 ──────────
# if __name__ == "__main__":
#     rospy.init_node("static_obstacle_avoidance", anonymous=True)
#     try:
#         StaticAvoidance()
#     except rospy.ROSInterruptException:
#         pass