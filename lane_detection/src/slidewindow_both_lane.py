import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import *
from matplotlib.pyplot import *
import math
# float로 조향값 public

class SlideWindow:
    
    def __init__(self):
        self.current_line = "DEFAULT"

        self.x_previous = 320
    

    def slidewindow(self, img):
        # --------------------------------------------------------------------
        # 1) 끊어진 노란선 연결: '세로 방향'으로 20 픽셀 정도 메워 주는 구조 요소
        # vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
        # img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, vert_kernel)

        # --------------------------------------------------------------------
        # 2) 히스토그램 기반 초기 차선 기준점 찾기
        height, width = img.shape
        # 바닥 절반 영역을 합산한 히스토그램
        histogram = np.sum(img[height//2 : , :], axis=0)
        # # 바닥 전체 영역을 합산한 히스토그램
        # histogram = np.sum(img_closed[height : , :], axis=0)
        mid = width // 2
        x_current_left  = np.argmax(histogram[:mid])
        x_current_right = np.argmax(histogram[mid:]) + mid

        # --------------------------------------------------------------------
        # 3) sliding window 파라미터
        nwindows = 14
        window_height = height // nwindows
        margin = 50
        minpix = 200

        # nonzero 픽셀 좌표
        nonzero    = img.nonzero()
        nonzeroy   = np.array(nonzero[0])
        nonzerox   = np.array(nonzero[1])

        # 인덱스 누적용 리스트
        left_lane_inds  = []
        right_lane_inds = []

        out_img = np.dstack((img,)*3) * 255

        # --------------------------------------------------------------------
        # 4) 각 윈도우별로 차선 픽셀 찾기
        for window in range(nwindows):
            # 윈도우 y 범위
            win_y_low  = (window)*window_height
            win_y_high = (window+1)*window_height

            # 왼쪽 윈도우 x 범위
            win_x_low_left  = x_current_left  - margin
            win_x_high_left = x_current_left  + margin
            # 오른쪽 윈도우 x 범위
            win_x_low_right  = x_current_right - margin
            win_x_high_right = x_current_right + margin

            # 사각형 그리기(디버깅용)
            cv2.rectangle(out_img,
                          (win_x_low_left,  win_y_low),
                          (win_x_high_left, win_y_high),
                          (0,255,0),  2)
            cv2.rectangle(out_img,
                          (win_x_low_right,  win_y_low),
                          (win_x_high_right, win_y_high),
                          (255,0,0),  2)

            # 윈도우 안의 nonzero 픽셀 인덱스
            good_left_inds  = ((nonzeroy >= win_y_low) 
                            & (nonzeroy <  win_y_high)
                            & (nonzerox >= win_x_low_left)
                            & (nonzerox <  win_x_high_left)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low)
                            & (nonzeroy <  win_y_high)
                            & (nonzerox >= win_x_low_right)
                            & (nonzerox <  win_x_high_right)).nonzero()[0]

            # 기준점 갱신
            if len(good_left_inds) > minpix:
                x_current_left = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                x_current_right = int(np.mean(nonzerox[good_right_inds]))

            left_lane_inds .extend(good_left_inds)
            right_lane_inds.extend(good_right_inds)

        # --------------------------------------------------------------------
        # 5) 결과 리턴
        #    out_img: 윈도우 / 차선 픽셀 표시된 컬러 이미지
        #    x_location: 트래킹 포인트(예: 두 선의 중간)

        # 각 차선이 신뢰성 있게 감지되었는지 판단 (축적된 픽셀 수가 minpix 이상인 경우)
        left_lane_reliably_found = len(good_left_inds) >= minpix
        right_lane_reliably_found = len(good_right_inds) >= minpix
        # print("Left lane inds:", len(good_left_inds), "Right lane inds:", len(good_right_inds))

        # 차선 폭 추정 (초기 히스토그램 기반, 실패 시 기본값 사용)
        hist_peak_left = np.argmax(histogram[:mid])
        hist_peak_right = np.argmax(histogram[mid:]) + mid
        estimated_lane_width = hist_peak_right - hist_peak_left
        
        # estimated_lane_width가 너무 작거나 (예: margin보다 작거나 같음) 음수일 경우 기본값 사용
        if estimated_lane_width <= margin: # margin은 윈도우 탐색 범위, 적절한 최소 차선 폭 기준으로 사용
            estimated_lane_width = int(width * 0.4) # 예: 이미지 너비의 40%

        alpha = 0.4

        if left_lane_reliably_found and right_lane_reliably_found:
            # 양쪽 차선 모두 신뢰성 있게 감지된 경우: x_current_left와 x_current_right의 중간점
            x_location = (x_current_left + x_current_right) // 2
            # print("Both lanes found: x_location =", x_location)
        elif left_lane_reliably_found:
            # 왼쪽 차선만 신뢰성 있게 감지된 경우: x_current_left 기준으로 설정
            # x_current_left는 왼쪽 차선의 중심이므로, 차량 경로 중심은 오른쪽으로 차선 폭의 절반 이동
            current_x_candidate = x_current_left + estimated_lane_width // 2
            x_location = int(alpha * current_x_candidate + (1 - alpha) * self.x_previous)
            # print("Left lane found: x_location =", x_location)
        elif right_lane_reliably_found:
            # 오른쪽 차선만 신뢰성 있게 감지된 경우: x_current_right 기준으로 설정
            # x_current_right는 오른쪽 차선의 중심이므로, 차량 경로 중심은 왼쪽으로 차선 폭의 절반 이동
            current_x_candidate = x_current_right - estimated_lane_width // 2
            x_location = int(alpha * current_x_candidate + (1 - alpha) * self.x_previous)
            # print("Right lane found: x_location =", x_location)
        else:
            # 양쪽 차선 모두 신뢰성 있게 감지되지 않은 경우: 이전 x_location 값 사용
            x_location = self.x_previous
            # print("No lanes found, using previous x_location =", x_location)

        # x_location이 이미지 경계를 벗어나지 않도록 조정
        # x_location = max(0, min(x_location, width - 1))
        
        cv2.circle(out_img, (x_location, height - window_height//2), 10, (0,0,255), -1)

        self.x_previous = x_location

        return out_img, x_location, self.current_line