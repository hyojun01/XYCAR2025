# shared_function.py

import cv2
import numpy as np


def roi_for_lane(img, vert_ratio=0.4):
    """
    이미지의 아래쪽 일부(차선이 주로 보이는 영역)만 남기고 나머지는 제거합니다.
    vert_ratio: ROI 윗선이 이미지 높이의 몇 %에 위치할지 (0~1)
    """
    h, w = img.shape[:2]
    
    # ROI의 시작 y 좌표 계산
    roi_top_y = int(h * vert_ratio)
    
    # 이미지를 ROI에 맞게 자르기
    # y좌표는 roi_top_y 부터 h (이미지 끝)까지, x좌표는 전체
    cropped_img = img[roi_top_y:h, :]
    
    return cropped_img


def process_image(img):
    """
    RGB → HLS 변환 후 흰색·노란색 차선만 남겨 0/1 이진 이미지로 반환합니다.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 노란색 범위 (H≈15~35°, S/L 적당)
    lower_yellow = np.array([15, 30,  90])
    upper_yellow = np.array([35, 200, 255])
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)

    # 흰색 범위 (L 밝고 S 작음)
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([255, 255, 120])
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    mask = cv2.bitwise_or(mask_yellow, mask_white)

    # 노이즈 제거 & 팽창/침식
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    binary = mask // 255  # 0/1 로 변환
    return binary.astype(np.uint8)


def warper(img, src_pts, dst_pts, out_size=(640, 480)):
    """
    src_pts (4×2): 원본 이미지의 트랩이즈(주로 ROI 내부 도로 영역) 꼭짓점
    dst_pts (4×2): 변환 후 평행사변형 → 직사각형이 되도록 배치
    """
    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    warped = cv2.warpPerspective(img, M, out_size, flags=cv2.INTER_LINEAR)
    return warped