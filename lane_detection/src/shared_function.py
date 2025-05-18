# shared_function.py

import cv2
import numpy as np

def roi_for_lane(image):
    """이미지의 하단 부분만 사용하여 ROI를 설정하는 함수"""
    # return image[246:396, :]
    return image[246:396, 60:580]    

def process_image(image):
    """이미지 전처리를 수행하는 함수 (노란색·흰색 차선만)"""
    # 1) HLS 색상 공간으로 변환 후 노란색, 흰색 마스크 생성
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # 노란색 범위 (H:15–35, L:30–204, S:115–255)
    lower_yellow = np.array([15,  30, 115])
    upper_yellow = np.array([35, 204, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    # 흰색 범위 (H:0–255, L:200–255, S:0–255)
    lower_white = np.array([  0, 200,   0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    # 둘을 합친 마스크로 원본 이미지 필터링
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    masked = cv2.bitwise_and(image, image, mask=mask)

    # 2) 그레이스케일 변환
    gray_img = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # 트랙바에서 가져온 현재 값들
    gaussian_kernel_size      = 13
    canny_lower, canny_upper = 33, 255
    morph_kernel_size        = 1

    # 3) 가우시안 블러 (노이즈 제거)
    blurred_image = cv2.GaussianBlur(gray_img,
                                     (gaussian_kernel_size, gaussian_kernel_size),
                                     0)

    # 4) 캐니 에지 검출
    edged = cv2.Canny(blurred_image, canny_lower, canny_upper)

    # 5) 형태학적 닫기 연산 (엣지 연결)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    return gray_img, blurred_image, mask, edged, closed_image

def warper(image):
    """원근 변환을 수행하는 함수"""
    y, x = image.shape[0:2]

    # src_point1 = [0, 126]  # 왼쪽 아래 점
    # src_point2 = [175, 46]  # 왼쪽 위 점
    # src_point3 = [456, 38]  # 오른쪽 위 점
    # src_point4 = [640, 103]  # 오른쪽 아래 점

    # src_points = np.float32([src_point1, src_point2, src_point3, src_point4])  # 원본 이미지에서의 점들

    src_point1 = [0 * x, 0.84 * y]  # 왼쪽 아래 점
    src_point2 = [0.273 * x, 0.306 * y]  # 왼쪽 위 점
    src_point3 = [0.712 * x, 0.253 * y]  # 오른쪽 위 점
    src_point4 = [1 * x, 0.686 * y]  # 오른쪽 아래 점

    src_points = np.float32([src_point1, src_point2, src_point3, src_point4])  # 원본 이미지에서의 점들
    
    # dst_point1 = [x // 4 + 9, y]  # 변환 이미지에서의 왼쪽 아래 점
    # dst_point2 = [x // 4 + 9, 0]  # 변환 이미지에서의 왼쪽 위 점
    # dst_point3 = [x // 4 * 3 + 9, 0]  # 변환 이미지에서의 오른쪽 위 점
    # dst_point4 = [x // 4 * 3 + 9, y]  # 변환 이미지에서의 오른쪽 아래 점

    dst_point1 = [x // 4, y]  # 변환 이미지에서의 왼쪽 아래 점
    dst_point2 = [x // 4, 0]  # 변환 이미지에서의 왼쪽 위 점
    dst_point3 = [x // 4 * 3, 0]  # 변환 이미지에서의 오른쪽 위 점
    dst_point4 = [x // 4 * 3, y]  # 변환 이미지에서의 오른쪽 아래 점

    dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])  # 변환 이미지에서의 점들
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)  # 원근 변환 행렬 계산
    warped_img = cv2.warpPerspective(image, matrix, (x, y))  # 원근 변환 적용
    
    return warped_img