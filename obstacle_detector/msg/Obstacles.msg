# obstacle_detector/msg/Obstacles.msg

std_msgs/Header       header       # 메시지 헤더
geometry_msgs/Point   point        # 감지된 점 좌표
float64               true_radius  # 측정된 실제 반지름