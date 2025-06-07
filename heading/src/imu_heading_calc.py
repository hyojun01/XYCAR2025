#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
import tf
import math

# ImuHeadingCalculator 클래스: IMU 데이터를 받아 heading 값을 계산하고 발행합니다.
class ImuHeadingCalculator:
    def __init__(self):
        self.heading = None # 계산된 heading 값을 저장할 변수
        # ROS 발행자 설정: "/heading" 토픽으로 heading 값 (Float64)을 발행
        self.heading_pub = rospy.Publisher("/heading", Float64, queue_size=1)
        # ROS 구독자 설정: "/imu" 토픽에서 IMU 메시지를 받아 self.imuCB 콜백 함수 호출
        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imuCB)


    # imuCB 메소드: IMU 메시지를 수신했을 때 호출되는 콜백 함수
    def imuCB(self, msg):
        # IMU 메시지에서 쿼터니언 값 추출
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )

        # 쿼터니언을 오일러 각도로 변환
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll, pitch, yaw = euler # roll, pitch, yaw 값 추출

        # yaw 값을 라디안에서 도로 변환하여 heading 값으로 사용
        self.heading = yaw * 180.0 / math.pi

        # 계산된 heading 값을 Float64 메시지로 만들어 발행
        heading_msg = Float64()
        heading_msg.data = self.heading
        self.heading_pub.publish(heading_msg)

if __name__ == "__main__":
    rospy.init_node("imu_heading_calculator")
    imuHeadingCalculator = ImuHeadingCalculator()
    rospy.spin()