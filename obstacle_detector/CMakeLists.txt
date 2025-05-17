cmake_minimum_required(VERSION 3.0.2)
project(obstacle_detector)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## 필수 패키지
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  tf
  message_generation
)

## 외부 수학 라이브러리
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

## 메시지 정의
add_message_files(
  FILES
  CircleObstacle.msg
  SegmentObstacle.msg
  Obstacles.msg
  Waypoint.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES obstacle_detector
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs tf message_runtime
)

## include 경로 설정
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${ARMADILLO_INCLUDE_DIRS}
  ${BLAS_INCLUDE_DIRS}
  ${LAPACK_INCLUDE_DIRS}
)

## 노드: obstacle_extractor_rubbercone_node
add_executable(obstacle_extractor_rubbercone_node
  src/obstacle_extractor_rubbercone_node.cpp
  src/obstacle_extractor_rubbercone.cpp
)

## 링커 설정
target_link_libraries(obstacle_extractor_rubbercone_node
  ${catkin_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${LAPACK_LIBRARIES}
)

add_dependencies(obstacle_extractor_rubbercone_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## 추가로 필요한 노드가 있으면 여기에 add_executable + target_link_libraries 추가하면 됨

