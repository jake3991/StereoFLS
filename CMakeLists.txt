cmake_minimum_required(VERSION 2.8.3)
project(stereo_sonar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  pybind11_catkin
  rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

catkin_package(DEPENDS EIGEN3)
catkin_python_setup()

pybind11_add_module(cfar src/${PROJECT_NAME}/cpp/cfar.cpp)
set_target_properties(cfar PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
