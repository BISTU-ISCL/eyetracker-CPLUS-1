cmake_minimum_required(VERSION 3.10)
project(eyetracker_cplus_1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(eye_tracker
    src/main.cpp
    src/EyeTrackerCalibrator.cpp
)

target_include_directories(eye_tracker PRIVATE include)
target_link_libraries(eye_tracker PRIVATE ${OpenCV_LIBS})

