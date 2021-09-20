cmake_minimum_required(VERSION 3.17)
project(RobustVideoMatting.lite.ai.toolkit)

message(">>>> Current project is: ${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 11)

# link opencv.
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/lib/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio) # need only
    message("=================================================================================")
    message(STATUS "    OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    message("=================================================================================")
else ()
    message(FATAL_ERROR "OpenCV library not found")
endif()

# link onnxruntime.
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime/)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIBRARY_DIR})

# link lite.ai.toolkit
set(LITEHUB_DIR ${CMAKE_SOURCE_DIR}/lite.ai.toolkit)
set(LITEHUB_INCLUDE_DIR ${LITEHUB_DIR}/include)
set(LITEHUB_LIBRARY_DIR ${LITEHUB_DIR}/lib)
include_directories(${LITEHUB_INCLUDE_DIR})
link_directories(${LITEHUB_LIBRARY_DIR})

# add your executable
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/examples/build)

add_executable(lite_rvm examples/test_lite_rvm.cpp)
target_link_libraries(lite_rvm lite.ai.toolkit onnxruntime ${OpenCV_LIBS})  # link lite.ai.toolkit


