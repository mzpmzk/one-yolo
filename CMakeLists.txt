#       ____  _   _ ______  __     ______  _      ____  
#      / __ \| \ | |  ____| \ \   / / __ \| |    / __ \ 
#     | |  | |  \| | |__     \ \_/ / |  | | |   | |  | |
#     | |  | | . ` |  __|     \   /| |  | | |   | |  | |
#     | |__| | |\  | |____     | | | |__| | |___| |__| |
#      \____/|_| \_|______|    |_|  \____/|______\____/ 
#                                                       
#                                     @zhouzhi since 2025           
cmake_minimum_required(VERSION 3.10)
project(one-yolo VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -w -fdiagnostics-color=always -pthread")

option(BUILD_WITH_ORT "enable ONNXRuntime?" OFF)
option(BUILD_WITH_OVN "enable OpenVINO(Intel Platform)?" OFF)
option(BUILD_WITH_TRT "enable TensorRT(Nvidia/CUDA Platform)?" OFF)
option(BUILD_WITH_RKN "enable RKNN(RockChip Platform)?" OFF)
option(BUILD_WITH_CML "enable CoreML(Apple Platform)?" OFF)
option(BUILD_WITH_PDL "enable PaddlePaddle?" OFF)
option(BUILD_WITH_CAN "enable CANN(HuaWei Platform)?" OFF)
option(BUILD_WITH_DEL "enable Denglin(Denglin Platform)?" OFF)
option(BUILD_WITH_CAB "enable Cambricon(Cambricon Platform)?" OFF)

# OpenCV required
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("include")

##########################
# build one-yolo library #
##########################
file(GLOB BASE_SRCS "src/*.cpp")
file(GLOB_RECURSE TRACK_SRCS "src/track/*.cpp")
file(GLOB_RECURSE ORT_SRCS "src/ort/*.cpp")
file(GLOB_RECURSE OVN_SRCS "src/ovn/*.cpp")
file(GLOB_RECURSE TRT_SRCS "src/trt/*.cpp")
file(GLOB_RECURSE RKN_SRCS "src/rkn/*.cpp")
list(APPEND ONE_YOLO_DEPEND_SRCS ${BASE_SRCS} ${TRACK_SRCS})
list(APPEND ONE_YOLO_DEPEND_LIBS ${OpenCV_LIBS})

# enable onnxruntime, 
# make sure onnxruntime installed and include & libs path could be found by cmake
if(BUILD_WITH_ORT)
    #include_directories("include path to onnxruntime")
    #link_directories("libs path to onnxruntime") 
    add_definitions(-DBUILD_WITH_ORT)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${ORT_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS onnxruntime)
endif()
# enable openvino, 
# make sure openvino installed and include & libs path could be found by cmake
if(BUILD_WITH_OVN)
    #include_directories("include path to openvino")
    #link_directories("libs path to openvino") 
    find_package(OpenVINO REQUIRED)
    add_definitions(-DBUILD_WITH_OVN)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${OVN_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS openvino::runtime)
endif()
# enable tensorrt, 
# make sure tensorrt installed and include & libs path could be found by cmake
if(BUILD_WITH_TRT)
    include_directories("/usr/local/tensorRT/include")
    link_directories("/usr/local/tensorRT/lib")
    add_definitions(-DBUILD_WITH_TRT)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${TRT_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS nvinfer cudart)
endif()
# enable rknn, 
# make sure rknn runtime installed and include & libs path could be found by cmake
if(BUILD_WITH_RKN)
    include_directories("/sd/z/rknnrt/include")
    link_directories("/sd/z/rknnrt/libs")
    add_definitions(-DBUILD_WITH_RKN)
    list(APPEND ONE_YOLO_DEPEND_SRCS ${RKN_SRCS})
    list(APPEND ONE_YOLO_DEPEND_LIBS rknnrt)
endif()

add_library(${PROJECT_NAME} SHARED ${ONE_YOLO_DEPEND_SRCS})
target_link_libraries(${PROJECT_NAME} ${ONE_YOLO_DEPEND_LIBS})

##########################
# build one-yolo samples #
##########################
add_subdirectory(samples)