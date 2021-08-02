# For more information about build system see
# https://docs.espressif.com/projects/esp-idf/en/latest/api-guides/build-system.html
# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.5)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(main)

set(CMAKE_SYSTEM_NAME freertos)
set(CMAKE_SYSTEM_PROCESSOR xtensa-esp32)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -lc ")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lpthread --specs=nosys.specs")
#set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -pthread -lpthread -libnosys --specs=nosys.specs")
#set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
#set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
#set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

option(NCNN_INSTALL_SDK "" OFF)
option(NCNN_PIXEL_ROTATE "" OFF)
option(NCNN_PIXEL_AFFINE "" OFF)
option(NCNN_PIXEL_DRAWING "" OFF)
option(NCNN_BUILD_BENCHMARK "" OFF)
option(NCNN_BUILD_TESTS "" OFF)
option(NCNN_BUILD_TOOLS "" OFF)
option(NCNN_BUILD_EXAMPLES "" OFF)
option(NCNN_DISABLE_RTTI "" ON)
option(NCNN_DISABLE_EXCEPTION "" ON)
option(NCNN_BF16 "" OFF)
option(NCNN_INT8 "" OFF)
option(NCNN_THREADS "" OFF)
option(NCNN_OPENMP "" OFF)
option(NCNN_STDIO "" OFF)
option(NCNN_STRING "" OFF)
option(NCNN_PLATFORM_API "" OFF)
option(NCNN_RUNTIME_CPU "" OFF)

add_subdirectory(ncnn)
target_link_libraries(${CMAKE_PROJECT_NAME}.elf ncnn)
