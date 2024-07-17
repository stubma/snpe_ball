# version required
cmake_minimum_required(VERSION 3.5)
PROJECT(hexagon)

# compile flags
#if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
#endif()

# collect sources
aux_source_directory(./ SRCS)
aux_source_directory(./snpe SRCS)
aux_source_directory(./media SRCS)
aux_source_directory(./shared SRCS)

# libSnpeHtpV73Skel.so只有armv7的版本, 而其他库都是arm64的, 所以目前没有办法
# 把snpe的所有依赖库都打包到apk里面, 所以目前只能用add_executable编译一个可执行
# 程序
#add_library(${PROJECT_NAME} SHARED ${SRCS})

# if want to build an executable program, use this and comment above add_library
add_executable(${PROJECT_NAME} ${SRCS})
add_definitions(-DSTANDALONE)

# libraries we need
find_library(log-lib log)
find_library(media-lib mediandk)

# imported library
set(IMPORT_DIR ${CMAKE_SOURCE_DIR}/../jniLibs)
add_library(libc++_shared SHARED IMPORTED)
add_library(libSNPE SHARED IMPORTED)
set_target_properties(libc++_shared PROPERTIES IMPORTED_LOCATION ${IMPORT_DIR}/${ANDROID_ABI}/libc++_shared.so)
set_target_properties(libSNPE PROPERTIES IMPORTED_LOCATION ${IMPORT_DIR}/${ANDROID_ABI}/libSNPE.so)

# import opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../../../opencv/native)
include_directories(${OpenCV_DIR}/jni/include)
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${OpenCV_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so)

# include dir
include_directories(
        .
        snpe
        media
        include
        include/SNPE
        include/SNPE/DiagLog
        include/SNPE/DlContainer
        include/SNPE/DlSystem
        include/SNPE/PlatformValidator
        include/SNPE/SNPE
        include/SNPE/SnpeUdo
)

# link
target_link_libraries(
        ${PROJECT_NAME}
        ${log-lib}
        ${media-lib}
        libc++_shared
        libSNPE
        lib_opencv
)