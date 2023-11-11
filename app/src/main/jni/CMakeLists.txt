# version required
cmake_minimum_required(VERSION 3.5)
PROJECT(hexagon)

# compile flags
#if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_definitions(-DDEBUG)
#endif()

# collect sources
aux_source_directory(./ SRCS)

# source files
add_library(${PROJECT_NAME} SHARED ${SRCS})

# libraries we need
find_library(log-lib log)

# imported library
set(IMPORT_DIR ${CMAKE_SOURCE_DIR}/../../../libs)
add_library(libc++_shared SHARED IMPORTED)
add_library(libSNPE SHARED IMPORTED)
set_target_properties(libc++_shared PROPERTIES IMPORTED_LOCATION ${IMPORT_DIR}/${ANDROID_ABI}/libc++_shared.so)
set_target_properties(libSNPE PROPERTIES IMPORTED_LOCATION ${IMPORT_DIR}/${ANDROID_ABI}/libSNPE.so)

# include dir
include_directories(
        .
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
        libc++_shared
        libSNPE
)