cmake_minimum_required(VERSION 2.8)
project( DirectAlignment )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

option(OPENMP_ENABLED "Whether to enable OpenMP" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" OFF)

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()


if(CUDA_ENABLED)
    find_package(CUDA QUIET)
endif()

################################################################################
find_package( OpenCV REQUIRED )
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})

IF(WIN32) 
	add_definitions(-DNOMINMAX)
	add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)	
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)	
ENDIF()

add_executable( DirectAlignment main.cpp
 DataIO.cpp
 MathUlti.cpp
 ImgUlti.cpp
 Visualization.cpp
)

IF(WIN32) 
target_link_libraries( DirectAlignment ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
elseif(UNIX)
target_link_libraries( DirectAlignment ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
ENDIF()
