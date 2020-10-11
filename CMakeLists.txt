cmake_minimum_required(VERSION 3.0)

project(directory_test)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

#SET("OpenCV_DIR" "/home/luca/opencv/build")
#find_package( OpenCV 4 REQUIRED PATHS "usr/local/OpenCV")

find_package( OpenCV 4)
#find_package(OpenCV REQUIRED PATHS /home/luca/opencv/build )

include_directories(${OpenCV_INCLUDE_DIRS})


MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

MESSAGE ( STATUS "Found OpenCV: ${OpenCV_VERSION}" )
MESSAGE ( STATUS "OpenCV_INCLUDE_DIRS= ${OpenCV_INCLUDE_DIRS}" )
MESSAGE ( STATUS "OpenCV_DIR= $ENV{OpenCV_DIR}" )

#Bring the headers, such as Student.h into the project

MESSAGE ( STATUS "found include: include" )


#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)



#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "src/*.cpp")

include_directories(include)
include_directories(keyPointExtraction)
include_directories(dataAssociation)
include_directories(fft-real/src)




add_executable(radarTest ${SOURCES}

        ../include/points_utils.cpp
        ../include/data_association.cpp
        ../include/smicp_solver.cpp
        ../keyPointExtraction/preProcessRadarScan.cpp
        ../dataAssociation/dataAssociationRadar.cpp


)
#include_directories(include)


target_link_libraries( radarTest ${OpenCV_LIBS} )
