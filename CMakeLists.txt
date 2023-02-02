cmake_minimum_required(VERSION 2.8)

project( PC_Alignment )

find_package ( OpenCV REQUIRED )
find_package(Eigen3)
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add_executable ( PC_Alignment src/main.cpp )
add_executable ( PC_Alignment src/main.cpp src/PC_ReaderWriter.cpp)
target_include_directories(PC_Alignment PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(PC_Alignment ${OpenCV_LIBS} ${PCL_LIBRARIES})

