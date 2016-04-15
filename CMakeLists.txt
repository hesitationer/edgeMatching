cmake_minimum_required(VERSION 2.8)

project(Chamfer-Matching)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(CHAMFER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Chamfer)
include_directories(${CHAMFER_DIR})

set(CHAMFER_HEADERS 
  ${CHAMFER_DIR}/include/Chamfer.hpp
  ${CHAMFER_DIR}/include/Utils.hpp
)
set(CHAMFER_SOURCES 
  ${CHAMFER_DIR}/src/Chamfer.cpp
  ${CHAMFER_DIR}/src/Utils.cpp
)


set(HOG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/HOG)
include_directories(${HOG_DIR})

set(HOG_HEADERS 
  ${HOG_DIR}/include/HOGDetector.hpp
)
set(HOG_SOURCES 
  ${HOG_DIR}/src/HOGDetector.cpp
)


# Define folder with data for tests
add_definitions( -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/")

# set the list of source files
set(test_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-chamfer-matching.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-polar-line.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-find-contours.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-hog.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-line-reconstruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-angle-error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test-contours-orientation.cpp
)


foreach(cpp ${test_cpp})
  get_filename_component(target ${cpp} NAME_WE)
  include_directories(${OpenCV_INCLUDE_DIRS})
  add_executable(${target} ${cpp} ${CHAMFER_HEADERS} ${CHAMFER_SOURCES} ${HOG_HEADERS} ${HOG_SOURCES})
  # Link your application with OpenCV libraries
  target_link_libraries(${target} ${OpenCV_LIBS})
endforeach()
