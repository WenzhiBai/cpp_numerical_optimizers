cmake_minimum_required(VERSION 3.10)
project(cpp_numerical_optimizers)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})

add_executable(line_opt_ceres src/LineOptimizationCeres.cpp)
target_link_libraries(line_opt_ceres ${CERES_LIBRARIES})

# add_executable(line_opt_sim src/LineOptimizationSim.cpp)