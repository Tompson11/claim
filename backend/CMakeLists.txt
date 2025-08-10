#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(lidar_image_align LANGUAGES CUDA CXX)
add_compile_options(-Wno-dev)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(Torch_DIR /data/conda/envs/ky_3dgs/lib/python3.10/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message(WARNING "${TORCH_PYTHON_LIBRARY}")

set(pybind11_DIR /data/conda/envs/ky_3dgs/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

# set(pybind11_DIR /data/conda/envs/ky_3dgs/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(
	include
	/data/conda/envs/ky_3dgs/include/python3.10
  ${TORCH_INCLUDE_DIRS}
)

pybind11_add_module(
  lidar_image_align
  src/loss_calculation.cu
  pybind_cuda.cpp
)

set_target_properties(lidar_image_align PROPERTIES CUDA_ARCHITECTURES "70;75;86")
target_link_libraries(lidar_image_align PRIVATE pybind11::module ${Python3_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${CERES_LIBRARIES})
target_include_directories(lidar_image_align PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

