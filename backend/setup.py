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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import os
base_path = os.path.dirname(os.path.abspath(__file__))

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")

CUDA_FLAGS = [
    "-gencode", "arch=compute_70,code=sm_70",
    "-gencode", "arch=compute_75,code=sm_75",
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_89,code=compute_89",
]

setup(
    name="lidar_image_align",
    packages=['lidar_image_align'],
    ext_modules=[
        CUDAExtension(
            name="lidar_image_align._C",
            sources=[
                f"src/loss_calculation.cu",
                f"pybind_cuda.cpp"],
            extra_compile_args={"nvcc": CUDA_FLAGS}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    runtime_library_dirs=[torch_lib],
)
