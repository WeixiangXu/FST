from setuptools import setup
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='SparseConv',
    ext_modules=[
        CUDAExtension(
            name='SparseConv_cuda',
            sources=[
                'src/cuda/sparse_conv_ext.cpp',
                'src/cuda/sparse_conv_cuda.cpp',
                'src/cuda/sparse_conv_cuda_kernel.cu'
            ],
            define_macros = [('WITH_CUDA', None)])
    ],

    cmdclass={
        'build_ext': BuildExtension
    }
)