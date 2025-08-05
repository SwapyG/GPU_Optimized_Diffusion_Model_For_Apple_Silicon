# Filename: setup.py

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Get the directory where PyTorch's C++ libraries are installed.
pytorch_lib_dir = torch.utils.cpp_extension.library_paths()[0]

setup(
    name="diffusion_metal",
    ext_modules=[
        CppExtension(
            name="diffusion_metal",
            sources=["diffusion_metal_wrapper.mm"],
            extra_compile_args={
                # Use -O1 for light, safe optimizations which avoids the compiler bug.
                'cxx': ['-std=c++17', '-g', '-O1'],
            },
            extra_link_args=[
                # Embed the path to PyTorch's libraries directly into our module.
                f"-Wl,-rpath,{pytorch_lib_dir}",
                # Link against Apple's core frameworks for Metal.
                "-framework", "Foundation",
                "-framework", "Metal"
            ],
            # Also tell the linker where to find the libraries during the build.
            library_dirs=[pytorch_lib_dir],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)