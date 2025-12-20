import os
import subprocess
import sys

if os.name == 'nt':
    if not os.environ.get('CUDA_HOME') or not os.environ.get('CUDA_PATH'):
        try:
            nvcc_path = subprocess.check_output(['where', 'nvcc']).decode().strip().split('\r\n')[0]
            if nvcc_path:
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
                print(f"Detected CUDA_HOME from nvcc: {cuda_home}")
                if not os.environ.get('CUDA_HOME'):
                    os.environ['CUDA_HOME'] = cuda_home
                if not os.environ.get('CUDA_PATH'):
                    os.environ['CUDA_PATH'] = cuda_home
        except Exception:
            print("Warning: CUDA_HOME/CUDA_PATH not set and nvcc not found in PATH.")

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dill-s6',
    version='0.1.0',
    description='Dill S6 Selective Scan with CUDA kernels',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='dill_s6.mamba_cuda_core',
            sources=[
                'dill_s6/cuda/scan_bridge.cpp',
                'dill_s6/cuda/scan_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.10',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)