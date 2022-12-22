from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv_cuda',
    version='0.0.1',
    ext_modules=[ Extension(
        name='conv_cuda',
        sources=['conv_cuda.cpp'],
        language='c++',
        include_dirs=[      # 添加编译时用到的头文件目录
            get_pybind_include(),
            get_pybind_include(user=True)
        ]),
        CUDAExtension('conv_cuda',
                      sources=['conv_cuda.cpp']),
    ],
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={
        'build_ext': BuildExtension
    })
# , 'conv_cuda_kernel.cu'