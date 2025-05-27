from setuptools import setup                  # setuptools：标准打包工具
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    # pip 安装包名称
    name='polynomial_cuda',

    # ext_modules 列表中定义所有 C++/CUDA 扩展模块
    ext_modules=[
        CUDAExtension(
            name='polynomial_cuda',            # 模块名，导入时使用
            sources=[                          # 源代码列表
                'polynomial_cuda.cu',
            ],
            extra_compile_args={               # 可选：传递给编译器的额外参数
                'cxx': [],                     # C++ 编译器参数
                'nvcc': [                      # NVCC 编译器参数
                    '-O3',                     # 优化等级 3
                    '--use_fast_math',         # 使用快速数学库
                ],
            }
        ),
    ],

    # 指定自定义 build_ext 命令，用于调用 PyTorch 提供的构建流程
    cmdclass={
        'build_ext': BuildExtension
    }
)
