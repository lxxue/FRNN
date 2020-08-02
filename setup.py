import os
import glob

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    main_source = os.path.join(this_dir, "ext.cpp")
    sources = glob.glob(os.path.join(this_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(this_dir, "*.cu"))

    extension  = CppExtension
    extra_compile_args = {"cxx": ["-std=c++14"]}

    sources = [os.path.join(this_dir, s) for s in sources]
    include_dirs = [this_dir]
    ext_modules = [
        extension(
            "FRNN",
            sources,
            include_dirs=include_dirs,
            define_macros=[],
            extra_compile_args=extra_compile_args
        )
    ]
    return ext_modules

class BuildExtension(torch.utils.cpp_extension.BuildExtension):
    def __init__(self, *args, **kwargs):
        super().__init__(use_ninja=False, *args, **kwargs)

# include_dirs = torch.utils.cpp_extension.include_paths()+["/local/home/lixxue/support_libs/gvdb-voxels/source/gvdb_library"]
include_dirs = torch.utils.cpp_extension.include_paths()
# print(include_dirs)

setup(
    name="FRNN",
    # author="Lixin Xue, GVDB",
    # description="Fixed Radius Nearest Neighbor search using GVDB's uniform grid implementation",
    # include_dirs=include_dirs,
    # ext_modules=get_extensions(),
    ext_modules=[
        CUDAExtension('FRNN.cpu', ['csrc/frnn_cpu.cpp', 'csrc/grid_cpu.cpp']),
        CUDAExtension('FRNN.cuda', ['csrc/frnn_cuda.cpp', 'csrc/frnn.cu', 'csrc/grid.cu', 'csrc/utils/prefix_sum.cu', 'csrc/utils/counting_sort.cu']),
    ],
    # cmdclass={"build_ext": BuildExtension},
    cmdclass={
        'build_ext': torch.utils.cpp_extension.BuildExtension
    }
)
