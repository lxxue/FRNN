import os
import glob

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")
    main_source = os.path.join(extensions_dir, "ext.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "*", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "*", "*.cu"))

    extension  = CUDAExtension
    sources += source_cuda
    extra_compile_args = {"cxx": ["-std=c++14"]}

    nvcc_args = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
    if nvcc_flags_env != "":
        nvcc_args.extend(nvcc_flags_env.split(" "))

    CC = os.environ.get("CC", None)
    if CC is not None:
        CC_arg = "-ccbin={}".format(CC)
        if CC_arg not in nvcc_args:
            if any(arg.startswith("-ccbin") for arg in nvcc_args):
                raise ValueError("Inconsistent ccbins")
            nvcc_args.append(CC_arg)

    extra_compile_args["nvcc"] = nvcc_args

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "frnn",
            sources,
            include_dirs=include_dirs,
            define_macros=[],
            extra_compile_args=extra_compile_args
        )
    ]
    return ext_modules

if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)

else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension


setup(
    name="frnn",
    author="Lixin Xue, Yifan Wang",
    description="Fixed radius nearest neighbor search on gpu",
    packages=find_packages(exclude=("tests")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)


# class BuildExtension(torch.utils.cpp_extension.BuildExtension):
#     def __init__(self, *args, **kwargs):
#         super().__init__(use_ninja=False, *args, **kwargs)
# 
# # include_dirs = torch.utils.cpp_extension.include_paths()+["/local/home/lixxue/support_libs/gvdb-voxels/source/gvdb_library"]
# include_dirs = torch.utils.cpp_extension.include_paths()
# # print(include_dirs)
# 
# setup(
#     name="FRNN",
#     # author="Lixin Xue, GVDB",
#     # description="Fixed Radius Nearest Neighbor search using GVDB's uniform grid implementation",
#     # include_dirs=include_dirs,
#     # ext_modules=get_extensions(),
#     # ext_modules=[
#     #     CUDAExtension('FRNN.cpu', ['csrc/frnn_cpu.cpp', 'csrc/grid_cpu.cpp']),
#     #     CUDAExtension('FRNN.cuda', ['csrc/frnn_cuda.cpp', 'csrc/frnn.cu', 'csrc/grid.cu', 'csrc/utils/prefix_sum.cu', 'csrc/utils/counting_sort.cu']),
#     # ],
#     # cmdclass={"build_ext": BuildExtension},
#     cmdclass={
#         'build_ext': torch.utils.cpp_extension.BuildExtension
#     }
# )
# 