"""Builds the C++17 extension. OpenMP is used when the compiler supports it; otherwise the build falls back to a single-threaded
extension instead of failing."""
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

if sys.platform == "win32":
    OMP_COMPILE, OMP_LINK = ["/openmp"], []
elif sys.platform == "darwin":
    OMP_COMPILE, OMP_LINK = ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
else:
    OMP_COMPILE, OMP_LINK = ["-fopenmp"], ["-fopenmp"]


def make_ext(openmp):
    return Pybind11Extension(
        "gkpbp._core", ["csrc/module.cpp"], include_dirs=["csrc"], cxx_std=17,
        extra_compile_args=(OMP_COMPILE if openmp else []) + (["-O3"] if sys.platform != "win32" else ["/O2"]),
        extra_link_args=OMP_LINK if openmp else [])


class BuildExt(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as err:                       # compiler without OpenMP: retry single-threaded
            print(f"gkpbp: OpenMP build failed ({err!r}); building single-threaded", file=sys.stderr)
            plain = make_ext(False)
            ext.extra_compile_args, ext.extra_link_args = plain.extra_compile_args, plain.extra_link_args
            super().build_extension(ext)


setup(ext_modules=[make_ext(True)], cmdclass={"build_ext": BuildExt})
