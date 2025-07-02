# Run using
# python setup.py build_ext --inplace

# distutils: language = c++
# cython: language_level=3

from setuptools import setup, Extension
from Cython.Build import cythonize

# ext = Extension( "em2d", sources = ["em2d.pyx", "class.cpp"])

source = "../em2d/"

ext = Extension( "em2d", sources = ["em2d.pyx",
                    source + "zdf.c",
                    source + "emf.cpp",
                    source + "laser.cpp"])

setup( ext_modules = cythonize( ext ) )

