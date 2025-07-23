# Run using
# python setup.py build_ext --inplace

# distutils: language = c++
# cython: language_level=3

import os
from setuptools import setup, Extension
from Cython.Build import cythonize

import sysconfig
from setuptools.command.build_ext import build_ext


##################################################################################
#
# Compiler customizations

# C++ compilers
cxx = sysconfig.get_config_var('CXX')

# C++ flags
cxxflags = sysconfig.get_config_var('CFLAGS')

# Remove '-g' option
cxxflags = cxxflags.replace(' -g', '')

# Disable common (safe) warnings
cxxflags = cxxflags.replace(' -Wunreachable-code', '')
cxxflags += ' -Wno-vla-cxx-extension'
cxxflags += ' -Wno-unused-function'
cxxflags += ' -Wno-delete-non-abstract-non-virtual-dtor'

# OpenMP support
cxxflags += ' -fopenmp'

# ARM Neon extensions
cxxflags += ' -DUSE_NEON'

# Enable AVX2 extensions with FMA support
# cxxflags += ' -DUSE_AVX2 -mavx2 -mfma'

# C++ linker
ldcxx = sysconfig.get_config_var('LDCXXSHARED')

# C++ linker flags
ldcxxflags = sysconfig.get_config_var('LDFLAGS')


class custom_build_ext(build_ext):
	def build_extensions(self):
		self.compiler.set_executable("compiler_so_cxx", cxx + " " + cxxflags)
		self.compiler.set_executable("linker_so_cxx", ldcxx + " " + cxxflags + " " + ldcxxflags )
		build_ext.build_extensions(self)

##################################################################################
#
# Module definitions

basepath = '../em2d'
sources  = ['zdf.c', 'emf.cpp', 'current.cpp', 
			 'particles.cpp', 'udist.cpp', 'density.cpp', 'species.cpp']

# note: udist.cpp and density.cpp must be compiled also for the base module

sources  = [ basepath + '/' + s for s in sources ]

ext = [
    Extension( "em2d.em2d",            sources = ["em2d/em2d.pyx"] + sources ),
    Extension( "em2d.laser.laser",     sources = ["em2d/laser/laser.pyx", '../em2d/laser.cpp' ] ),
    Extension( "em2d.udist.udist",     sources = ["em2d/udist/udist.pyx", '../em2d/udist.cpp' ] ),
    Extension( "em2d.density.density", sources = ["em2d/density/density.pyx", '../em2d/density.cpp' ] ),
    Extension( "em2d.filter.filter",   sources = ["em2d/filter/filter.pyx" ] )
]

setup( 
      ext_modules = cythonize( ext ),
	  cmdclass={"build_ext": custom_build_ext}
)

