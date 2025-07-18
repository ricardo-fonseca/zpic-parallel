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
sources  = ['zdf.c', 'emf.cpp', 'current.cpp', 'laser.cpp',
			 'particles.cpp', 'species.cpp', 'density.cpp', 'udist.cpp']

sources  = [ basepath + '/' + s for s in sources ]

ext = Extension( "em2d", 
	sources = ["em2d.pyx"] + sources
)

setup( 
      ext_modules = cythonize( ext ),
	  cmdclass={"build_ext": custom_build_ext}
)

