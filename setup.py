from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension("BayesOptimizer_wrapper",
                             ["BayesOptimizer_wrapper.pyx",
                              "BayesOptimizer.cpp"], language="c++",extra_compile_args=['-O3', '-lboost_system',  '-lboost_filesystem'], )],
      cmdclass = {'build_ext': build_ext})
