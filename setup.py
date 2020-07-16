from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(ext_modules=[Extension("BayesOptimizer_wrapper",
                             ["BayesOptimizer_wrapper.pyx",
                              "BayesOptimizer.cpp"], language="c++",extra_compile_args=['-lboost_system',  '-lboost_filesystem', '-lboost_thread', '-lpthread', '-L/usr/local/lib/'], )],
      cmdclass = {'build_ext': build_ext})
