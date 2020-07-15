#This ia Cython file and extracts the relevant classes from the C++ header file.

# distutils: language = c++
# distutils: sources = rectangle.cpp
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t)
cdef extern from "BayesOptimizer.hpp" namespace "BayesianOptimization":
    cdef cppclass BayesOptimizer:
        BayesOptimizer(map[string, vector[vector[int16_t]]])
        map[string, vector[int16_t]] config_space
        int space_length
        void build_search_space()
        void print_space()

cdef class PyBayesOptimizer:
    cdef BayesOptimizer *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, map[string, vector[vector[int16_t]]] s):
        self.thisptr = new BayesOptimizer(s)
    def __dealloc__(self):
        del self.thisptr
    def print_space(self):
        self.thisptr.print_space()
