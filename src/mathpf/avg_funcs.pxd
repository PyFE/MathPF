# Cython header — allows other .pyx files to cimport the C-level scalar kernels
# Usage:  from mathpf.avg_funcs cimport _avg_exp, _avg_inv, _avg_pow

cdef double _avg_exp(double x) noexcept nogil
cdef double _avg_inv(double x) noexcept nogil
cdef double _avg_pow(double x, double a) noexcept nogil
