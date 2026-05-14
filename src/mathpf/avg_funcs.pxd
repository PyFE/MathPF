# Cython header — allows other .pyx files to cimport the C-level scalar kernels
# Usage:  from mathpf.avg_funcs cimport _logrel, _powrel

cdef double _logrel(double x) noexcept nogil
cdef double _powrel(double x, double a) noexcept nogil
