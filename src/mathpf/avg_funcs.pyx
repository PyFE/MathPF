# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Numerically stable relative functions.

All functions handle the x=0 limit analytically and use compensated
primitives (expm1, log1p) to avoid catastrophic cancellation.
"""
from libc.math cimport expm1, log1p
import numpy as np
cimport numpy as np

np.import_array()


# ── C-level scalar kernels (cimport-able by other Cython extensions) ──────────

cdef double _logrel(double x) noexcept nogil:
    """log(1 + x) / x,  limit 1 at x = 0."""
    if x == 0.0:
        return 1.0
    return log1p(x) / x


cdef double _powrel(double x, double a) noexcept nogil:
    """((1+x)^(a+1) - 1) / ((a+1)*x),  limit 1 at x = 0."""
    cdef double a1p = a + 1.0
    if x == 0.0:
        return 1.0
    if a1p == 0.0:          # a = -1: reduces to logrel
        return log1p(x) / x
    return expm1(a1p * log1p(x)) / (a1p * x)


# ── Python-callable array wrappers ────────────────────────────────────────────

def logrel(x):
    """
    Numerically stable computation of ``log(1 + x) / x``.

    Average of ``1/t`` over the interval ``[1, 1+x]``:

        (1/x) ∫_1^{1+x}  (1/t) dt  =  log(1 + x) / x

    Uses ``log1p`` to avoid cancellation for small ``x``.
    Limit at ``x = 0`` is ``1``.  Requires ``x > -1``.

    Args:
        x: scalar or ndarray, any numeric dtype; must satisfy ``x > -1``

    Returns:
        ``log(1 + x) / x``, dtype float64, same shape as ``x``
    """
    cdef double[::1] f, o
    cdef Py_ssize_t i

    x_arr = np.asarray(x, dtype=np.float64)
    if np.any(x_arr <= -1.0):
        raise ValueError("x must be greater than -1.")
    scalar = x_arr.ndim == 0
    flat = np.ascontiguousarray(x_arr.ravel())
    out  = np.empty_like(flat)
    f = flat
    o = out
    for i in range(f.shape[0]):
        o[i] = _logrel(f[i])
    return float(out[0]) if scalar else out.reshape(x_arr.shape)


def powrel(x, a):
    """
    Numerically stable computation of ``((1+x)^(a+1) - 1) / ((a+1) * x)``.

    Average of ``t^a`` over the interval ``[1, 1+x]``:

        (1/x) ∫_1^{1+x}  t^a dt  =  ((1+x)^(a+1) - 1) / ((a+1) x)

    The case ``a = -1`` reduces to ``logrel``.
    Uses ``expm1`` and ``log1p`` to avoid cancellation.
    Limit at ``x = 0`` is ``1`` for all ``a``.  Requires ``x > -1``.

    Args:
        x: scalar or ndarray, any numeric dtype; must satisfy ``x > -1``
        a: power exponent, scalar or ndarray broadcastable with ``x``

    Returns:
        ``((1+x)^(a+1) - 1) / ((a+1) * x)``, dtype float64,
        shape is the broadcast shape of ``x`` and ``a``
    """
    cdef double[::1] fx, fa, o
    cdef Py_ssize_t i

    x_arr = np.asarray(x, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    if np.any(x_arr <= -1.0):
        raise ValueError("x must be greater than -1.")
    bx, ba = np.broadcast_arrays(x_arr, a_arr)
    scalar  = bx.ndim == 0
    fx = np.ascontiguousarray(bx.ravel())
    fa = np.ascontiguousarray(ba.ravel())
    out = np.empty_like(fx)
    o   = out
    for i in range(fx.shape[0]):
        o[i] = _powrel(fx[i], fa[i])
    return float(out[0]) if scalar else out.reshape(bx.shape)
