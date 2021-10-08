import numpy as np


def bfunc00(r, r0, sig2, m):
    "Simple 3-parameter structure function"
    C = 1.0 / (1.0 + (r / r0)**m)
    return 2.0 * sig2 * (1.0 - C)

def seeing(r, s0):
    return (np.tanh((r / (2.0 * s0))**2))**2

def bfunc01(r, r0, sig2, m, s0):
    "Structure function with added seeing (scale `s0`)"
    return seeing(r, s0) * bfunc00(r, r0, sig2, m)

def bfunc02(r, r0, sig2, m, s0, noise):
    "Structure function with added seeing (scale `s0`) and noise"
    return seeing(r, s0) * bfunc00(r, r0, sig2, m) + noise

def seeing_empirical(r, s0, r0):
    return bfac(s0 / r0) * ratio_empirical(r, s0, a=0.75)

def bfunc03(r, r0, sig2, m, s0, noise):
    "Structure function with better seeing (scale `s0`) and noise"
    return seeing_empirical(r, s0, r0) * bfunc00(r, r0, sig2, m) + noise

def bfunc00s(r, r0, sig2, m):
    "Simple 3-parameter structure function"
    C = np.exp(-np.log(2) * (r / r0)**m)
    return 2.0 * sig2 * (1.0 - C)

def bfunc03s(r, r0, sig2, m, s0, noise):
    "Structure function with better seeing (scale `s0`) and noise"
    return seeing_empirical(r, s0, r0) * bfunc00s(r, r0, sig2, m) + noise

def ratio_empirical(rad, s0, a=1.0):
    """
    Simple tanh law in semi-log space to fit the seeing

    Reduction in B(r) is always 0.5 when r = 2 * s0
    Parameter `a` controls the slope of the transition.
    """
    x = np.log(rad / (2 * s0))
    y = np.tanh(a * x)
    return 0.5 * (1.0 + y)

def rtheo(rad, s0, s00=2, **kwds):
    """
    Theoretical ratio of B(r) structure function

    For an additional seeing FWHM of s0, assuming that
    the original seeing was s00 (widths assumed to add in quadrature)
    """
    s1 = np.hypot(s0, s00)
    return (
        ratio_empirical(rad, s1, **kwds)
        / ratio_empirical(rad, s00, **kwds)
    )

def bfac(x):
    """
    Across-the board reduction in B(r) for x = s0 / r0

    Where s0 is RMS seeing width and r0 is correlation length
    """
    return 1 / (1 + 4*x**2)
