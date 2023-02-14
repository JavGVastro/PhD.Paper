def ratio_empirical(rad, s0, a=1.0):
    """
    Simple tanh law in semi-log space to fit the seeing

    Reduction in B(r) is always 0.5 when r = 2 * s0
    Parameter `a` controls the slope of the transition.
    """
    x = np.log(rad / (2 * s0))
    y = np.tanh(a * x)
    return 0.5 * (1.0 + y)


def bfac(x):
    """
    Across-the board reduction in B(r) for x = s0 / r0

    Where s0 is RMS seeing width and r0 is correlation length
    """
    #return 1 / (1 + 4 * x ** 2)
    return np.exp(-x)


def seeing_empirical(r, s0, r0, a=0.75):
    return bfac(s0 / r0) * ratio_empirical(r, s0, a)
