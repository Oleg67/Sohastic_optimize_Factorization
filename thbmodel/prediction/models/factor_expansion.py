import numpy as np

from ...utils.math import triangular_upper_idx
from ...utils import dispdots


def taylor_expand_factors(X, degree):
    nFactors = X.shape[0]
    assert degree <= nFactors, 'Expansion degree must not be larger than the number of factors.'
    eXpanded = None
    for degree in xrange(1, degree + 1):
        # construct degree-sized tuples of various factor combinations
        idx = triangular_upper_idx(degree, nFactors)
        if degree > 1:
            diagonal_indices = np.arange(nFactors).reshape((-1, 1)) * np.ones(degree, dtype=int)
            idx = np.append(diagonal_indices, idx, axis=0)

        # compute various products of the factors
        Xprod = np.zeros((idx.shape[0], X.shape[1]))
        for i in xrange(idx.shape[0]):
            dispdots(i, 1)
            Xprod[i, :] = np.prod(X[idx[i, :], :], axis=0)

        # eXpanded - sexy, huh?
        if eXpanded is None:
            eXpanded = Xprod
        else:
            eXpanded = np.append(eXpanded, Xprod, axis=0)
    return eXpanded


def expand_factors(facs, degree=1):
    assert degree < 4, 'Only degrees less than 4 are implemented.'
    if len(facs) == 3:
        fb, fl, f1 = facs
        exp_factors = np.concatenate((fb, fl, f1), axis=0)
        if degree >= 2:
            exp_factors = np.concatenate((exp_factors, fb ** 2, fb * f1, f1 ** 2), axis=0)
        if degree >= 3:
            exp_factors = np.concatenate((exp_factors, fb ** 3, fb ** 2 * f1, fb * f1 ** 2, f1 ** 3), axis=0)
    if len(facs) == 2:
        fm, f1 = facs
        exp_factors = np.concatenate((fm, f1), axis=0)
        if degree >= 2:
            exp_factors = np.concatenate((exp_factors, fm ** 2, fm * f1, f1 ** 2), axis=0)
        if degree >= 3:
            exp_factors = np.concatenate((exp_factors, fm ** 3, fm ** 2 * f1, fm * f1 ** 2, f1 ** 3), axis=0)
    return exp_factors


def expand_factor(x, degree=1):
    '''From a vector x, get the matrix of all exponents of x, i.e. x, x**2, x**3, ..., x**degree'''
    assert len(x.shape) == 1
    exp_x = np.zeros((degree, len(x)))
    for n in xrange(degree):
        exp_x[n, :] = x ** (n + 1)
    return exp_x

