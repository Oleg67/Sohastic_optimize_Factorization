import numpy as np
import numba as nb

from .likelihoods import LikelihoodCL
from ..tools.jaccumarray import jaccum_sum


class jLikelihoodCL(LikelihoodCL):

    @nb.jit(cache=True)
    def preprocessing(self, theta, factors, strata):
        strength = np.dot(theta, factors)
        expstrength = np.exp(strength)
        expsum = jaccum_sum(strata, expstrength)
        return strength, expstrength, expsum


    @nb.jit(cache=True)
    def first_derivative(self, factors, theta, winners, probs, lmbd):
        delta = winners - probs
        return np.dot(delta, factors.transpose()) - 2 * lmbd * theta


    @nb.jit(cache=True)
    def second_derivative(self, strata, factors, probs, lmbd):
        nVars = np.int64(factors.shape[0])
        d2ll = np.full((nVars, nVars), np.nan, dtype=np.float64)
        expsumZ = np.zeros((nVars, np.max(strata) + 1), dtype=np.float64)
        for i in xrange(nVars):
            _jaccum_vecprod(strata, factors[i, :], probs, expsumZ[i, :])
            for j in xrange(i + 1):
                d2ll[i, j] = d2ll[j, i] = -_prodloop3(factors[i, :], factors[j, :], probs) + \
                                         _prodloop2(expsumZ[i, :], expsumZ[j, :]) - 2 * lmbd * (i == j)
        return d2ll


@nb.njit(cache=True)
def _jaccum_vecprod(strata, a, b, outvec):
    '''Performs the same as accum(strata, a*b)'''
    s = -1
    for i in xrange(len(strata)):
        if strata[i] != s:
            s = strata[i]
        outvec[s] += a[i] * b[i]


@nb.njit(cache=True)
def _prodloop2(a, b):
    x = 0
    for i in range(len(a)):
        x += a[i] * b[i]
    return x

@nb.njit(cache=True)
def _prodloop3(a, b, c):
    x = 0
    for i in range(len(a)):
        x += a[i] * b[i] * c[i]
    return x
