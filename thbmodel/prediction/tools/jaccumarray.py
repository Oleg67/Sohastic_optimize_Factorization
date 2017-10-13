import numpy as np
import numba as nb


@nb.njit(nb.float64[:](nb.int64[:], nb.float64[:]), cache=True)
def jaccum_sum(strata, x):
    '''Only the sum is implemented yet'''
    s = -1
    xsum = np.zeros(np.max(strata) + 1)
    for i in xrange(len(strata)):
        if strata[i] != s:
            s = strata[i]
        xsum[s] = xsum[s] + x[i]
    return xsum


COMMANDS = {'sum': jaccum_sum, sum: jaccum_sum, np.sum: jaccum_sum}

def jaccum(strata, x, func='sum'):
    return COMMANDS[func](strata, x)

