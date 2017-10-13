""" A bunch of small functions that replace and improve former usage of numexpr and bottleneck """

import numpy as np
import numba as nb


@nb.njit(cache=True, nogil=True)
def anynan(x):
    for x_ in x.flat:
        if np.isnan(x_):
            return True
    return False


@nb.njit(cache=True, nogil=True)
def allnan(x):
    for x_ in x.flat:
        if not np.isnan(x_):
            return False
    return True


@nb.njit(cache=True, nogil=True)
def nanmin(x):
    ret = np.nan
    for i, x_ in enumerate(x.flat):
        if not np.isnan(x_):
            ret = x_
            break

    # TODO: This should be x.flat[i:]
    for x_ in x.flat:
        if x_ < ret:
            ret = x_
    return ret


@nb.njit(cache=True, nogil=True)
def nanmax(x):
    ret = np.nan
    for i, x_ in enumerate(x.flat):
        if not np.isnan(x_):
            ret = x_
            break

    # TODO: This should be x.flat[i:]
    for x_ in x.flat:
        if x_ > ret:
            ret = x_
    return ret


@nb.njit(cache=True, nogil=True)
def nanargmin(x):
    # TODO: This only works for 1d C-contiguous arrays
    pos = -1
    val = np.nan
    for i, x_ in enumerate(x):
        if not np.isnan(x_):
            pos = i
            val = x_
            break

    for j, x_ in enumerate(x[i:], i):
        if x_ < val:
            pos = j
            val = x_

    if pos == -1:
        raise ValueError("All-NaN slice encountered")
    return pos


@nb.njit(cache=True, nogil=True)
def nanargmax(x):
    # TODO: This only works for 1d C-contiguous arrays
    pos = -1
    val = np.nan
    for i, x_ in enumerate(x):
        if not np.isnan(x_):
            pos = i
            val = x_
            break

    for j, x_ in enumerate(x[i:], i):
        if x_ > val:
            pos = j
            val = x_

    if pos == -1:
        raise ValueError("All-NaN slice encountered")
    return pos


@nb.vectorize
def nanmaximum(x, y):
    if x > y:
        return x
    if np.isnan(y):
        return x
    return y


@nb.vectorize
def nanminimum(x, y):
    if x < y:
        return x
    if np.isnan(y):
        return x
    return y


@nb.njit(cache=True, nogil=True)
def nansum(x):
    ret = 0.0
    for x_ in x.flat:
        if not np.isnan(x_):
            ret += x_
    return ret


@nb.njit(cache=True, nogil=True)
def nanprod(x):
    ret = 1.0
    for x_ in x.flat:
        if not np.isnan(x_):
            ret *= x_
    return ret


@nb.njit(cache=True, nogil=True)
def nanmean(x):
    ret = 0.0
    cnt = 0
    for x_ in x.flat:
        if not np.isnan(x_):
            cnt += 1
            ret += x_
    return np.divide(ret, cnt)


@nb.njit(cache=True, nogil=True)
def nanvar(x, ddof=0):
    ret = 0.0
    cnt = 0
    # Inline that loop from nanmean, so that we can reuse cnt
    for x_ in x.flat:
        if not np.isnan(x_):
            cnt += 1
            ret += x_
    if cnt == ddof or cnt == 0:
        return np.nan
    mean = ret / cnt
    ex_mean = 0
    for x_ in x.flat:
        if not np.isnan(x_):
            inc = x_ - mean
            ex_mean += inc * inc
    return ex_mean / (cnt - ddof)


@nb.njit(cache=True, nogil=True)
def nanstd(x, ddof=0):
    return np.sqrt(nanvar(x, ddof=ddof))


@nb.vectorize
def nanequal(x, y):
    return (x == y) | (np.isnan(x) & np.isnan(y))


@nb.vectorize
def nanclose(x, y, delta):
    return (np.abs(x - y) < delta) | (np.isnan(x) & np.isnan(y))


@nb.vectorize
def nanclose_int(x, y, delta, nanval):
    return (np.abs(x - y) < delta) | ((x == nanval) & (y == nanval))


@nb.njit(cache=True, nogil=True)
def contains(x, scalar):
    """ Faster version of 'scalar in x' or np.any(x == scalar)"""
    for x_ in x.flat:
        if scalar == x_:
            return True
    return False


@nb.njit(cache=True, nogil=True)
def allequal(x, scalar):
    """ Faster version of np.all(x == scalar)"""
    for x_ in x.flat:
        if scalar != x_:
            return False
    return True
