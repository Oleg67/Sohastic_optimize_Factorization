"""
Compatibility module for previous accumarray implementation, using new aggregate module.
"""
import numpy as np
import numba as nb
from numpy_groupies import (aggregate_nb as accum, aggregate_np as accum_np, unpack,
                            aggregate_py as accum_py)  # @UnusedImport for reimport

from .intnan import isnan, nanval, nancumsum
from .math import unique_custom


def uaccum(group_idx, a, **kwargs):
    return unpack(group_idx, accum(group_idx, a, **kwargs))


@nb.njit
def step_count(group_idx):
    """ Determine the size of the result array
        for contiguous data
    """
    cmp_pos = 0
    steps = 1
    if len(group_idx) < 1:
        return 0
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            steps += 1
    return steps


@nb.njit
def step_indices(group_idx):
    """ Get the edges of areas within group_idx, which are filled 
        with the same value
    """
    ilen = step_count(group_idx) + 1
    indices = np.empty(ilen, np.int64)
    indices[0] = 0
    indices[-1] = group_idx.size
    cmp_pos = 0
    ri = 1
    for i in range(len(group_idx)):
        if group_idx[cmp_pos] != group_idx[i]:
            cmp_pos = i
            indices[ri] = i
            ri += 1
    return indices


def step_iter(group_idx):
    indices = step_indices(group_idx)
    for i in xrange(len(indices) - 1):
        yield slice(indices[i], indices[i + 1])


def accum_sort(accmap, x):
    '''Sorts x within each group defined by accmap. Feuer gemacht mit Indexgymnastik. Author und Copyright bis 3540: Arthur ;-)'''
    _, goodaccmap = unique_custom(accmap, return_inverse=True)
    q = np.concatenate(accum_np(goodaccmap, x, func='sort'))
    # disambiguate strata in accmap
    _, disambaccmap = unique_custom(goodaccmap + np.arange(len(accmap)) / 1e10, return_inverse=True)
    return q[disambaccmap]


def accum_cumsum(accmap, x, shift=False):
    '''
    N -> N accum operation of cumsum. Perform cumulative sum for each stratum.
    Example:
    accmap = array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    x = array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8], dtype=float)
    Returns:
    array([ 3,  4,  5,  6, 15,  9, 15, 22,  7,  0, 15, 17,  6, 14, 31, 39])
    '''
    accmap_sorted = np.sort(accmap, kind='mergesort')
    sortidx = np.argsort(accmap, kind='mergesort')
    invsortidx = np.argsort(sortidx, kind='mergesort')
    bad_accmap = isnan(accmap_sorted)
    accmap_sorted[bad_accmap] = np.max(accmap) + 1

    x_sorted = x[sortidx]
    x_sorted_shifted = x_sorted.copy()
    # shift x_sorted to the right, such that current entries are not taken into account
    if shift:
        last_entry = unpack(accmap_sorted, accum(accmap_sorted, np.arange(len(x)), func='max'))
        x_sorted_shifted[last_entry] = nanval(x)
        x_sorted_shifted = np.concatenate((np.array([np.nan]), x_sorted_shifted[:-1]))
    cumsum_x_sorted = nancumsum(x_sorted_shifted)

    increasing = np.arange(len(x))
    increasing[isnan(x_sorted_shifted)] = len(x) + 1
    first_entry = unpack(accmap_sorted, accum(accmap_sorted, increasing, func='min'))

    cumsum_x_sorted_first_entry = np.full(len(x), np.nan)
    anyentry = first_entry < len(x)
    cumsum_x_sorted_first_entry[anyentry] = cumsum_x_sorted[first_entry[anyentry]]

    bad = isnan(x_sorted_shifted) & (np.arange(len(x)) < first_entry)
    cumsum_x_sorted[bad] = nanval(x_sorted_shifted)
    x_sorted_shifted[isnan(x_sorted_shifted)] = 0
    cumsum_x_sorted[anyentry] += -cumsum_x_sorted_first_entry[anyentry] + x_sorted_shifted[first_entry[anyentry]]
    cumsum_x_sorted[bad_accmap] = nanval(x)
    return cumsum_x_sorted[invsortidx]


def accum_n2n(accmap, x, func):
    '''Arbirtrary N -> N accum operation. This is slow.'''
    accmap_sorted = np.sort(accmap, kind='mergesort')
    sortidx = np.argsort(accmap, kind='mergesort')
    invsortidx = np.argsort(sortidx, kind='mergesort')
    x_sorted = x[sortidx]

    result = np.zeros_like(x)
    steps = step_indices(accmap_sorted)
    for i in xrange(len(steps) - 1):
        idx = slice(steps[i], steps[i + 1])
        result[idx] = func(x_sorted[idx])

    return result[invsortidx]
