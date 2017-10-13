import numpy as np
from utils import intnan
from utils.accumarray import uaccum

def standardize(x, strata):
    """Standardize a numeric variable"""
    missing = intnan.isnan(x) | np.isinf(x)
    x[missing] = np.nan
    x = subtract_mean(x, strata)
    x = normalize_stddev(x, ~missing)
    return x

def subtract_mean(x, strata):
    """Subtract mean of factor at every valid value"""
    meanx = uaccum(strata, x, func='nanmean')
    invalid = np.isnan(meanx) | np.isinf(meanx)
    meanx[invalid] = np.mean(meanx[~invalid])
    x -= meanx
    return x

def normalize_stddev(x, mask):
    """Normalize standard deviation to 1."""
    correction = np.nanstd(x[mask], ddof=1)
    if correction > 1e-10:
        x /= correction
    return x

def cbind_1d(x):
    """Bind a tuple of one dimensional arrays by column"""
    return np.concatenate(x).reshape((len(x[0]), len(x)), order='F')


def find_missing(x):
    """Find the missing values in an array"""
    if is_categorical(x):
        missing = x == ''
    else:
        missing = intnan.isnan(x) | np.isinf(x)
    return missing

def is_categorical(x):
    """Tests if the elements in an array are instances of str"""
    return isinstance(x[0], str)










