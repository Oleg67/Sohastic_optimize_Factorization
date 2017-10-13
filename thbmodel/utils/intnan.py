'''
Module for handling integer arrays with missing values.
Missing values are handled as special values and functions
to skip them in processing are provided here.

The special nan values are chosen depending on the array type.
Large negative values are used, so that especially python indexing 
(from the end) is unlikely to work.

'''
import sys
import numpy as np
import nbtools as nbt

__all__ = ['NANVALS', 'INTNAN', 'INTNAN32', 'INTNAN64', 'nanval', 'isnan', 'replacenan',
           'nanmin', 'nanmax', 'nanstd', 'nanvar', 'nansum',
           'nanequal', 'nanclose', 'nancumsum', 'allnan', 'anynan']

INTNAN32 = np.iinfo('int32').min  # -2147483648
INTNAN64 = long(np.iinfo('int64').min)  # -9223372036854775808
NANVALS = dict(d=np.nan, f=np.nan, e=np.nan, S='', l=INTNAN64, q=INTNAN32, i=INTNAN32,
               b=-1, h=-1, B=0, H=0, L=0, Q=0, O=None)

# For compatibility:
INTNAN = INTNAN32


def nanval(x):
    """ Return the corresponding NAN value for a column """
    return NANVALS.get(x.dtype.char)


def isnan(x):
    if isinstance(x, np.ndarray):
        nanval = NANVALS.get(x.dtype.char, 0)
        if nanval is np.nan:
            return np.isnan(x)
        elif nanval is None:
            return np.array([val is None for val in x])
        else:
            return x == nanval
    elif x in {np.nan, None, '', INTNAN, INTNAN64}:
        return True
    else:
        try:
            return np.isnan(x)
        except TypeError:
            return False


def replacenan(x, replacement=0):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.where(np.isnan(x), replacement, x)
    elif nanval is None:
        ret = np.zeros_like(x)
        x_flat = x.flat
        for i in xrange(x.size):
            if x_flat[i] is None:
                ret[i] = replacement
            else:
                ret[i] = x_flat[i]
        return ret
    else:
        return np.where(x == nanval, replacement, x)


def asfloat(x):
    if isinstance(x.dtype.type, np.floating):
        return x.copy()
    return replacenan(x, np.nan)


def anynan(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.anynan(x)
    elif nanval is None:
        return any(val is None for val in x.flat)
    elif issubclass(x.dtype.type, str):
        return np.any(x == '')
    else:
        return nbt.contains(x, nanval)


def allnan(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.allnan(x)
    elif nanval is None:
        return all(val is None for val in x.flat)
    elif issubclass(x.dtype.type, str):
        return np.all(x == '')
    else:
        return nbt.allequal(x, nanval)


def nanmax(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nanmax(x)
    else:
        try:
            return np.max(x[x != nanval])
        except ValueError as e:
            if 'zero-size' in str(e):
                return nanval
            else:
                raise


def nanmin(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nanmin(x)
    else:
        try:
            return np.min(x[x != nanval])
        except ValueError as e:
            if 'zero-size' in str(e):
                return nanval
            else:
                raise


def nanmaximum(x, y):
    """ Does the same as numpy.maximum (element-wise maximum operation of two arrays) but ignores NaNs """
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        with np.errstate(invalid='ignore'):
            return nbt.nanmaximum(x, y)
    else:
        z = np.maximum(x, y)
        badx = isnan(x)
        bady = isnan(y)
        z[badx] = y[badx]
        z[bady] = x[bady]
        return z


def nanminimum(x, y):
    """ Does the same as numpy.minimum (element-wise minimum operation of two arrays) but ignores NaNs """
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        with np.errstate(invalid='ignore'):
            return nbt.nanminimum(x, y)
    else:
        z = np.minimum(x, y)
        badx = isnan(x)
        bady = isnan(y)
        z[badx] = y[badx]
        z[bady] = x[bady]
        return z


def nansum(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nansum(x)
    else:
        return np.sum(x[x != nanval])


def nanprod(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nanprod(x)
    else:
        return np.prod(x[x != nanval])


def nancumsum(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    result = np.cumsum(replacenan(x))

    if anynan(x):
        # cumsum is undefined before the first valid number appears, so we need to replace
        # the nans starting from the beginning of the array
        # TODO: Finding the first instance of a value this way is huge overhead
        if nanval is np.nan:
            good_idx = np.where(~isnan(x))[0]
        else:
            good_idx = np.where(x != nanval)[0]
        if len(good_idx) > 0:
            result[:good_idx[0]] = nanval
        else:
            result[:] = nanval
    return result


def nanmean(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nanmean(x)
    else:
        with np.errstate(invalid='ignore'):
            return  np.mean(x[x != nanval])


def nanvar(x, ddof=0):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nanvar(x, ddof=ddof)
    else:
        with np.errstate(invalid='ignore'):
            return  np.var(x[x != nanval], ddof=ddof)


def nanstd(x, ddof=0):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return nbt.nanstd(x, ddof=ddof)
    else:
        with np.errstate(invalid='ignore'):
            return  np.std(x[x != nanval], ddof=ddof)


def nanequal(x, y):
    if x.dtype != y.dtype:
        raise TypeError("nanequal requires same data type: %s != %s" % (x.dtype, y.dtype))
    if issubclass(x.dtype.type, (np.integer, str)):
        return x == y
    else:
        return nbt.nanequal(x, y)


def nanclose(x, y, delta=sys.float_info.epsilon):
    if x.dtype != y.dtype:
        raise TypeError("nanclose requires same data type: %s != %s" % (x.dtype, y.dtype))
    if issubclass(x.dtype.type, np.integer):
        nanval = NANVALS.get(x.dtype.char, 0)
        return nbt.nanclose_int(x, y, delta, nanval)
    elif issubclass(x.dtype.type, str):
        return x == y
    else:
        return nbt.nanclose(x, y, delta)
