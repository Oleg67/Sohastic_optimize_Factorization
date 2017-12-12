import pytest
import itertools
import numpy as np

from .. import intnan as inn, AttrDict


def test_nanval():
    assert inn.nanval(np.ones(10, dtype=np.int64)) == -2 ** 63
    assert inn.nanval(np.ones(10, dtype=np.int32)) == -2 ** 31
    assert np.isnan(inn.nanval(np.ones(10, dtype=np.float64)))
    assert np.isnan(inn.nanval(np.ones(10, dtype=np.float32)))


ninp_list = itertools.product(['small', 'large'],
                              ['nonans', 'nans', 'allnans'],
                              [np.int64, np.int32, np.float64, np.float32])

@pytest.fixture(params=ninp_list, ids=lambda x: '-'.join((x[0], x[1], x[2].__name__)))
def ninp(request):
    sizestr, nanstate, dtype = request.param
    if sizestr == 'small':
        size = 100
    else:
        if dtype == np.float32:
            size = 10000
        else:
            size = 100000

    a = np.arange(size, dtype=dtype)
    a_nanmask = np.zeros_like(a, dtype=bool)
    if nanstate == 'nans':
        a_nanmask[::2] = True
    elif nanstate == 'allnans':
        a_nanmask[:] = True
    a[a_nanmask] = inn.nanval(a)

    b = np.arange(size, dtype=dtype) + 1
    b_nanmask = np.zeros_like(b, dtype=bool)
    b_nanmask[::3] = True
    b[b_nanmask] = inn.nanval(a)
    return AttrDict(locals())


@pytest.mark.parametrize("val", [np.nan, inn.INTNAN32, inn.INTNAN64],
                         ids=['nan', 'INTNAN32', 'INTNAN64'])
def test_isnan(val):
    assert inn.isnan(val)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_array_element_isnan(dtype):
    nanval = inn.nanval(np.array([], dtype=dtype))
    arr = np.array([nanval], dtype=dtype)
    assert inn.isnan(arr[0])


@pytest.mark.parametrize("val", [0, -1])
def test_not_isnan(val):
    assert not inn.isnan(val)


def test_isnan_array(ninp):
    mask = np.zeros_like(ninp.a, dtype=bool)
    np.testing.assert_array_equal(ninp.a_nanmask, inn.isnan(ninp.a))


def test_anynan(ninp):
    assert inn.anynan(ninp.a) == (ninp.nanstate != 'nonans')


def test_allnan(ninp):
    assert inn.allnan(ninp.a) == (ninp.nanstate == 'allnans')


def test_replacenan(ninp):
    repl = inn.replacenan(ninp.a)
    assert not inn.anynan(repl)
    assert np.all(repl[ninp.a_nanmask] == 0)


def test_nanmax(ninp):
    if ninp.nanstate != 'allnans':
        assert inn.nanmax(ninp.a) == np.nanmax(ninp.a)
    else:
        np.testing.assert_equal(inn.nanmax(ninp.a), inn.nanval(ninp.a))


def test_nanmin(ninp):
    if ninp.nanstate == 'nonans':
        assert inn.nanmin(ninp.a) == 0
    elif ninp.nanstate == 'nans':
        assert inn.nanmin(ninp.a) == 1
    else:
        np.testing.assert_equal(inn.nanmin(ninp.a), inn.nanval(ninp.a))


def test_nanmaximum(ninp):
    res = inn.nanmaximum(ninp.a, ninp.b)
    # Wherever a is nan, value from b needs to be picked
    np.testing.assert_array_equal(res[ninp.a_nanmask], ninp.b[ninp.a_nanmask])
    # Wherever b is nan, value from a needs to be picked
    np.testing.assert_array_equal(res[ninp.b_nanmask], ninp.a[ninp.b_nanmask])
    # Wherever both are nan, expect nanval
    assert inn.allnan(res[ninp.a_nanmask & ninp.b_nanmask])


def test_nanminimum(ninp):
    res = inn.nanminimum(ninp.a, ninp.b)
    # Wherever a is nan, value from b needs to be picked
    np.testing.assert_array_equal(res[ninp.a_nanmask], ninp.b[ninp.a_nanmask])
    # Wherever b is nan, value from a needs to be picked
    np.testing.assert_array_equal(res[ninp.b_nanmask], ninp.a[ninp.b_nanmask])
    # Wherever both are nan, expect nanval
    assert inn.allnan(res[ninp.a_nanmask & ninp.b_nanmask])


def test_nansum(ninp):
    assert inn.nansum(ninp.a) == np.sum(ninp.a[~inn.isnan(ninp.a)])


def test_nancumsum(ninp):
    if ninp.nanstate == 'allnans':
        ref = np.full_like(ninp.a, inn.nanval(ninp.a))
    else:
        ref = np.cumsum(inn.replacenan(ninp.a))
        nanval = inn.nanval(ninp.a)
        for i, val in enumerate(ninp.a):
            if inn.isnan(val):
                ref[i] = nanval
            else:
                break
    np.testing.assert_array_equal(inn.nancumsum(ninp.a), ref)


def test_nanprod(ninp):
    if ninp.dtype == np.float32:
        ref_dtype = np.float64
    elif ninp.dtype == np.int32:
        ref_dtype = np.int64
    else:
        ref_dtype = ninp.dtype
    ref = np.prod(ninp.a[~inn.isnan(ninp.a)], dtype=ref_dtype)
    assert inn.nanprod(ninp.a) == ref


def test_nanmean(ninp):
    with np.errstate(invalid='ignore'):
        ref = np.mean(ninp.a[~inn.isnan(ninp.a)])
    np.testing.assert_equal(inn.nanmean(ninp.a), ref)


@pytest.mark.parametrize("ddof", [0, 1])
def test_nanstd(ninp, ddof, tolerance=1e-6):
    with np.errstate(invalid='ignore'):
        ref = np.std(ninp.a[~inn.isnan(ninp.a)], ddof=ddof)
    np.testing.assert_allclose(inn.nanstd(ninp.a, ddof=ddof), ref, atol=tolerance)


def test_nanvar(ninp, tolerance=1e-6):
    with np.errstate(invalid='ignore'):
        ref = np.var(ninp.a[~inn.isnan(ninp.a)])
    np.testing.assert_allclose(inn.nanvar(ninp.a), ref, atol=tolerance)


def test_nanequal(ninp):
    clone = ninp.a.copy()
    assert np.all(inn.nanequal(ninp.a, clone))
    if ninp.nanstate == 'allnans':
        clone[51] = 20
    else:
        clone[51] = inn.nanval(clone)
    assert np.count_nonzero(~inn.nanequal(ninp.a, clone)) == 1
    clone = clone.astype(np.int16)
    pytest.raises(TypeError, inn.nanequal, ninp.a, clone)


def test_nanclose(ninp, tolerance=1e-9):
    clone = ninp.a.copy()
    assert np.all(inn.nanclose(ninp.a, clone, tolerance))

    if issubclass(ninp.a.dtype.type, np.floating):
        clone += np.random.random(ninp.a.shape) * tolerance / 10
        assert np.all(inn.nanclose(ninp.a, clone, tolerance))

    clone[50] = -5
    assert np.count_nonzero(~inn.nanclose(ninp.a, clone, tolerance)) == 1

    if ninp.nanstate == 'allnans':
        clone[51] = 20
    else:
        clone[51] = inn.nanval(clone)
    assert np.count_nonzero(~inn.nanclose(ninp.a, clone, tolerance)) == 2
    clone = clone.astype(np.int16)
    pytest.raises(TypeError, inn.nanclose, ninp.a, clone, tolerance)
