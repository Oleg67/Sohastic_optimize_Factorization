import pytest
import numpy as np

from utils.containers import AttrDict
from utils.arrayview import ArrayView
from prediction.models.model_parameters import ModelParameters, factor_build_start
from ..factor_management import Factor, FactorList


def test_model_parameters():
    av = ArrayView.dummy(sim_start=factor_build_start)
    mod = ModelParameters(av, checked=False)
    assert mod.is_default()
    mod = ModelParameters(av, depth=3, verbose=True, checked=False)
    assert mod.is_default()
    mod = ModelParameters(av, transformation_degree=1, checked=False)
    assert not mod.is_default()
    mod = ModelParameters(av, depth=10, build_start=100, checked=False)
    assert mod.build_start == 100
    assert not mod.is_default()
    with pytest.raises(TypeError):
        ModelParameters(av, depth=10, invalid_param=100, checked=False)


@pytest.fixture(scope='module')
def model(request):
    seed = request.config.getoption("--seed") or np.random.randint(10, 20)
    av = ArrayView.dummy(events=200, seed=seed)
    build_start = np.min(av.start_time[av.start_time > 0])  # factor build start
    build_end = build_start + (np.max(av.start_time) - build_start) * 0.66  # factor build end
    oos_start = build_start + (np.max(av.start_time) - build_start) * 0.85
    mod = ModelParameters(av, build_start=build_start, build_end=build_end, oos_start=oos_start)
    return mod


def test_standardize_factor():
    col = np.array([1, np.nan, 3, 2, 3, 7])
    std_mask = np.array([True, True, True, False, False, False])
    strata = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    factor = Factor(col.copy(), 'some_factor')
    factor.standardize(std_mask, strata, fill_missing=False)
    soll1 = np.array([-1, np.nan, 1, -2, -1, 3], dtype=np.float32) / np.std([-1, 1], ddof=1)
    np.testing.assert_array_almost_equal(factor.col, soll1)

    factor = Factor(col.copy(), 'some_factor')
    factor.standardize(std_mask, strata, fill_missing=True)
    soll2 = np.array([-1, 0, 1, -2, -1, 3], dtype=np.float32) / np.std([-1, 1], ddof=1)
    np.testing.assert_array_almost_equal(factor.col, soll2)
    assert np.abs(np.nanmean(factor.col[np.isfinite(factor.col)])) < 1e-7, 'New nan-mean should be zero.'


def test_subtract_mean_general():
    factor = np.array([1, np.nan, 3, 2, 3, 7])
    strata = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    f = Factor(factor, 'x')
    f._subtract_mean(strata)
    soll = np.array([-1, np.nan, 1, -2, -1, 3])
    np.testing.assert_array_almost_equal(f.col, soll)


def test_subtract_mean_all_nan():
    factor = np.ones(6) * np.nan
    strata = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    f = Factor(factor, 'x')
    f._subtract_mean(strata)
    np.testing.assert_array_almost_equal(f.col, np.full(6, np.nan))


def test_subtract_mean_some_all_nan():
    factor = np.array([np.nan, np.nan, np.nan, 2, 3, 7])
    strata = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    f = Factor(factor, 'x')
    f._subtract_mean(strata)
    soll = np.array([np.nan, np.nan, np.nan, -2, -1, 3])
    np.testing.assert_array_almost_equal(f.col, soll)


@pytest.fixture
def lmock(request):
    nRaces = 100
    horses_per_race = 10
    nData = nRaces * horses_per_race
    nFactors = 3
    strata = np.tile(np.arange(nRaces), (horses_per_race, 1)).transpose().reshape(-1)
    factors = np.random.randn(nFactors, nData)
    # scatter some NaN's
    factors[np.random.rand(nFactors, nData) > 0.8] = np.nan
    standardization_mask = np.zeros(nData, dtype=bool)
    standardization_mask[:int(nData * 0.7)] = True
    return AttrDict(locals())


def test_missing_value_regression_not_any_allgood(model):
    factors = np.array([[-0.9379, -2.8922, np.nan, np.nan, -0.2047, np.nan],
        [    np.nan, 1.0077, -0.3842, 1.0997, -0.0601, -0.0745],
        [ 1.1244, np.nan, -1.3453, np.nan, 0.6897, -0.3437]])
    fl = FactorList.from_matrix(factors)
    model.build_mask = np.array([ True, True, True, True, False, False], dtype=bool)
    with pytest.raises(AssertionError):
        fl.missing_value_regression(model)


def test_uniqueness_of_missing_patterns():
    fl = FactorList.from_matrix(np.empty((1, 1)))
    n = 13
    combs = np.zeros((n, 2 ** n), dtype=np.int8)
    for i in xrange(2 ** n):
        s = bin(i)[2:]
        combs[-len(s):, i] = np.array(list(s)).astype(np.int8)
    patterns = fl.missing_patterns(combs)
    assert len(np.unique(patterns)) == len(patterns)


def test_fill_missing_values(lmock, model):
    factors0 = lmock.factors.copy()
    model.build_mask = lmock.standardization_mask.copy()
    # factors0 = np.array([[-0.9379, -2.8922,  1.1573,     np.nan, -0.2047,     np.nan],
    #    [    np.nan,  1.0077, -0.3842,  1.0997, -0.0601, -0.0745],
    #    [ 1.1244,     np.nan, -1.3453,     np.nan,  0.6897, -0.3437]])
    # in_sample_mask = np.array([ True,  True,  True,  True, False, False], dtype=bool)
    factors = factors0.copy()
    fl = FactorList.from_matrix(factors)
    coefs = fl.missing_value_regression(model)[1]

    factors = factors0.copy()
    fl = FactorList.from_matrix(factors)
    fl.fill_missing_values(model)
    factors = fl.asmatrix()

    # a simple test: every missing entry has to end up computable as a linear combination of all other entries, weighted by the coefficients
    for i in xrange(factors.shape[1]):
        F = np.where(np.isnan(factors0[:, i]))[0]  # indices of missing entries at run i
        for t in xrange(len(F)):
            np.testing.assert_almost_equal(factors[F[t], i], coefs[F[t], 0] + np.dot(coefs[F[t], 1:], factors[:, i]), decimal=2)
