import pytest
import numpy as np

from utils import MINUTE, HOUR, AttrDict
from utils.arrayview import ArrayView, TimeseriesView

from .. import prediction


def test_mixed_coeffs():
    nruns = 15
    bprice = np.random.random(nruns)
    lprice = np.random.random(nruns)
    factors = np.random.random((len(prediction.factornames_trimmed), nruns))

    slicenum = prediction.slicenum_by_ttm_jitted(HOUR)
    probs_step1 = prediction.predict_step1(bprice, lprice, factors, prediction.step1coefs[slicenum])
    probs_2step = prediction.predict_step2(bprice, lprice, probs_step1, prediction.step2coefs[slicenum])
    probs_mixed = prediction.predict_step1(bprice, lprice, factors, prediction.mixed_coefs[slicenum])
    np.testing.assert_array_almost_equal(probs_mixed, probs_2step)


def test_slicenum():
    def slicenum_by_ttm_ref(ttm, ttm_slice=prediction.TTM_SLICE):
        return np.maximum(np.searchsorted(ttm_slice, ttm, side='right') - 1, 0)
    for ttm_base in prediction.TTM_SLICE:
        for offset in [-0.1, 0, 0.1]:
            ttm = ttm_base + offset
            assert prediction.slicenum_by_ttm_jitted(ttm) == slicenum_by_ttm_ref(ttm)


@pytest.fixture(scope='module')
def tsav():
    av = ArrayView.dummy(events=5)
    ts = TimeseriesView.dummy_from_av(av, steps=5, nonrunner_chance=0)
    return AttrDict(av=av, ts=ts)


def test_raw_probs_unavailable(tsav):
    for sl in tsav.ts.slices():
        tsl = tsav.ts[sl]
        with np.errstate(invalid='ignore'):
            res = prediction.predict_step2_by_ttm(tsl.back_price, tsl.lay_price,
                                                  np.ones_like(tsl.back_price) * np.nan, 25 * HOUR)
        assert np.all(np.isnan(res)), "Values created even if no probs were available"
        assert len(tsl) == len(res)


def test_calculation_2stepped(tsav):
    for sl in tsav.ts.slices():
        tsl = tsav.ts[sl]
        probs = tsav.av.step1probs[tsav.av.row_lookup(tsl.run_id)]
        with np.errstate(invalid='ignore'):
            res = prediction.predict_step2_by_ttm(tsl.back_price, tsl.lay_price, probs, 10 * MINUTE)
        assert not np.any(np.isnan(res)), "Result contains nan values"
        assert len(tsl) == len(res)
        res.sort()
        assert np.all(np.diff(res)), "Prediction contains equal values"
