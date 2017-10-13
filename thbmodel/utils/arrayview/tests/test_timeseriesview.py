import pytest
import numpy as np

from utils import intnan
from ..arrayview import ArrayView
from ..timeseriesview import TimeseriesView


@pytest.fixture(scope='module')
def dummyts():
    av = ArrayView.dummy()
    return TimeseriesView.dummy_from_av(av)


def lastvalid_ref(field, run_id):
    assert np.min(run_id) >= 0
    uruns = np.unique(run_id)
    last_valid_lookup = np.full(intnan.nanmax(uruns) + 1, intnan.INTNAN64, dtype=int)
    last_valid_idx = np.zeros(len(field), dtype=int)
    for i in xrange(len(run_id)):
        if not np.isnan(field[i]):
            last_valid_lookup[run_id[i]] = i
        last_valid_idx[i] = last_valid_lookup[run_id[i]]
    return last_valid_idx


def strata_ref(event_id, timestamp):
    assert len(event_id) == len(timestamp)
    assert (np.diff(timestamp) >= 0).all(), "Timestamps not sorted"
    strata = np.cumsum((timestamp[1:] != timestamp[:-1]) | (event_id[1:] != event_id[:-1]))
    return np.concatenate(([0], strata))


def test_lastvalid(dummyts):
    for col in dummyts:
        ref = lastvalid_ref(dummyts[col], dummyts.run_id)
        np.testing.assert_equal(dummyts.lastvalid(col), ref)


def test_strata(dummyts):
    ref = strata_ref(dummyts.event_id, dummyts.timestamp)
    np.testing.assert_equal(dummyts.strata(), ref)
