import numpy as np
import pytest

from utils.arrayview import ArrayView
from utils.containers import AttrDict
from .. import helpers as hp

@pytest.fixture()
def mock():
    av = ArrayView.dummy(events=5)
    return AttrDict(locals())


def test_strata_mask_compatible(mock):
    mask = np.zeros(len(mock.av))
    hp.strata_mask_compatible(mock.av.event_id, mask)

    mask = np.ones(len(mock.av))
    hp.strata_mask_compatible(mock.av.event_id, mask)

    mask = np.zeros(len(mock.av))
    mask[:5] = True
    with pytest.raises(AssertionError):
        hp.strata_mask_compatible(mock.av.event_id, mask)


def test_check_results():
    strata = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    result = np.array([9, 4, 3, 6, 5, 10, 8, 1, 7, 2, 7, 5, 4, 3, 10, 2, 8, 9, 6, 1, 6, 9, 8, 10, 2, 3, 7, 5, 4, 1, 7, 1, 2, 9, 3, 10, 5, 4,
                       8, 6, 5, 1, 2, 8, 6, 9, 4, 10, 3, 7], dtype=np.int8)
    hp.check_results(strata, result, max_result=5)

    result1 = result.copy()
    result1[3] = 2
    with pytest.raises(AssertionError):
        hp.check_results(strata, result1, max_result=5)

    result1 = result.copy()
    result1[7] = 3
    with pytest.raises(AssertionError):
        hp.check_results(strata, result1, max_result=5)


def test_combine_ids():
    a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                  2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                  4, 4, 4, 4])
    b = np.array([3, 2, 4, 4, 4, 0, 1, 3, 3, 2, 3, 2, 3, 2, 2, 2, 4, 1, 0, 2, 0, 1, 0,
                  0, 1, 3, 0, 3, 3, 4, 1, 4, 4, 4, 3, 1, 2, 1, 2, 0, 0, 1, 0, 2, 4, 3,
                  4, 1, 1, 0])
    ref = np.array([ 3, 2, 4, 4, 4, 0, 1, 3, 3, 2, 8, 7, 8, 7, 7, 7, 9,
                    6, 5, 7, 10, 11, 10, 10, 11, 12, 10, 12, 12, 13, 15, 18, 18, 18,
                    17, 15, 16, 15, 16, 14, 19, 20, 19, 21, 23, 22, 23, 20, 20, 19])
    assert np.all(hp.combine_ids([a, b]) == ref)
