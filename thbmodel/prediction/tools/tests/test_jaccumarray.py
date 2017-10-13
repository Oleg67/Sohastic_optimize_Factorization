import pytest
import numpy as np
from utils.accumarray import accum

try:
    from ..jaccumarray import jaccum
except (ImportError, OSError):
    pytest.skip("Can not import from jaccumarray")


def test_jaccum():
    nRaces = 10000
    horses_per_race = 10
    nData = nRaces * horses_per_race
    strata = np.tile(np.arange(nRaces), (horses_per_race, 1)).transpose().reshape(-1)
    x = np.random.randn(nData)

    xsum_ref = accum(strata, x)
    xsum = jaccum(strata, x)
    np.testing.assert_array_almost_equal(xsum, xsum_ref, decimal=10)


