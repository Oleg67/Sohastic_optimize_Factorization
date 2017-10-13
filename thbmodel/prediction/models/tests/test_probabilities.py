import pytest
import numpy as np

from utils.accumarray import accum, unpack

from ..probabilities import ProbabilitiesCL

def get_random_data(self, coef_len=30, nruns=1000):
    coefs = np.random.random(coef_len).reshape((-1, 1))
    factors = np.random.random((coef_len, nruns))
    strata = np.repeat(np.arange(nruns / 10, dtype=int), 10)
    return coefs, factors, strata


@pytest.mark.parametrize("i", range(1, 5))
def test_probabilities_cl(i):
    def reference(coefs, factors, strata):
        reptheta = np.tile(coefs, factors.shape[1])
        V = np.sum(reptheta * factors, axis=0)
        # substract race-wise mean in order to avoid to compute exponentials on very large numbers
        V -= unpack(strata, accum(strata, V, func='nanmean'))
        expV = np.exp(V)
        expsum = accum(strata, expV)
        return expV / unpack(strata, expsum)

    coefs, factors, strata = get_random_data(10 * i, 1000 * i)
    ref = reference(coefs, factors, strata)
    res = ProbabilitiesCL().compute(coefs, factors, strata)
    np.testing.assert_array_almost_equal(res, ref, decimal=6)

