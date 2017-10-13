import pytest
import numpy as np

from utils import AttrDict
from utils.accumarray import accum, unpack
from utils.arrayview.simdata import draw_winners

from ..likelihoods import LikelihoodCL
from ..jlikelihoods import jLikelihoodCL

implementations = [LikelihoodCL, jLikelihoodCL]


class LikelihoodCLref(object):
    def __init__(self, factors, strata, winners, lmbd=0):
        self.factors = factors
        self.strata = strata
        self.winners = winners
        self.lmbd = lmbd

    def compute(self, theta, flag=(True, False, False)):
        nVariables, nRuns = self.factors.shape

        ll = np.nan
        dll = np.full(nVariables, np.nan)
        d2ll = np.full((nVariables, nVariables), np.nan)

        if any(flag):
            # calculates exp(sum(theta * factors)) for each race separately
            # TODO: Use ProbabilitiesCL instead of own implementation for the calculations
            reptheta = np.tile(theta, nRuns)
            expV = np.exp(np.sum(reptheta * self.factors, 0))
            expsum = accum(self.strata, expV)

        if flag[0]:
            ll = np.sum(np.dot(theta.reshape((1, -1)), self.factors[:, self.winners]) - np.log(expsum)) - self.lmbd * np.sum(theta ** 2)

        if any(flag[1:3]):
            # factors * exp(theta * factors) (a derivative term)
            expVfactors = np.tile(expV, (nVariables, 1)) * self.factors
            # sum previous term for each race
            expsumZ = np.zeros((nVariables, len(np.unique(self.strata))))
            for i in xrange(nVariables):
                expsumZ[i, :] = accum(self.strata, expVfactors[i, :])

        if flag[1]:
            # ratio that comes from deriving the logarithm
            expratio = expsumZ / np.tile(expsum, (nVariables, 1))
            dll = np.sum(self.factors[:, self.winners] - expratio, 1) - 2 * self.lmbd * theta.reshape((1, -1))

        if flag[2]:
            # ratio that comes from deriving a ratio: quotient rule
            for i in xrange(nVariables):
                for j in xrange(nVariables):
                    expsumfactors = accum(self.strata, self.factors[i, :] * self.factors[j, :] * expV)
                    d2ll[i, j] = -np.sum((expsumfactors * expsum - expsumZ[i, :] * expsumZ[j, :]) / expsum ** 2) - 2 * self.lmbd * (i == j)

        return [ll, dll, d2ll]


@pytest.fixture(params=implementations, ids=lambda x: x.__name__)
def lmock(request):
    likelihood_class = request.param
    nRaces = 1000
    horses_per_race = 10
    nData = nRaces * horses_per_race
    nFactors = 3
    strata = np.tile(np.arange(nRaces), (horses_per_race, 1)).transpose().reshape(-1)
    factors = np.random.randn(nFactors, nData)
    true_coefs = np.random.randn(nFactors)
    strengths = np.dot(true_coefs, factors)
    probs = np.exp(strengths) / unpack(strata, accum(strata, np.exp(strengths)))
    winners = draw_winners(strata, probs)
    theta = np.random.randn(nFactors)
    lmbd = 10

    func_soll = LikelihoodCLref(factors, strata, winners, lmbd).compute
    llclass = likelihood_class(factors, strata, winners, lmbd)
    return AttrDict(locals())


def test_likelihood(lmock):
    ll_soll = lmock.func_soll(lmock.theta.reshape((-1, 1)), flag=(True, False, False))[0]
    strength, _, expsum = lmock.llclass.preprocessing(lmock.theta, lmock.factors, lmock.strata)
    ll_ist = lmock.llclass.likelihood(lmock.theta, strength, expsum)
    np.testing.assert_array_almost_equal(ll_soll, ll_ist)


def test_first_derivative_ref(lmock):
    dll_soll = lmock.func_soll(lmock.theta.reshape((-1, 1)), flag=(False, True, False))[1].flat
    probs = lmock.llclass.probabilities(lmock.theta)
    dll_ist = lmock.llclass.first_derivative(lmock.factors, lmock.theta, lmock.winners, probs, lmock.lmbd)
    np.testing.assert_array_almost_equal(dll_soll, dll_ist, decimal=10)


def test_second_derivative_ref(lmock):
    d2ll_soll = lmock.func_soll(lmock.theta.reshape((-1, 1)), flag=(False, False, True))[2]
    probs = lmock.llclass.probabilities(lmock.theta)
    d2ll_ist = lmock.llclass.second_derivative(lmock.strata, lmock.factors, probs, lmock.lmbd)
    np.testing.assert_array_almost_equal(d2ll_soll, d2ll_ist, decimal=10)


def test_first_derivative(lmock):
    probs = lmock.llclass.probabilities(lmock.theta)
    dll = lmock.llclass.first_derivative(lmock.factors, lmock.theta, lmock.winners, probs, lmock.lmbd)

    strength, _, expsum = lmock.llclass.preprocessing(lmock.theta, lmock.factors, lmock.strata)
    ll1 = lmock.llclass.likelihood(lmock.theta, strength, expsum)
    
    d = 1e-10
    for i in xrange(lmock.factors.shape[0]):
        theta1 = lmock.theta.copy()    
        theta1[i] += d
        strength, _, expsum = lmock.llclass.preprocessing(theta1, lmock.factors, lmock.strata)
        ll2 = lmock.llclass.likelihood(theta1, strength, expsum)    
        np.testing.assert_almost_equal((ll2 - ll1) / d / dll[i], 1, decimal=2)


def test_second_derivative(lmock):
    probs = lmock.llclass.probabilities(lmock.theta)
    d2ll = lmock.llclass.second_derivative(lmock.strata, lmock.factors, probs, lmock.lmbd)
    dll1 = lmock.llclass.first_derivative(lmock.factors, lmock.theta, lmock.winners, probs, lmock.lmbd)
    
    d = 1e-10
    for i in xrange(1): #lmock.factors.shape[0]):
        theta1 = lmock.theta.copy()    
        theta1[i] += d
        probs = lmock.llclass.probabilities(theta1)
        dll2 = lmock.llclass.first_derivative(lmock.factors, theta1, lmock.winners, probs, lmock.lmbd)
        np.testing.assert_array_almost_equal((dll2 - dll1) / d / d2ll[:,i], np.ones_like(dll1), decimal=2)

