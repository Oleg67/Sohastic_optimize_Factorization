import numpy as np

from ...utils.math import sleep
from ...utils.accumarray import accum, unpack

from ..tools.helpers import strata_scale_down
from .probabilities import ProbabilitiesCL, ProbabilitiesHorseStrength

    

class Likelihood(object):
    probs_class = None

    def __init__(self, factors, strata, winners, lmbd=0):
        if self.probs_class:
            self.probs = self.probs_class()
        self.factors = factors
        self.strata = strata
        self.winners = winners
        self.lmbd = lmbd

    def compute(self, theta, ll=False, dll=False, d2ll=False):
        assert any([ll, dll, d2ll]), 'Compute either the likelihood or one of its derivatives.'
        out = []
        theta = theta.flatten()
        out = self._compute(out, theta, ll, dll, d2ll)
        if not out:
            return
        elif len(out) == 1:
            return out[0]
        else:
            return tuple(out)

    def _compute(self, out, theta, ll, dll, d2ll):
        raise NotImplementedError


class LikelihoodCL(Likelihood):
    probs_class = ProbabilitiesCL

    def preprocessing(self, theta, factors, strata):
        strength = np.dot(theta, factors)
        expstrength = np.exp(strength)
        expsum = accum(strata, expstrength)
        return strength, expstrength, expsum

    def likelihood(self, theta, strength, expsum):
        return np.sum(strength[self.winners] - np.log(expsum)) - self.lmbd * np.sum(theta ** 2)

    def first_derivative(self, factors, theta, winners, probs, lmbd):
        delta = winners - probs
        return np.dot(delta, factors.transpose()) - 2 * lmbd * theta

    def second_derivative(self, strata, factors, probs, lmbd):
        nVars = factors.shape[0]
        d2ll = np.full((nVars, nVars), np.nan)
        for i in xrange(nVars):
            expsumZi = accum(strata, factors[i, :] * probs)
            sleep()
            for j in xrange(i, nVars):
                expsumZj = accum(strata, factors[j, :] * probs)
                d2ll[i, j] = -np.sum(factors[i, :] * factors[j, :] * probs) + np.sum(expsumZi * expsumZj) - 2 * lmbd * (i == j)
                d2ll[j, i] = d2ll[i, j]
        return d2ll

    def probabilities(self, theta=None, expstrength=None, expsum=None):
        if expstrength is None or expsum is None:
            if theta is None:
                raise ValueError('Function inputs are missing.')
            expstrength, expsum = self.preprocessing(theta, self.factors, self.strata)[1:]
        return expstrength / unpack(self.strata, expsum)

    def _compute(self, out, theta, ll, dll, d2ll):
        strength, expstrength, expsum = self.preprocessing(theta, self.factors, self.strata)
        sleep()

        if ll:
            out.append(self.likelihood(theta, strength, expsum))
            sleep()

        if dll or d2ll:
            probs = self.probabilities(expstrength=expstrength, expsum=expsum)

        if dll:
            out.append(self.first_derivative(self.factors, theta, self.winners, probs, self.lmbd))
            sleep()

        if d2ll:
            out.append(self.second_derivative(self.strata, self.factors, probs, self.lmbd))

        return out


class LikelihoodHorseStrength(Likelihood):
    '''Like CL model, but each horse is assigned a constant strength. 
    The job is to figure out that strength based on win flags'''
    probs_class = ProbabilitiesHorseStrength

    def __init__(self, ids, strata, winners, lmbd=0):
        if self.probs_class:
            self.probs = self.probs_class()
        self.ids = strata_scale_down(ids)
        self.strata = strata_scale_down(strata)
        self.winners = winners
        self.lmbd = lmbd

    def compute_likelihood(self, strengths):
        assert np.max(self.ids) + 1 == len(strengths), 'The number of different IDs does not match the number of different strengths'

        run_strengths = strengths[self.ids]
        expV = np.exp(run_strengths)
        expsum = accum(self.strata, expV)
        ll = np.sum(run_strengths[self.winners] - np.log(expsum)) - self.lmbd * np.sum(strengths ** 2)
        sleep()

        return ll

    def compute_likelihood_derivative(self, strengths):
        assert np.max(self.ids) + 1 == len(strengths), 'The number of different IDs does not match the number of different strengths'

        run_strengths = strengths[self.ids]
        expV = np.exp(run_strengths)
        expsum = accum(self.strata, expV)
        prob = expV / unpack(self.strata, expsum)
        dll = accum(self.ids, self.winners - prob) - 2 * self.lmbd * strengths
        sleep()

        return dll
