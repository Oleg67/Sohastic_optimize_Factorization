import numpy as np

from ...utils.accumarray import uaccum


class Probabilities(object):

    @classmethod
    def compute(cls, coefs, factors, strata=None, rng=None, full_range=True):
        """ Do usual checks and preprocessing. If the checked conditions are granted
            already, use _compute instead.
        """
        # factors.shape = ncoefs, nruns
        assert coefs.shape[0] == factors.shape[0], "coefs doesn't match with factors"
        assert coefs.shape[1] == 1, "coefs must be a column vector"

        if strata is None:
            strata = np.zeros(factors.shape[1], dtype=int)

        if rng is None:
            probs = cls._compute(coefs, factors, strata)
        else:
            factors_rng = factors[:, rng]
            strata_rng = np.unique(strata[rng], return_inverse=True)[1]
            probs = cls._compute(coefs, factors_rng, strata_rng)

        if not full_range or rng is None:
            return probs
        else:
            full_probs = np.zeros(factors.shape[1])
            full_probs[rng] = probs
            return full_probs

    @classmethod
    def _compute(cls, coefs, factors, strata):
        raise NotImplementedError


class ProbabilitiesCL(Probabilities):
    @classmethod
    def _compute(cls, coefs, factors, strata):
        V = np.dot(coefs.transpose(), factors).reshape(-1)
        assert np.all(~np.isnan(V)), 'At least one horse strength is not a number.'
        # subtract race-wise mean in order to avoid to compute exponentials on very large numbers
        V -= uaccum(strata, V, func='nanmean')
        expV = np.exp(V)
        expV[expV == 0] = 1e-10
        expV[np.isinf(expV)] = 1e20
        return expV / uaccum(strata, expV)


class ProbabilitiesHorseStrength(Probabilities):
    '''Like CL model, but each horse is assigned a constant strength. 
    The job is to figure out that strength based on win flags'''
    @classmethod
    def _compute(cls, ids, strengths, strata):
        assert np.max(ids) + 1 == len(strengths), 'The number of different IDs does not match the number of different strengths'
        run_strengths = strengths[ids]
        expV = np.exp(run_strengths)
        prob = expV / uaccum(strata, expV)
        return prob

