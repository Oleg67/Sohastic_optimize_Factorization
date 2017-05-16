from __future__ import division
import numpy as np
import numba as nb
import bottleneck as bn

from utils import get_logger

from .probabilities import ProbabilitiesCL
from . import parameters

# TODO: Simplify and define the predict interface

logger = get_logger(__package__)

TTM_SLICE = np.concatenate((np.arange(0, 60, 15), np.arange(60, 300, 120), np.arange(300, 20 * 60, 180))) * 60

factornames = sorted(parameters.step1coefs)
factornames_trimmed = [f for f in factornames if not f.startswith('-')]

step1coefs = np.array([parameters.step1coefs[factor] for factor in factornames]).transpose()
step1coefs_trimmed = np.array([parameters.step1coefs[factor] for factor in factornames_trimmed]).transpose()
step2coefs = parameters.step2coefs
step2coefs_dummy = np.arange(len(TTM_SLICE) * 2).reshape((-1, 2)) / (len(TTM_SLICE) * 2)


def slicenum_by_ttm(ttm, ttm_slice=TTM_SLICE):
    return np.maximum(np.searchsorted(ttm_slice, ttm, side='right') - 1, 0)


@nb.njit
def slicenum_by_ttm_jitted(ttm):
    slicenum = len(TTM_SLICE) - 1
    for k in range(len(TTM_SLICE)):
        if TTM_SLICE[k] > ttm:
            slicenum = np.maximum(k - 1, 0)
            break

    return slicenum


def dynamic_slicenum(ttm, bp, lp, bvol, lvol, nr, coefs, edges):
    bp[nr] = 1000.0
    lp[nr] = 1000.0
    bvol[nr] = 0.001
    lvol[nr] = 0.001
    if (ttm <= 0) or np.any(np.isnan(bp + lp + bvol + lvol)):
        return -1  # invalid slice, don't bet

    strength = coefs[0] + coefs[1] * np.log(ttm)
    return np.searchsorted(edges, strength, side='right')


def mix_coefficients(step1_coefs, step2_coefs):
    """ Mix step1 and step2 coefficients, so that they can be applied in one step """
    mixed_coefs = np.zeros_like(step1_coefs)
    mixed_coefs[:, :2] = step2_coefs[:, :2] + step1_coefs[:, :2] * np.tile(step2_coefs[:, 2], (2, 1)).transpose()  # B/L prices
    mixed_coefs[:, 2:] = step1_coefs[:, 2:] * np.tile(step2_coefs[:, 2], (step1_coefs.shape[1] - 2, 1)).transpose()  # Other factors
    return mixed_coefs

mixed_coefs = mix_coefficients(step1coefs, step2coefs)

def effective_coefficients(step1_coefs, step2_coefs):
    """ Mix step1 and step2 coefficients, so that they can be applied in one step. 
    Just like mix_coefficients but assuming that the step 1 probabilities are already computed. """
    coefs = step2_coefs.copy()
    for i in xrange(2):  # for back and lay
        coefs[:, i] = step2_coefs[:, i] + step1_coefs[:, i] * step2_coefs[:, 2]
    return coefs

def prepare_step1(mixed_coefs, strata, factors):
    # TODO: This doesn't contain any nan check
    probs_step1 = np.full((mixed_coefs.shape[0], factors.shape[1]), np.nan)
    for i in xrange(mixed_coefs.shape[0]):
        probs_step1[i, :] = ProbabilitiesCL.compute(mixed_coefs[i, 2:].reshape((-1, 1)), factors, strata)
    return probs_step1.transpose()


def predict_step1(bprice, lprice, factors, coefs):
    """ New reference function for creating step1 probs with all factors weighted in directly.
        The result should still be fed through step2 prediction.
    """
    try:
        all_factors = np.concatenate((np.log(1 / bprice.astype(np.float64).reshape((1, -1))),
                                      np.log(1 / lprice.astype(np.float64).reshape((1, -1))),
                                      factors.astype(np.float64)), axis=0)
    except ValueError as e:
        raise ValueError(e.message + '\n' + '\n    '.join((str(bprice), str(lprice), str(factors))))
    V = np.dot(coefs, all_factors)
    V -= bn.nanmean(V)
    expV = np.exp(V)
    return (expV / bn.nansum(expV)).astype(np.float32)


def predict_step2(bprice, lprice, probs_step1, coefs):
    """ New reference function for mixing prices and raw probabilities together according to
        given coefficients grouped by market. Needs further speedup by replacing probs_cl
        with a faster and more strapped down version.
    """
    if np.all(np.isnan(probs_step1)):
        return probs_step1
    try:
        factors = np.log(np.concatenate((1 / bprice.reshape((1, -1)),
                                         1 / lprice.reshape((1, -1)),
                                         probs_step1.reshape((1, -1))), axis=0))
    except ValueError as e:
        raise ValueError(e.message + '\n' + '\n    '.join((str(bprice), str(lprice), str(probs_step1))))
    V = np.dot(coefs, factors)
    V -= bn.nanmean(V)
    expV = np.exp(V)
    return expV / bn.nansum(expV)


def predict_step2_by_ttm(bprice, lprice, probs_step1, ttm):
    coefs = step2coefs[slicenum_by_ttm_jitted(ttm)]
    return predict_step2(bprice, lprice, probs_step1, coefs)


def missing_factors(av, run_id):
    row_id = av.row_lookup(run_id)
    return [factor for factor in factornames_trimmed if av.get_value(factor, row_id) is None]


