from __future__ import division
import numpy as np
import numba as nb

from ...utils import get_logger, nbtools as nbt
from ...utils.accumarray import uaccum

from . import parameters

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


def effective_coefficients(step1_coefs, step2_coefs):
    """ Mix step1 and step2 coefficients, so that they can be applied in one step. """
    coefs = step2_coefs.copy()
    for i in xrange(2):  # for back and lay
        coefs[:, i] = step2_coefs[:, i] + step1_coefs[:, i] * step2_coefs[:, 2]
    return coefs

eff_coefs = effective_coefficients(step1coefs, step2coefs)


def _prep_factor(x, logarithmize):
    fac = x.astype(np.float64)
    if logarithmize:
        np.log(fac, fac)
    if len(fac.shape) == 1:
        fac = fac.reshape((1, -1))
    return fac


def factors_to_probs(coefs, factors, event_id=None, logarithmize=None):
    factors = factors if isinstance(factors, list) else [factors]
    n = factors[-1].shape[-1]
    event_id = np.zeros(n, dtype=int) if event_id is None else event_id
    coefs = coefs.reshape((1, -1)) if len(coefs.shape) == 1 else coefs
    if np.any(~np.isfinite(coefs)):
        raise ValueError('Coefficients contain nans or infs.')
    if logarithmize is None:
        logarithmize = []
    logarithmize.extend([False] * (len(factors) - len(logarithmize)))
    count = 0
    temp = np.zeros((coefs.shape[0], n))
    with np.errstate(invalid='ignore'):
        for f, log_this in zip(factors, logarithmize):
            if log_this and np.any(f <= 0):
                raise ValueError('Cannot use logarithm with zero or negative entries.')
            fac = _prep_factor(f, log_this)
            temp += np.dot(coefs[:, count:count + fac.shape[0]], fac)
            count += fac.shape[0]
        if coefs.shape[1] != count:
            raise ValueError('Coefficients do not match the shape of factors.')

        if np.any(np.abs(temp) > 100):
            raise ValueError('Bad probabilitiy computation. Strengths contain too large values.')

        if np.any(np.isfinite(temp)):
            temp -= np.nanmean(temp)
            expS = np.exp(temp)
            for sl in xrange(coefs.shape[0]):
                temp[sl] = (expS[sl] / uaccum(event_id, expS[sl], func='nansum')).astype(np.float32)

        if np.any(temp < 0):
            raise ValueError('Probabilities contain negative entries.')

    return temp.squeeze()


def missing_factors(av, run_id):
    row_id = av.row_lookup(run_id)
    return [factor for factor in factornames_trimmed if av.get_value(factor, row_id) is None]


@nb.njit
def compute_probabilities(ei, ec, sc, step1probs, coefs):
    # TODO: Testcase to make sure this applies to live betting
    Vmean = 0.0
    valid_sum = 0
    for j in range(ei.nruns):
        ec.temp64[j] = coefs[2] * np.log(step1probs[sc.row_id[j]])
        if ei.lprice[j] > 0:
            ec.temp64[j] += coefs[1] * np.log(1 / ei.lprice[j])
        else:
            ec.temp64[j] = np.nan
        if ei.bprice[j] > 0:
            ec.temp64[j] += coefs[0] * np.log(1 / ei.bprice[j])
        else:
            ec.temp64[j] = np.nan
        if not np.isnan(ec.temp64[j]):
            Vmean += ec.temp64[j]
            valid_sum += 1
    if valid_sum > 0:
        Vmean /= valid_sum

    # Subtract mean (so that exponential is less likely to blow up) and exponentiate
    Vsum = 0.0
    for j in xrange(ei.nruns):
        if not np.isnan(ec.temp64[j]):
            ec.temp64[j] = np.exp(ec.temp64[j] - Vmean)
            Vsum += ec.temp64[j]

    # Normalize probs
    if Vsum > 0:
        for j in xrange(ei.nruns):
            ei.probs[j] = ec.temp64[j] / Vsum


