from __future__ import division
from os import listdir
import numpy as np
from ...utils import settings
from ...utils.accumarray import uaccum
from ...utils.arrayview import ArrayView
from ..tools.helpers import strata_scale_down, spread_from_av

def load_slices(path=None):
    if path is None:
        path = settings.paths.join()
    slicefiles = [f for f in listdir(path) if f.find('slice') > 0 and f.find('av.bcolz') > 0]
    tsav = {}
    for sl in xrange(len(slicefiles)):
        tsav[sl] = ArrayView.from_file(path + '/brain_final2_slice_%i.av.bcolz' % sl)
    return tsav


def ts_probs(ts, avrun_id, factors, valid1, step1_coefs, step2_coefs, verbose=False):
    nSlices = step2_coefs.shape[0]
    nModels = step1_coefs.shape[1]
    ts_strength = np.zeros(len(ts)) * np.nan

    if verbose:
        print 'Computing time series probabilities...'
    for sl in xrange(nSlices - 1):
        if verbose:
            print 'Slice: %s' % sl
        sl_idx = ts.slice_idx == sl
        lb = -np.log(ts.back_price[sl_idx])
        lb[ts.nonrunner[sl_idx]] = np.log(1e-3)
        ll = -np.log(ts.lay_price[sl_idx])
        ll[ts.nonrunner[sl_idx]] = np.log(1e-3)
        ts_strength[sl_idx] = step1_coefs[sl, 0] * lb + step1_coefs[sl, 1] * ll
        for m in xrange(2, nModels):
            if valid1[m]:
                f = spread_from_av(ts.run_id[sl_idx], avrun_id, factors[m - 2, :], cut_target=False)[0]
                ts_strength[sl_idx] += step1_coefs[sl, m] * f
        ts_strength[sl_idx] *= step2_coefs[sl, 2]
        ts_strength[sl_idx] += step2_coefs[sl, 0] * lb + step2_coefs[sl, 1] * ll

    ts_probs = np.exp(ts_strength)
    tsstrata = strata_scale_down(ts.strata())
    ts_probs /= uaccum(tsstrata, ts_probs)
    return ts_probs


def print_slice_likelihoods(row_lut, result, tsrun_id, slice_idx, ts_prob):
    # row_lut = av._row_lut()
    winners = np.zeros_like(slice_idx)
    valid = tsrun_id < len(row_lut)
    winners[valid] = (result[row_lut[tsrun_id[valid]]] == 1)
    nSlices = np.nanmax(slice_idx) + 1
    LL = np.zeros(nSlices)
    for sl in xrange(nSlices):
        idx = (slice_idx == sl)

        # print likelihoods of is1, is2 and oos
        LL[sl] = np.nanmean(np.log(ts_prob[idx & winners])) * 1000
        # BIC1 = compute_BIC(LL1, sum(valid1), sum(winners))
        print 'Slice: %i,  log-likelihood: %.2f\n' % (sl, LL[sl])
    return LL


def valid_races(strata, result, course, depth=1):
    # select only races where the number of winners, second placed, third placed, ..., depth-placed is exactly 1.
    good_depth = np.ones_like(strata, dtype=bool)
    for r in xrange(1, depth + 1):
        good_depth &= uaccum(strata, result == r) == 1

    # big selector, which data to use in model
    eval_rng = (course <= 88) & good_depth
    return eval_rng


def print_factor_order(stats, factornames, valid1=None):
    '''compute factor order over all slices according to their t-score sum'''
    if valid1 is None:
        valid1 = np.ones(len(factornames) + 2, dtype=bool)
    sum_t_scores = np.sum(stats.student_t[:, 2:][:, valid1[2:]], axis=0)
    si = np.argsort(sum_t_scores)
    sorted_names = np.array(factornames)[valid1[2:]][si[::-1]]
    for i in xrange(len(si)):
        print '%3i:  %70s    t-score sum: %4.2f' % (i, sorted_names[i], sum_t_scores[si[::-1]][i])
    return sorted_names
