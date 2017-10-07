from __future__ import division
import pprint
import pickle
import numpy as np

from utils import intnan, Folder, settings
from utils.accumarray import accum, uaccum, step_count
from utils.math import intersect

#TODO: Move one-time-used functions back to their original modules

def get_logpmkt(bsp, result, event_id, mask=None):
    """ Preprocessing for market odds """
    if mask is None:
        mask = np.ones(len(bsp), dtype=bool)
    logpmkt = -np.log(bsp[mask])
    logpmkt[result[mask] == -2] = -7.0
    validbsp = uaccum(event_id[mask], ~np.isnan(logpmkt), func=all)
    return logpmkt, validbsp


def strata_scale_down(strata):
    return np.unique(strata, return_inverse=True)[1]


def strata_contiguous(strata):
    assert step_count(strata) == len(np.unique(strata)), 'Strata are not contiguous.'
    # s = strata_scale_down(strata)
    # assert all(accum(s, np.arange(len(s)), func=lambda x: np.all(np.diff(x) == 1)))


def strata_contains(strata1, strata2):
    '''checks whether one strata contains another,
    i.e. whether strata2 is always constant within strata1'''
    assert all(accum(strata1, strata2, func=lambda x: all(x == x[0])))


def strata_exclusive(strata1, strata2):
    '''checks whether one strata is always constant within the other strata and vice versa'''
    assert all(accum(strata1, strata2, func=lambda x: all(x == x[0])))
    assert all(accum(strata2, strata1, func=lambda x: all(x == x[0])))


def strata_relabel_sorted(strata):
    '''
    Scales down strata and make it monotonically increasing with index,
    i.e. all(diff(new_strata) >= 0) will be True and 
    len(unique(new_strata)) == max(strata)+1 will be True
    Operation is only well-defined if strata are contiguous
    '''
    strata_contiguous(strata)
    startidx = np.unique(strata_scale_down(strata), return_index=True)[1]
    startidx_sorted = np.sort(startidx)
    startidx_sorted = np.append(startidx_sorted, len(strata))
    new_strata = np.zeros_like(strata)
    count = 0
    for i in xrange(len(startidx_sorted) - 1):
        new_strata[startidx_sorted[i]:startidx_sorted[i + 1]] = count
        count += 1
    return new_strata


def expand_strata(strata, n):
    # st = strata_scale_down(strata)
    nRaces = max(strata) + 1
    # nData = len(strata)
    q = np.arange(0, n * nRaces, nRaces).reshape((-1, 1))
    # return np.tile(q, (1, nData)) + np.tile(strata, (n, 1))
    return strata + q


def strata_mask_compatible(strata, mask):
    """Check that mask does not start or end in the middle of a stratum."""
    x = np.unique(accum(strata, mask, func='mean'))
    assert np.all(np.in1d(x, [0, 1]))

def check_results(strata, result, mask=None, max_result=1):
    """Check if every event has exactly one winner, one second placed, third placed etc. until max_result is reached."""
    if mask is not None:
        strata = strata_scale_down(strata[mask])
        result = result[mask]
    for r in xrange(1, max_result + 1):
        assert np.all(accum(strata, result == r) == 1), ('Result %i occurs either more than once or is missing entirely.' % r)


def arraymap(id1, id2, x1):
    '''
    Writes x1 entries in array of size id2 at those indices where id1 and id2 are equal, i.e.
    the mapping satisfies the test: x2[i] == x1[id1 == id2[i]] 
    Example:
    In [76]: x1
    Out[97]: array([-1., -2., -3., -4., -5., -6.])
    
    In [77]: id1
    Out[77]: array([ 2,  7,  4,  8,  5, 10])
    
    In [79]: id2
    Out[79]: array([7, 7, 4, 5, 4, 1, 7, 1, 5, 5, 5, 5])
    
    In [98]: arraymap(id1, id2, x1)
    Out[98]: array([ -2.,  -2.,  -3.,  -5.,  -3.,  nan,  -2.,  nan,  -5.,  -5.,  -5.,  -5.])
    '''
    assert len(id1) == len(x1), 'Source IDs <id1> must be as numerous as the source array <x1>.'
    assert len(id1) == len(np.unique(id1)), 'Source IDs for mapping should be uniquely defined.'

    _, i1, i2 = intersect(id1, id2)
    x2 = np.full_like(id2, intnan.NANVALS[x1.dtype.char])
    x2[i2] = x1[i1][strata_scale_down(id2[i2])]
    return x2


def combine_ids(idlist):
    """ Combines several ID vectors into a single one, assigning a different number 
        to every combinations of the ID vectors. 
    """
    length = len(idlist[0])
    a = np.zeros((length, len(idlist)), dtype=int)
    for i, x in enumerate(idlist):
        assert len(x) == length, 'ID vectors must have the same lengths'
        a[:, i] = np.unique(x, return_inverse=True)[1]
    b = a.view(np.dtype((np.void, a.dtype.descr * a.shape[1])))
    return np.unique(b, return_inverse=True)[1]


def spread_av2ts(ts, av, x, eval_rng, fresh=False):
    if (ts._spread_av_tmp is not None) and not fresh:
        runid_in_av, tsrunid_scaled_down, si = ts._spread_av_tmp
    else:
        avrunid = av.run_id[eval_rng]
        assert(all(np.in1d(avrunid, ts.run_id)))
        assert len(avrunid) == len(np.unique(avrunid))
        runid_in_av = np.in1d(ts.run_id, avrunid)
        tsrunid_scaled_down = np.unique(ts.run_id[runid_in_av], return_inverse=True)[1]
        si = np.argsort(avrunid, kind='mergesort')
        ts._spread_av_tmp = runid_in_av, tsrunid_scaled_down, si
    return x[eval_rng][si][tsrunid_scaled_down]


def spread_from_av_old(ts, avrun_id, x, fresh=False):
    if (ts._spread_av_tmp is not None) and not fresh:
        runid_in_av, tsrunid_scaled_down, si = ts._spread_av_tmp
    else:
        tsrun_id = ts.run_id
        assert all(np.in1d(avrun_id, tsrun_id))
        assert len(avrun_id) == len(np.unique(avrun_id))
        runid_in_av = np.in1d(tsrun_id, avrun_id)
        tsrunid_scaled_down = np.unique(tsrun_id[runid_in_av], return_inverse=True)[1]
        si = np.argsort(avrun_id, kind='mergesort')
        ts._spread_av_tmp = runid_in_av, tsrunid_scaled_down, si

    if x.dtype != np.dtype('float16'):
        tsx = np.full_like(tsrun_id, intnan.NANVALS.get(x.dtype.char))
    else:
        tsx = np.full_like(tsrun_id, np.nan, dtype=np.float16)
    tsx[runid_in_av] = x[si][tsrunid_scaled_down]
    return tsx


def spread_from_av(target_id, source_id0, source0, eval_rng=None, cut_target=True):
    assert isinstance(source0[0], np.float64) or isinstance(source0[0], np.float32), 'Source has to be a float array.'
    if eval_rng is not None:
        source = source0[eval_rng]
        source_id = source_id0[eval_rng]
    else:
        source = source0
        source_id = source_id0

    assert len(source_id) == len(source)
    assert len(source_id) == len(np.unique(source_id)), 'Source IDs have to be pair-wise different.'

    max_id = np.max([np.nanmax(source_id), np.nanmax(target_id)])
    transfer = np.ones(max_id + 1) * 123456789  # some in <source> probably not occurring number
    transfer[source_id] = source

    target = transfer[target_id]
    runid_in_av = (target != 123456789)
    if cut_target:
        target = target[runid_in_av]
    else:
        target[~runid_in_av] = np.nan

    return target, runid_in_av


def print_params(step1coefs, step2coefs, astext=True):
    if astext:
        np.set_printoptions(precision=8, suppress=True)
        with open(Folder(__file__).up()['parameters.py'], 'w') as ofile:
            pp = pprint.PrettyPrinter(stream=ofile, indent=4)

            ofile.write('from numpy import array\n\n')
            ofile.write('step1coefs = ')
            pp.pprint(step1coefs)
            ofile.write('\n\n')

            ofile.write('step2coefs = ')
            pp.pprint(step2coefs)
            ofile.write('\n\n')
        np.set_printoptions(precision=4, suppress=True, threshold=1000)
    else:
        with open('parameters.dat', 'w') as ofile:
            pickle.dump((step1coefs, step2coefs), ofile)


def write_simdata(step1probs, oos, coefs, cluster_number=None):
    '''
    <step1probs> is expected to be a matrix N_slices x len(av). 
    <oos> is a boolean mask denoting the out of sample range. len(oos) shoud equal len(av)
    <coefs> is a coefficient matrix with the size N_slices x 3
    <cluster_number> is an integer array with the cluster numbers per race. Size: len(av)
    '''
    f = file(settings.paths.join('simdata.p'), 'wb')
    if cluster_number is None:
        s1p = step1probs[:, oos]
    else:
        cluster_number = cluster_number[oos]
        s1p = step1probs[:, :, oos]
    pickle.dump([s1p, oos, coefs, cluster_number], f)
    f.close()
