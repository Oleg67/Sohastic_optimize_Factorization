'''
Module for calculations to adjust the influence of different factors to the runner win probability
'''
from __future__ import division
import logging
import numpy as np

from utils import AttrDict, get_logger, nbtools as nbt
from utils.accumarray import uaccum
from utils.math import sleep

from . import jlikelihoods
from ..tools.helpers import strata_scale_down, check_results, strata_mask_compatible

logger = get_logger(__package__)



def run_cl_model(factors, result, strata, is1=None, is2=None, oos=None, depth=1, lmbd=0, verbose=False, delay=0):
    """Fits a conditional logit model given factors, race results and strata that define the races.
    
    Inputs:
    -------
    If there are n factors and m runs in total, then <factors> is a n x m matrix of valid floats (no NaN's).
    <strata> are group indices that denote which runs belong to the same race. For example, if
    <strata> = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2]) then the first 3 runs belong to race 0, the subsequent 4 runs to race 1,
    the subsequent 2 runs to race 2 etc. <strata> is an integer array of length m.
    <result> is also an integer array of length m that denotes the finishing position of a run. For example, consistent with the strata
    example above, we could have 
    result = np.array([2, 3, 1, 3, 2, 4, 1, 2, 1])
    <is1>, <is2> and <oos> are boolean masks that denote the runs that belong to the first or second in sample or out of sample range.
    <depth> denotes the depth of the explosion process.
    <lmbd> is the Tikhonov regularization parameter.
    
    Outputs:
    --------
    <stats> is a dictionary with the entries <coef> for the fitting CL-model coefficients, their standard errors <coefse> and Student-t-scores <t>.
    <probs> is an array of length m denoting the winning probability for each run.
    """
    is1 = is1 if is1 is not None else np.ones(len(strata), dtype=bool)
    is2 = is2 if is2 is not None else is1
    oos = oos if oos is not None else is1

    # check whether all strata in the is1 range have exactly one winner
    for mask in (is1, is2, oos):
        strata_mask_compatible(strata, mask)
    check_results(strata, result, is1 | is2, max_result=depth)

    # use explosion process to use more data for fitting
    exploded = _explosion_process(factors[:, is1], strata[is1], result[is1], depth=depth)

    stats = AttrDict()
    model = jlikelihoods.jLikelihoodCL(*exploded, lmbd=lmbd)
    stats.coef = newton(model, verbose=verbose)
    sleep(delay / 10)

    hessian = model.compute(stats.coef, d2ll=True)
    stats.update(_student_t(hessian, stats.coef))

    probs = model.probs_class().compute(stats.coef, factors, strata, is1 | is2 | oos)

    stats.ll = [_normalized_likelihood(strata, probs, result == 1, mask, check=False) for mask in (is1, is2, oos)]
    stats.pr2 = [_pseudo_r2(strata, probs, result == 1, mask, check=False) for mask in (is1, is2, oos)]
    return stats, probs


def _explosion_process(factors, strata, result, depth=1):
    strata = strata_scale_down(strata)
    exploded_winners = np.array([], dtype=np.bool_)

    exploded_factors = factors.copy()
    reduced_factors = factors.copy()

    exploded_strata = strata.copy()
    reduced_strata = strata.copy()

    position = result.copy()
    for d in xrange(1, depth + 1):
        sleep()
        winners = (position == d)
        exploded_winners = np.append(exploded_winners, winners)

        if d < depth:
            reduced_factors = reduced_factors[:, ~winners]
            exploded_factors = np.append(exploded_factors, reduced_factors, axis=1)

            reduced_strata = strata_scale_down(reduced_strata[~winners])
            exploded_strata = strata_scale_down(np.append(exploded_strata, reduced_strata + max(exploded_strata) + 1))
            position = position[~winners]

    assert len(np.unique(exploded_strata)) == np.sum(exploded_winners)
    return exploded_factors, exploded_strata, exploded_winners


def _normalized_likelihood(strata, prob, winflag, mask, check=True):
    '''Normalized log likelihood'''
    if check:
        assert np.all(prob[mask] > 0)
        strata_mask_compatible(strata, mask)
        check_results(strata, winflag, mask)
    return np.mean(np.log(prob[winflag & mask])) * 1000


def _pseudo_r2(strata, prob, winflag, mask, check=True):
    '''Pseudo R^2 measure'''
    LLmodel = _normalized_likelihood(strata, prob, winflag, mask, check=check)
    uniformprob = 1 / uaccum(strata, np.ones_like(strata))
    LLuniform = np.mean(np.log(uniformprob[ winflag & mask])) * 1000
    PR2 = 1 - LLmodel / LLuniform
    return PR2


def _student_t(hessian, coef):
    """Compute the standard error on CL model coefficients and the Student-t-score."""
    information_matrix = hessian if nbt.allnan(hessian) else np.linalg.inv(-hessian)
    coefse = np.sqrt(np.diag(information_matrix))
    t = np.abs(coef.squeeze() / coefse)
    return dict(coefse=coefse, t=t)


def newton(model, step=0.9, tolerance=1e-5, max_iterations=20, max_tries=5, verbose=False, check_condition=True):
    """
    Performs Newton-Ralphson method for finding an extremum of a function
    Input:
    func = function handle that computes the function and its first and second derivative (Hessian)
    x0 = starting point for iteration, a column vector
    step = step size
    tolerance = halting criterion as threshold for the norm of the first derivative
    max_iterations = maximal iterations to try until the first retry
    max_tries = starting point get randomly reinitialized if no convergence has been 
                achieved and a new trial starts until max_tries is reached
    verbose = prints something during operation if set to True
    """
    x0 = np.zeros((model.factors.shape[0], 1))
    xold = x0.reshape((-1, 1))

    if check_condition:
        d2ll = model.compute(xold, d2ll=True)
        if nbt.allnan(d2ll):
            jc = np.nan  # Prevent segfault in cond
        else:
            jc = 1 / np.linalg.cond(d2ll, 1)
        if jc < np.finfo(np.float).eps:
            raise ArithmeticError('Hessian in bad condition, probably due to collinearity')

    for tries_count in xrange(max_tries):
        if tries_count > 0:
            if verbose:
                logging.info("Newton method did not converge. Trying again with different x0...")
            xold = np.random.randn(*xold.shape)
        for iter_count in xrange(max_iterations):
            sleep()
            dll, d2ll = model.compute(xold, dll=True, d2ll=True)
            if nbt.allnan(d2ll):
                xnew = np.full_like(xold, np.nan)
            else:
                xnew = xold - step * np.linalg.solve(d2ll.transpose(), dll.reshape((-1, 1)))
            if verbose:
                logging.info('Derivative norm after %d iterations: %g' % (iter_count, np.linalg.norm(dll)))
            if nbt.anynan(xnew):
                # Does not converge
                break
            elif np.linalg.norm(dll) < tolerance or \
                    np.linalg.norm(xold - xnew, np.inf) / np.linalg.norm(xnew, np.inf) < tolerance:
                if verbose:
                    print
                if np.any(np.isnan(xnew)):
                    raise ArithmeticError('Newton-Ralphson method returns NaN coefficients.')
                return xnew
            else:
                xold = xnew
    else:
        raise ArithmeticError("Newton approximation did not converge after %d retries" % max_tries)




