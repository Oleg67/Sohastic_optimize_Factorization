from __future__ import division
import numpy as np

from utils import get_logger, intnan, AttrDict, DAY
from utils.accumarray import uaccum
from utils.math import sleep, multiple_linear_regression, logarithmize
from utils.helpers import dispdots

from ..tools.helpers import strata_scale_down, spread_from_av
from .clmodel import run_cl_model
from .probabilities import ProbabilitiesCL
from .factor_expansion import expand_factor
from .parameters import factor_build_start, factor_build_end
from .prediction import TTM_SLICE, step1coefs as current_step1coefs, step2coefs as current_step2coefs, effective_coefficients

logger = get_logger(__package__)


########################################## The modeling process is split into the following ranges ###############################################
#    ----|---- standardization mask / factor training range ----|--- model training (is1) ---|--- validation (is2) ---|--- test (oos) ---|       #
#   build_start                                             build_end                                             oos_start         end of data  #
##################################################################################################################################################

class Model(object):
    def __init__(self, av, build_start=None, build_end=None, oos_start=None, transformation_degree=0, depth=3, lmbd=10):
        # get nice strata without gaps
        self.build_start = build_start or factor_build_start
        self.build_end = build_end or factor_build_end
        self.oos_start = oos_start or np.nanmax(av.start_time)
        assert np.nanmin(av.start_time) - DAY <= self.build_start < self.build_end < self.oos_start <= np.nanmax(av.start_time)
        self.av = av
        self.strata = strata_scale_down(self.av.event_id)
        self.standardization_mask = (self.build_start <= self.av.start_time) & (self.av.start_time < self.build_end)
        self.is1, self.is2, self.oos = self.model_mask(self.strata, self.av.start_time, self.av.result, self.av.course,
                                                        t0=self.build_end, t2=self.oos_start, depth=depth)
        self.depth = depth
        self.lmbd = lmbd
        self.transformation_degree = transformation_degree

    def _preprocess_factors(self, factornames, verbose=False, fill_missing_by_mean=False,
                           high_kurtosis_factors=None, price_factors=None, **kwargs):
        """Common preprocessing from an ArrayView instance to a factors matrix."""
        self.factornames = factornames[:]
        factors = np.full((len(factornames), len(self.av)), np.nan, dtype=np.float32)

        if factors.shape[0] == 1:
            fill_missing_by_mean = True

        if verbose:
            logger.info('Getting factors from av and rescaling...')

        for i, factorname in enumerate(factornames):
            try:
                col = self.av[factorname]
            except KeyError:
                logger.warn("Skipping AV column %s", factorname)
                continue

            if np.all(~np.isfinite(col[self.standardization_mask])):
                raise ValueError('Factor %s is all NaN in the in sample mask.' % factorname)

            high_kurtosis = high_kurtosis_factors is not None and factorname in high_kurtosis_factors
            price = price_factors is not None and factorname in price_factors
            factors[i, :] = self._standardize_factor(factorname, col, fill_missing=fill_missing_by_mean,
                                               high_kurtosis=high_kurtosis, price=price, verbose=verbose)
            if verbose:
                dispdots(i, 10)
            sleep()

        if not fill_missing_by_mean:
            self._fill_missing_values(factors, verbose=verbose)

        if np.any(np.isnan(factors)):
            raise RuntimeError('NaNs in the following factor(s) found: ', list(np.array(factornames)[np.any(np.isnan(factors), axis=1)]))

        if self.transformation_degree > 0:
            factors = self._transform_factors(factors, factornames, degree=self.transformation_degree, verbose=verbose)
        return factors

    # TODO: implement a proper outlier treatment
    def _standardize_factor(self, factorname, factor, fill_missing=False, high_kurtosis=False, price=False, verbose=False):
        '''Main pre-processing of a factor. Try to achieve a standard normal distribution of a factor.
        Get rid of skewness or kurtosis of a factor. Remove missing values. Shift mean to zero and normalize standard deviation to one.
        
        Logairthmize the factor if it is a price factor, i.e. when the factor is essentially an inverse probability 
        (hence the distribution is highly skewed to the right), or if it is listed among the high kurtosis factors.
        Replace all inf's by a nan.
        Normalize the mean to zero by subtracting the factor mean from every value.
        If <fill_missing> is true, then the nan's are replaced by a zero. Otherwise, the function <fill_missing_values> turns to more sophisticated methods to fill up the missing entries.
        The <correction> is essentially the standard deviation of the factor in the in sample region <standardization_mask>.
        Divide the factor by its standard deviation in order to normalize it to one.
        
        Return pre-processed factor.
        '''
        factor32 = factor.astype(np.float32)
        if price:
            np.log(factor32, factor32)

        if high_kurtosis:
            factor32 = logarithmize(factor32, count=1)

        missing = intnan.isnan(factor) | np.isinf(factor32)
        factor32[missing] = np.nan

        self._subtract_mean(factor32, self.strata)

        if fill_missing:
            factor32[missing] = 0

        self._normalize_stddev(factor32, ~missing & self.standardization_mask)

        # if verbose and kurtosis(factor32[~missing]) > 5.0:
        #    logger.warn('Factor %s has too high kurtosis.' % factorname)

        return factor32

    def _subtract_mean(self, x, strata):
        """Subtract mean of factor at every valid value"""
        meanx = uaccum(strata, x, func='nanmean')
        invalid = np.isnan(meanx) | np.isinf(meanx)
        meanx[invalid] = np.mean(meanx[~invalid])
        x -= meanx

    def _normalize_stddev(self, x, mask):
        """Normalize standard deviation to 1."""
        correction = np.nanstd(x[mask], ddof=1)
        if correction > 1e-10:
            x /= correction

    def _fill_missing_values(self, factors, lmbd=0.01, verbose=False):
        '''Try to reconstruct missing values by linear regression applied iteratively'''

        if verbose:
            logger.info('Filling in missing values...')

        coefs = self._missing_value_regression(factors, verbose)
        pattern = self._missing_patterns(np.isnan(factors))

        upats = np.unique(pattern)
        if verbose:
            logger.info('Number of missing patterns: %i' % len(upats))

        for k, pat in enumerate(upats):
            if verbose:
                dispdots(k, 1000)
            if pat == 0:  # all values present
                continue
            same_pat_idx = np.where(pattern == pat)[0]

            i = same_pat_idx[0]
            F = np.where(np.isnan(factors[:, i]))[0]  # the factors that are missing in this missing-pattern
            notF = np.where(~np.isnan(factors[:, i]))[0]  # the others
            y = np.dot(coefs[np.ix_(F, notF + 1)], factors[np.ix_(notF, same_pat_idx)])
            # y += np.dot(coefs[np.ix_(F, np.array([0]))], np.ones((1, len(same_pat_idx))))
            y += np.tile(coefs[F, 0], (len(same_pat_idx), 1)).transpose()
            S = np.eye(len(F)) - coefs[np.ix_(F, F + 1)]

            X = np.dot(np.linalg.inv(np.dot(S.transpose(), S) + lmbd * np.eye(len(F))), S.transpose())
            factors[np.ix_(F, same_pat_idx)] = np.dot(X, y)

        self._fix_outliers(factors)

    def _missing_patterns(self, missing, base=1.123):
        """Compute unique factor missing patterns, i.e. a number for each binary missing pattern."""
        binfac = np.tile(base ** np.arange(missing.shape[0]).reshape((-1, 1)), missing.shape[1])
        return np.sum(missing * binfac, axis=0)

    def _missing_value_regression(self, factors, verbose):
        """For every run in the factors matrix, if an entry is missing for a factor but all other entries are available for that run, use linear regression in order to predict the missing value."""
        if verbose:
            logger.info('Computing each factor as linear combination of all the others...')

        # fit every factor by all others
        missing = np.isnan(factors)
        nFactors = factors.shape[0]
        allgood = ~np.any(missing, axis=0) & self.standardization_mask
        if not np.any(allgood):
            raise ValueError('No single run has all factors available in the in sample mask. Missing value regression fails.')
        coefs = np.zeros((nFactors, nFactors + 1))
        for i in xrange(nFactors):
            if verbose:
                dispdots(i, 10)
            not_me = np.where(~np.in1d(np.arange(nFactors), i))[0]  # since i is me :)

            MLR = multiple_linear_regression(factors[not_me, :][:, allgood].transpose(), factors[i, allgood], lmbd=0.001)
            coefs[i, 0] = MLR.b[0]
            coefs[i, not_me + 1] = MLR.b[1:]

            only_me_missing = missing[i, :] & ~np.any(missing[not_me, :], axis=0)
            if np.any(only_me_missing):
                pred = MLR.b[0] + np.dot(MLR.b[1:], factors[not_me, :][:, only_me_missing])
                factors[i, only_me_missing] = pred
        return coefs

    def _fix_outliers(self, factors, sigma_scale=10):
        ''' Deletes all entries of factors that exceed <sigma_scale> standard deviations (those have been normalized to 1).'''
        factors[factors > sigma_scale] = 0.0
        factors[factors < -sigma_scale] = 0.0

    def model_mask(self, strata, start_time, result, course, t0=None, t1=None, t2=None, depth=3):
        eval_rng = valid_races(strata, result, course, depth=depth)
        if t1 is None:
            t1 = t2  # t0 + 0.9 * (t2 - t0)
        is1 = (start_time >= t0) & (start_time < t1) & eval_rng
        if t1 == t2:
            is2 = is1.copy()
        else:
            is2 = (start_time >= t1) & (start_time < t2) & eval_rng
        oos = (start_time >= t2) & eval_rng
        return is1, is2, oos

    def _transform_factors(self, factors, factornames, degree=1, verbose=False):
        '''Reduces kurtosis with logarithm and expand-fits each factor with polynomials of degree <degree>'''
        if verbose:
            logger.info('Transforming factors by applying CL-model on their Taylor expansions...')
        assert factors.shape[1] == len(self.av)
        in_sample = self.standardization_mask & valid_races(self.strata, self.av.result, self.av.course, depth=1)
        out_of_sample = self.is1 | self.is2 | self.oos
        new_factors = np.zeros_like(factors)
        for n in xrange(factors.shape[0]):
            sleep()
            if verbose:
                dispdots(n, 10)
            # exclude all races where the factor is zero for all runners
            allzero = uaccum(self.strata, abs(factors[n, :]) < 1e-10, func=all)
            x = factors[[n], :]

            # FIXME: Insert this assert statement again
            # TODO: Outlier detection missing
            # assert kurtosis(x[0, :]) < 10, 'Factor has too high kurtosis.'
            exp_factor = expand_factor(x[0, :], degree)
            if (np.sum(in_sample & ~allzero) > 0) and (np.sum(out_of_sample & ~allzero) > 0):
                stats = run_cl_model(exp_factor, self.av.result, self.strata, in_sample & ~allzero, out_of_sample & ~allzero, verbose=False)[0]
                probs = ProbabilitiesCL.compute(stats['coef'], exp_factor, self.strata, np.ones(len(self.strata), dtype=np.bool_))
                new_factors[n, :] = np.log(probs)
                new_factors[n, probs == 0] = -10
            else:
                new_factors[n, :] = x[0, :].copy()
            meanF = uaccum(self.strata, new_factors[n, :], func='nanmean')
            new_factors[n, :] -= meanF

        if np.any(np.isnan(new_factors)):
            raise RuntimeError('At least one entry became NaN after transformation.')
        return new_factors

    def _get_valid_mask(self, factors):
        '''remove factors that have zero variance or are constant for too long'''
        is_factors = factors[:, self.is1]
        goodfactors = (abs(np.std(is_factors, axis=1)) > 1e-10) & (np.mean(abs(is_factors) < 1e-10, axis=1) < 0.75)
        valid1 = np.ones(factors.shape[0] + 2, dtype=bool)
        valid1[0] = False
        valid1[2:] = goodfactors
        valid2 = np.ones(3, dtype=bool)
        valid2[:2] = valid1[:2]
        return valid1, valid2



    def fit_slices(self, tsav, factors, depth=None, lmbd=None, verbose=False, fit_afresh=True):
        depth = depth or self.depth
        lmbd = lmbd or self.lmbd
        nModels = factors.shape[0] + 2
        nSlices = len(TTM_SLICE)
        self.stats1 = AttrDict()
        self.stats1.coef = np.full((nSlices, nModels), np.nan)
        self.stats1.coefse = np.full((nSlices, nModels), np.nan)
        self.stats1.student_t = np.full((nSlices, nModels), np.nan)
        self.stats2 = AttrDict()
        self.stats2.coef = np.full((nSlices, 3), np.nan)
        self.stats2.coefse = np.full((nSlices, 3), np.nan)
        self.stats2.student_t = np.full((nSlices, 3), np.nan)
        self.stats2.ll  = np.zeros((nSlices, 3))
        self.stats2.pr2 = np.zeros((nSlices, 3))
         

        step1probs = np.zeros((nSlices, factors.shape[1])) # new
        step2probs = np.zeros((nSlices, len(tsav[0])))  # new

        if not fit_afresh:
            self.step1_coefs = current_step1coefs
            self.step2_coefs = current_step2coefs
            self.coefs = effective_coefficients(self.step1_coefs, self.step2_coefs)
            for sl in xrange(nSlices):
                strength = np.dot(self.step1_coefs[sl, 2:], factors)
                step1probs[sl, :] = np.exp(strength) / uaccum(self.strata, np.exp(strength))
            return self.coefs, step1probs, None, None

        assert np.all(np.in1d(tsav[0].run_id, self.av.run_id)), 'Some runs in time series are missing in the model av.'
        ints = np.where(np.in1d(self.av.run_id, tsav[0].run_id))[0]
        assert np.all(self.av.run_id[ints] == tsav[0].run_id), 'Ordering of run IDs are different in time series and model av.'
        strata = strata_scale_down(self.strata[ints])
        result = self.av.result[ints]

        valid1, valid2 = self._get_valid_mask(factors)
                

        if verbose:
            logger.info('Fitting...')
        for sl in xrange(nSlices - 1):
            if verbose:
                logger.info('Slice: %s' % sl)
            good = ~np.isnan(tsav[sl + 1].log_pmkt_back) & ~np.isnan(tsav[sl + 1].log_pmkt_lay)
            good = uaccum(strata, good, func='all')

            fb = tsav[sl + 1].log_pmkt_back.reshape((1, -1))
            fl = tsav[sl + 1].log_pmkt_lay.reshape((1, -1))
            
            step1factors = np.concatenate((fb, fl, factors[:, ints]), axis=0)
            step1factors[:, tsav[sl + 1].nonrunner == 1] = np.log(1e-3)

            rngis1 = self.is1[ints] & good
            rngis2 = self.is2[ints] & good
            rngoos = self.oos[ints] & good
            if np.any(rngis1) and np.any(rngis2) and np.any(rngoos):
                self.stats, probs = run_cl_model(step1factors[valid1, :], result, strata, rngis1, rngis2, rngoos,    verbose=verbose,depth=depth,          lmbd=lmbd)[:2]
                self.stats1.coef[sl, valid1] = self.stats.coef.squeeze()
                self.stats1.coefse[sl, valid1] = self.stats.coefse.squeeze()
                self.stats1.student_t[sl, valid1] = self.stats.t.squeeze()

                step2factors = np.concatenate((fb, fl, np.log(probs).reshape((1, -1))), axis=0)
                step2factors[:, tsav[sl + 1].nonrunner == 1] = np.log(1e-3)
                
                self.stats, prob2 = run_cl_model(step2factors[valid2, :], result, strata, rngis2, rngoos, rngoos, verbose=False)[:2]
                if verbose:
                    logger.info('ll{}'.format(self.stats.ll))
                    logger.info('pr2{}'.format(self.stats.pr2))


                self.stats2.ll[sl]  = self.stats.ll
                self.stats2.pr2[sl] = self.stats.pr2
                step2probs[sl] = prob2
                self.stats2.coef[sl, valid2] = self.stats.coef.squeeze()
                self.stats2.coefse[sl, valid2] = self.stats.coefse.squeeze()
                self.stats2.student_t[sl, valid2] = self.stats.t.squeeze()

        postprocess_coefs(self.stats1, valid1)
        postprocess_coefs(self.stats2, valid2)

        
        self.step1_coefs = self.stats1.coef.copy()
        self.step2_coefs = self.stats2.coef.copy()

        if verbose:
            logger.info('Compute step 1 probabilities...')
        self.coefs = effective_coefficients(self.stats1.coef, self.stats2.coef)

        for sl in xrange(nSlices):
            dispdots(sl, 1)
            strength = np.dot(self.step1_coefs[sl, 2:], factors)
            step1probs[sl, :] = np.exp(strength) / uaccum(self.strata, np.exp(strength))

        if verbose:
            logger.info('Done.')
        return self.coefs, step1probs, step2probs, self.stats2.ll


    def fit_step2_slices(self, tsav, modelprobs, depth=1, lmbd=0, verbose=False):
        nSlices = len(TTM_SLICE)
        self.stats = AttrDict()
        self.stats.coef = np.full((nSlices, 3), np.nan)
        self.stats.coefse = np.full((nSlices, 3), np.nan)
        self.stats.student_t = np.full((nSlices, 3), np.nan)

        assert np.all(np.in1d(tsav[0].run_id, self.av.run_id)), 'Some runs in time series are missing in the model av.'
        ints = np.where(np.in1d(self.av.run_id, tsav[0].run_id))[0]
        assert np.all(self.av.run_id[ints] == tsav[0].run_id), 'Ordering of run IDs are different in time series and model av.'
        strata = strata_scale_down(self.strata[ints])
        result = self.av.result[ints]

        if len(modelprobs.shape) == 1:
            modelprobs = np.tile(modelprobs.reshape((1, -1)), (nSlices, 1))

        step2probs = np.zeros((nSlices, len(tsav[0])))
        valid2 = np.array([False, True, True], dtype=bool)

        if verbose:
            logger.info('Fitting...')
        for sl in xrange(nSlices - 1):
            if verbose:
                logger.info('Slice: %s' % sl)
            good = ~np.isnan(tsav[sl + 1].log_pmkt_back) & ~np.isnan(tsav[sl + 1].log_pmkt_lay)
            good = uaccum(strata, good, func='all')

            rngis1 = self.is1[ints] & good
            rngis2 = self.is2[ints] & good
            rngoos = self.oos[ints] & good
            assert np.any(rngis1)
            assert np.any(rngis2)
            assert np.any(rngoos)

            fb = tsav[sl + 1].log_pmkt_back.reshape((1, -1))
            fl = tsav[sl + 1].log_pmkt_lay.reshape((1, -1))
            step2factors = np.concatenate((fb, fl, np.log(modelprobs[sl, ints]).reshape((1, -1))), axis=0)
            step2factors[:, tsav[sl + 1].nonrunner == 1] = np.log(1e-3)
            stats = run_cl_model(step2factors[valid2, :], result, strata, rngis2, rngoos, rngoos, verbose=False)[0]
            self.stats.coef[sl, valid2] = stats.coef.squeeze()
            self.stats.coefse[sl, valid2] = stats.coefse.squeeze()
            self.stats.student_t[sl, valid2] = stats.t.squeeze()
            strength = np.dot(stats.coef.flatten(), step2factors[valid2, :])
            step2probs[sl, :] = np.exp(strength) / uaccum(self.strata[ints], np.exp(strength))

        postprocess_coefs(self.stats, valid2)

        if verbose:
            logger.info('Done.')
        return self.stats.coef, step2probs


def postprocess_coefs(stats, valid):
    T = stats.coef.shape[0]
    # fill out other slices that could not be fitted
    stats.coef[:, ~valid] = 0
    stats.coefse[:, ~valid] = 0

    lastgood = np.where(~np.isnan(np.sum(stats.coef[:, valid], axis=1)))[0][-1]
    stats.coef[lastgood + 1:, :] = np.tile(stats.coef[[lastgood], :], [T - lastgood - 1, 1])
    stats.coefse[lastgood + 1:, :] = np.tile(stats.coefse[[lastgood], :], [T - lastgood - 1, 1])
    stats.student_t[lastgood + 1:, :] = np.tile(stats.student_t[[lastgood], :], [T - lastgood - 1, 1])
    #
    # # fill out other slices that could not be fitted
    # stats2.coef[:, ~valid2] = 0
    # stats2.coefse[:, ~valid2] = 0
    #
    # lastgood = np.where(~np.isnan(np.sum(stats2.coef[:, valid2], axis=1)))[0][-1]
    # stats2.coef[lastgood + 1:, :] = np.tile(stats2.coef[[lastgood], :], [T - lastgood - 1, 1])
    # stats2.coefse[lastgood + 1:, :] = np.tile(stats2.coefse[[lastgood], :], [T - lastgood - 1, 1])
    # stats2.student_t[lastgood + 1:, :] = np.tile(stats2.student_t[[lastgood], :], [T - lastgood - 1, 1])

    if np.any(np.isnan(stats.coef)):
        raise ArithmeticError('Some mixing coefficients are NaN.')


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
