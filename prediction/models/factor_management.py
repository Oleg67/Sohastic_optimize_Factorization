
import numpy as np

from utils import intnan, get_logger
from utils.arrayview import ArrayView
from utils.accumarray import uaccum
from utils.math import multiple_linear_regression, logarithmize, sleep
from utils.helpers import dispdots

from .clmodel import run_cl_model
from .factor_expansion import expand_factor
from .probabilities import ProbabilitiesCL


logger = get_logger(__package__)



HIGH_KURTOSIS_FACTORS_hashed = set(['z64f5be67e', 'z90adc182a', 'z7081bf371', 'z34b808e99', 'z757be272e', 'z5a85cd6a9', 'zf991b634a', 'z62651f605',
                                    'zd002b7067', 'z2ef7fedca', 'z6f11029f7', 'z412893062', 'z919b9585a', 'z89b0eda37', 'z31780b3f4', 'z6631693d3',
                                    'z0b27f29ad', 'zd7cd94e4c', 'zf5b2aef2a'])
PRICE_FACTORS_hashed = set(['zb392bb74a', 'z6809c316d', 'zd678f0538', 'z027f9f0f5', 'z88e79930c', 'z4a72dc02f', 'z1a3573928', 'z7b15df227'])



class Factor(object):
    '''Organizes what a factor can do'''

    def __init__(self, col, name, anonymized=True):
        self.name = name
        self.col = col  # the array storing the factor
        self.set_flags(name, anonymized)

    def set_flags(self, name, anonymized):
        hk, pf = HIGH_KURTOSIS_FACTORS_hashed, PRICE_FACTORS_hashed
        if not anonymized:
            from ..factors.factor_attributes import HIGH_KURTOSIS_FACTORS as hk, PRICE_FACTORS as pf
        self.high_kurtosis = name in hk
        self.isa_price = name in pf

    def standardize(self, std_mask, strata, fill_missing=False):
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
        self.col = self.col.astype(np.float32)
        if self.isa_price:
            np.log(self.col, self.col)

        if self.high_kurtosis:
            self.col = logarithmize(self.col, count=1)

        missing = intnan.isnan(self.col) | np.isinf(self.col)
        self.col[missing] = np.nan

        self._subtract_mean(strata)

        if fill_missing:
            self.col[missing] = 0

        self._normalize_stddev(~missing & std_mask)

    def _subtract_mean(self, strata):
        """Subtract mean of factor at every valid value"""
        meanx = uaccum(strata, self.col, func='nanmean')
        invalid = np.isnan(meanx) | np.isinf(meanx)
        meanx[invalid] = np.mean(meanx[~invalid])
        self.col -= meanx

    def _normalize_stddev(self, mask):
        """Normalize standard deviation to 1."""
        correction = np.nanstd(self.col[mask], ddof=1)
        if correction > 1e-10:
            self.col /= correction

    def transform(self, model):
        allzero = uaccum(model.strata, abs(self.col) < 1e-10, func=all)
        is1, oos = model.build_mask & ~allzero, model.model_mask & ~allzero

        exp_factor = expand_factor(self.col, model.transformation_degree)
        if (np.sum(is1) > 0) and (np.sum(oos) > 0):
            stats = run_cl_model(exp_factor, model.result, model.strata, is1, oos, verbose=False)[0]
            probs = ProbabilitiesCL.compute(stats['coef'], exp_factor, model.strata, np.ones(len(model.strata), dtype=np.bool_))
            self.col = np.log(probs)
            self.col[probs == 0] = -10
        self._subtract_mean(model.strata)
        assert np.all(np.isfinite(self.col)), 'At least one entry became NaN after transformation in factor %s.' % self.name



class FactorList(object):
    '''Organizes the set of factors in a model'''

    def __init__(self, av, factornames, anonymized=True):
        self.factors = self.from_av(av, factornames, anonymized)

    def from_av(self, av, factornames, anonymized):
        factors = []        
        for i, name in enumerate(factornames):
            if isinstance(av, ArrayView):
                try:
                    col = av[name]
                except KeyError:
                    logger.warn("Skipping AV column %s", name)
                    continue
            elif isinstance(av, np.ndarray):
                col = av[i, :]
            else:
                raise KeyError('av has to be an ArrayView instance or a matrix.')
            factors += [Factor(col, name, anonymized=anonymized)]
        self.len = len(av) if isinstance(av, ArrayView) else av.shape[1]
        return factors
    
    def asmatrix(self):
        '''Convert the FactorList object to an actual matrix of factors'''
        factors = np.zeros((len(self.factors), self.len))
        for i, factor in enumerate(self.factors):
            factors[i, :] = factor.col
        return factors
    
    def asobject(self, factors):
        '''Convert a factor matrix to the present FactorList object'''
        assert len(self.factors) == factors.shape[0]
        for i in xrange(len(self.factors)):
            self.factors[i].col = factors[i, :]

    def check_all_finite(self):
        for factor in self.factors:
            assert np.all(np.isfinite(factor.col)), 'NaNs or Infs in the following factor found: ' % factor.name

    def preprocess(self, model, fill_missing_by_mean=False):
        """Common preprocessing from an ArrayView instance to a factors matrix."""
        if len(self.factors) == 1:
            fill_missing_by_mean = True

        if model.verbose:
            logger.info('Getting factors from av and rescaling...')

        for i, factor in enumerate(self.factors):
            if model.verbose:
                dispdots(i, 10)
            assert len(factor.col) == len(model.strata)
            assert np.any(np.isfinite(factor.col)), 'Factor %s is all NaN.' % factor.name

            factor.standardize(model.build_mask, model.strata, fill_missing=fill_missing_by_mean)
            sleep()

        if not fill_missing_by_mean:
            self.fill_missing_values(model)

        self.check_all_finite()

        if model.transformation_degree > 0:
            self.transform_factors(model)


    def fill_missing_values(self, model, lmbd=0.01):
        '''Try to reconstruct missing values by linear regression applied iteratively'''

        if model.verbose:
            logger.info('Filling in missing values...')

        factors, coefs = self.missing_value_regression(model)
        pattern = self.missing_patterns(np.isnan(factors))

        upats = np.unique(pattern)
        if model.verbose:
            logger.info('Number of missing patterns: %i' % len(upats))

        for k, pat in enumerate(upats):
            if model.verbose:
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

        self.fix_outliers(factors)
        self.asobject(factors)

    def missing_patterns(self, missing, base=1.123):
        """Compute unique factor missing patterns, i.e. a number for each binary missing pattern."""
        binfac = np.tile(base ** np.arange(missing.shape[0]).reshape((-1, 1)), missing.shape[1])
        return np.sum(missing * binfac, axis=0)

    def missing_value_regression(self, model):
        """For every run in the factors matrix, if an entry is missing for a factor but all other entries are available for that run, use linear regression in order to predict the missing value."""
        if model.verbose:
            logger.info('Computing each factor as linear combination of all the others...')

        factors = self.asmatrix()
        # fit every factor by all others
        missing = np.isnan(factors)
        nFactors = factors.shape[0]
        allgood = ~np.any(missing, axis=0) & model.build_mask
        assert np.any(allgood), 'No single run has all factors available in the in sample mask. Missing value regression fails.'

        coefs = np.zeros((nFactors, nFactors + 1))
        for i in xrange(nFactors):
            if model.verbose:
                dispdots(i, 10)
            not_me = np.where(~np.in1d(np.arange(nFactors), i))[0]  # since i is me :)

            MLR = multiple_linear_regression(factors[not_me, :][:, allgood].transpose(), factors[i, allgood], lmbd=0.001)
            coefs[i, 0] = MLR.b[0]
            coefs[i, not_me + 1] = MLR.b[1:]

            only_me_missing = missing[i, :] & ~np.any(missing[not_me, :], axis=0)
            if np.any(only_me_missing):
                pred = MLR.b[0] + np.dot(MLR.b[1:], factors[not_me, :][:, only_me_missing])
                factors[i, only_me_missing] = pred
        return factors, coefs

    def fix_outliers(self, factors, sigma_scale=10):
        ''' Deletes all entries of factors that exceed <sigma_scale> standard deviations (those have been normalized to 1).'''
        factors[factors > sigma_scale] = 0.0
        factors[factors < -sigma_scale] = 0.0

    def transform_factors(self, model):
        '''Reduces kurtosis with logarithm and expand-fits each factor with polynomials of degree <degree>'''
        if model.verbose:
            logger.info('Transforming factors by applying CL-model on their Taylor expansions...')

        for n, factor in enumerate(self.factors):
            sleep()
            if model.verbose:
                dispdots(n, 10)
            factor.transform(model)

