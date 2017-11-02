import numpy as np
import pdb

from ...utils import get_logger
from ...utils.accumarray import uaccum

from ..tools.helpers import strata_scale_down
from .clmodel import run_cl_model
from .prediction import step1coefs as current_step1coefs, step2coefs as current_step2coefs, effective_coefficients, factors_to_probs

logger = get_logger(__package__)



########################################## The modeling process is split into the following ranges ###############################################
#    ----|---- standardization mask / factor training range ----|--- model training (is1) ---|--- validation (is2) ---|--- test (oos) ---|       #
#   build_start                                             build_end                                             oos_start         end of data  #
##################################################################################################################################################



class CLStats(object):

    def __init__(self, nSlices, nModels):
        self.coef = np.full((nSlices, nModels), np.nan)
        self.coefse = np.full((nSlices, nModels), np.nan)
        self.student_t = np.full((nSlices, nModels), np.nan)
        self.ll = np.full((nSlices, 3), np.nan)
        self.pr2 = np.full((nSlices, 3), np.nan)

    def read(self, sl, valid, stats):
        self.coef[sl, valid] = stats.coef.squeeze()
        self.coefse[sl, valid] = stats.coefse.squeeze()
        self.student_t[sl, valid] = stats.t.squeeze()
        self.ll[sl] = stats.ll
        self.pr2[sl] = stats.pr2

    def postprocess(self, valid):
        T = self.coef.shape[0]
        # fill out other slices that could not be fitted
        self.coef[:, ~valid] = 0
        self.coefse[:, ~valid] = 0

        lastgood = np.where(~np.isnan(np.sum(self.coef[:, valid], axis=1)))[0][-1]
        self.coef[lastgood + 1:, :] = np.tile(self.coef[[lastgood], :], [T - lastgood - 1, 1])
        self.coefse[lastgood + 1:, :] = np.tile(self.coefse[[lastgood], :], [T - lastgood - 1, 1])
        self.student_t[lastgood + 1:, :] = np.tile(self.student_t[[lastgood], :], [T - lastgood - 1, 1])

        if np.any(np.isnan(self.coef)):
            raise ArithmeticError('Some mixing coefficients are NaN.')





class TSModel(object):

    def __init__(self, factors, tsav, params):
        self.factors = factors
        self.tsav = tsav
        self.params = params

        nModels, nRuns = factors.shape[0] + 2, factors.shape[1]
        nSlices = len(tsav)
        assert len(params.strata) == nRuns

        self.step1probs = np.zeros((nSlices, nRuns))
        self.step2probs = np.zeros((nSlices, len(tsav[0])))
        
        self.stats1 = CLStats(nSlices, nModels)
        self.stats2 = CLStats(nSlices, 3)
    
    def cut_to_ts(self, run_id_small, run_id_big):
        '''Cut model parameters to tsav size, where the time series is present'''
        assert np.all(np.in1d(run_id_small, run_id_big)), 'Some runs in time series are missing in the model av.'
        ts_idx = np.where(np.in1d(run_id_big, run_id_small))[0]
        assert np.all(run_id_big[ts_idx] == run_id_small), 'Ordering of run IDs are different in time series and model av.'
        strata = strata_scale_down(self.params.strata[ts_idx])
        result = self.params.result[ts_idx]
        return ts_idx, strata, result

    def get_valid_mask(self, factors, is1):
        '''remove factors that have zero variance or are constant for too long'''
        is_factors = factors[:, is1]
        goodfactors = (abs(np.std(is_factors, axis=1)) > 1e-10) & (np.mean(abs(is_factors) < 1e-10, axis=1) < 0.75)
        valid1 = np.ones(factors.shape[0] + 2, dtype=bool)
        valid1[0] = False
        valid1[2:] = goodfactors
        valid2 = np.ones(3, dtype=bool)
        valid2[:2] = valid1[:2]
        return valid1, valid2

    def concat(self, strata, nonrunner, factorlist):
        result = []
        n = None
        for arg in factorlist:
            assert 1 <= len(arg.shape) <= 2
            if n is None:
                n = arg.shape[-1]
            assert arg.shape[-1] == n
            if len(arg.shape) == 1:
                result += [arg.reshape((1, -1))]
            else:
                result += [arg]
        result = np.concatenate(result, axis=0)
        result[:, nonrunner] = np.log(1e-5)
        valid = np.all(~np.isnan(result), axis=0)
        valid = uaccum(strata, valid, func='all')
        return result, valid

    def finalize(self, coef1=None, coef2=None):
        if coef1 is not None and coef2 is not None:
            self.stats1.coef, self.stats2.coef = coef1, coef2
        self.eff_coefs = effective_coefficients(self.stats1.coef, self.stats2.coef)
        self.step1probs = factors_to_probs(self.stats1.coef[:, 2:], self.factors, self.params.strata)

    def concat_and_fit(self, strata, result, nonrunner, factorlist, ts_idx, valid, verbose=False, depth=1, lmbd=0, step=1):
        factors, valid_runs = self.concat(strata, nonrunner, factorlist)
        #self.params.valid_runs = valid_runs

        ranges = ['is1', 'is2', 'oos'] if step == 1 else ['is2', 'oos', 'oos']
        mask = {}
        for rng in ranges:
            mask[rng] = self.params.__dict__[rng][ts_idx] & valid_runs

        assert all([np.any(mask[rng]) for rng in mask])
        stats, probs = run_cl_model(factors[valid, :], result, strata, mask[ranges[0]], mask[ranges[1]], mask[ranges[2]], verbose=False, depth=depth, lmbd=lmbd)[:2]
        # pdb.set_trace()
        return stats, probs

    def fit_slices(self, fit_afresh=True):
        if not fit_afresh:
            self.finalize(current_step1coefs, current_step2coefs)
            return
        nSlices = len(self.tsav)
        ts_idx, strata, result = self.cut_to_ts(self.tsav[0].run_id, self.params.run_id)
        self.params.ts_idx = ts_idx
        valid1, valid2 = self.get_valid_mask(self.factors, self.params.is1)

        for sl in xrange(nSlices - 1):
            if self.params.verbose:
                logger.info('Fitting slice %s' % sl)
            nonrunner = self.tsav[sl + 1].nonrunner == 1 
            fback = self.tsav[sl + 1].log_pmkt_back 
            flay = self.tsav[sl + 1].log_pmkt_lay

            stats, probs = self.concat_and_fit(strata, result, nonrunner, [fback, flay, self.factors[:, ts_idx]], ts_idx, valid1, verbose=self.params.verbose, depth=self.params.depth, lmbd=self.params.lmbd, step=1)
            self.stats1.read(sl, valid1, stats)

            stats, self.step2probs[sl] = self.concat_and_fit(strata, result, nonrunner, [fback, flay, np.log(probs)], ts_idx, valid2, verbose=False, step=2)
            self.stats2.read(sl, valid2, stats)

        self.stats1.postprocess(valid1)
        self.stats2.postprocess(valid2)
        self.finalize()

    def fit_factors_only(self):
        is_factors = self.factors[:, self.params.is1]
        valid = (abs(np.std(is_factors, axis=1)) > 1e-10) & (np.mean(abs(is_factors) < 1e-10, axis=1) < 0.75)
        valid_runs = np.all(~np.isnan(self.factors), axis=0)
        valid_runs = uaccum(self.params.strata, valid_runs, func='all')
        ranges = ['is1', 'is2', 'oos']
        mask = {}
        for rng in ranges:
            mask[rng] = self.params.__dict__[rng] & valid_runs
        assert all([np.any(mask[rng]) for rng in mask])
        stats, probs = run_cl_model(self.factors[valid, :], self.params.result, self.params.strata, mask[ranges[0]],
                                    mask[ranges[1]], mask[ranges[2]], verbose=False, depth=self.params.depth,
                                    lmbd=self.params.lmbd)[:2]
        win_flag = self.params.result == 1
        trainll, testll = np.mean(np.log(probs[win_flag & mask[ranges[0]]])) * 1000, np.mean(np.log(probs[win_flag & mask[ranges[2]]])) * 1000
        return trainll, testll

