""" Build target definition for factors """

from __future__ import division
import gc
import numpy as np

from utils.math import sleep
from utils import intnan, get_logger

from ..models.parameters import factor_build_end

logger = get_logger(__package__)


class FactorTarget(object):
    provided = []
    required = []
    temporary = []
    missing_ratio_max = dict()
    missing_ratio_max_default = 0.05
    err_settings = dict(invalid='ignore')

    def __init__(self, **kwargs):
        pass

    def build(self, av, sel=None, lazy=False, verbose=False, factors=None, build_end=None, **kwargs):
        if sel is None:
            sel = slice(None)
        if build_end is None:
            build_end = factor_build_end

        self.check_requirements(av, sel)
        if lazy and not self.missing_cols(av, sel):
            logger.info('%s: Provided data fully present, skipping', type(self).__name__)
            return
        elif verbose:
            logger.info("%s: Building started", type(self).__name__)
        self.prepare_output_columns(av, lazy=lazy, sel=sel, factors=factors)
        gc.collect()
        sleep()
        with np.errstate(**self.err_settings):
            self.run(av, sel, verbose=verbose, factors=factors, build_end=build_end, **kwargs)
        sleep()
        self.clean(av)
        sleep()

    def check_requirements(self, av, sel=None):
        if sel is None:
            sel = np.s_[:]
        for item in self.required:
            if item not in av:
                raise ValueError("Arrayview has no column %s" % item)
            elif intnan.allnan(getattr(av, item)):
                raise ValueError("Required column %s is empty" % item)

    def get_missing_ratio_max(self, col):
        return self.missing_ratio_max.get(col, self.missing_ratio_max_default)

    def missing_cols(self, av, sel):
        for col in self.provided:
            if col not in av or np.mean(intnan.isnan(getattr(av, col)[sel])) > self.get_missing_ratio_max(col):
                return True
        return False

    @classmethod
    def prepare_output_columns(cls, av, lazy=True, sel=None, factors=None):
        for item in cls.provided:
            if factors is None or item in factors:
                if item not in av:
                    av.create_col(item)
                elif not lazy:
                    if sel is None:
                        sel = np.s_[:]
                    av[item][sel].fill(intnan.NANVALS.get(av[item].dtype.char, 0))

    def clean(self, av):
        for item in self.temporary:
            del av[item]
        gc.collect()

    def run(self, av, sel, verbose=False, **kwargs):
        raise NotImplementedError()


class DynamicFactorTarget(FactorTarget):
    """ Some factors depend on a dynamic list of input factors """

    def __init__(self, requirements=None):
        if not isinstance(requirements, (tuple, list)):
            raise ValueError("No iterable factor names provided")
        elif not len(requirements):
            raise ValueError("No factors provided")
        self._required = self.filter_required(requirements)

    def filter_required(self, factornames):
        """ Return a list of required factor names, calculated from the provided ones """
        for factorname in self.provided:
            assert factorname not in factornames, "Cannot require provided factor name"
        # No prices, no fully aggregated factors
        return [f for f in factornames if not f.startswith('-') and f not in ('step1probs', 'thb_est')]

    @property
    def required(self):
        return self._required
