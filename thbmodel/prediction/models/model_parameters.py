import numpy as np

from ...utils import DAY
from ...utils.accumarray import uaccum
from .parameters import factor_build_end, factor_build_start
from ..tools.helpers import strata_scale_down


class ModelParameters(object):
    defaults = dict(transformation_degree=0, depth=3, lmbd=10, cut_mask=None,
                        build_start=factor_build_start, build_end=factor_build_end)

    def __init__(self, av, oos_start=None, verbose=False, checked=True, **kwargs):
        for k, v in kwargs.items():
            if k not in self.defaults:
                raise TypeError("%s got an unexpected keyword argument '%s'" % (type(self).__name__, k))
        for k, v in self.defaults.items():
            setattr(self, k, kwargs.get(k, v))
        self._av_start = np.nanmin(av.start_time)
        self._av_end = np.nanmax(av.start_time)
        self.oos_start = oos_start or self._av_end
        self.verbose = verbose

        self.strata = strata_scale_down(av.event_id)
        self.run_id = av.run_id.copy()
        self.start_time = av.start_time.copy()
        self.result = av.result.copy()
        self.course = av.course.copy()

        self.valid = self.valid_mask(av.result, av.course, depth=self.depth)
        if self.cut_mask is not None:
            self.valid &= self.cut_mask
        self.build_mask = (self.build_start <= av.start_time) & (av.start_time < self.build_end) & self.valid
        self.is1, self.is2, self.oos = self.get_model_mask(self.valid, t0=self.build_end, t1=self.build_end + DAY * 14, t2=self.oos_start)
        self.model_mask_ = self.is1 | self.is2 | self.oos
        if checked:
            self.check()

    def get_model_mask(self, valid, t0=None, t1=None, t2=None):
        if t1 is None:
            t1 = t2  # t0 + 0.9 * (t2 - t0)
        is1 = (self.start_time >= t0) & (self.start_time < t1) & valid
        is2 = is1.copy() if t1 == t2 else (self.start_time >= t1) & (self.start_time < t2) & valid
        oos = (self.start_time >= t2) & valid
        return is1, is2, oos

    def valid_mask(self, result, course, depth=1):
        # select only races where the number of winners, second placed, third placed, ..., depth-placed is exactly 1.
        good_depth = np.ones_like(self.strata, dtype=bool)
        for r in xrange(1, depth + 1):
            good_depth &= uaccum(self.strata, result == r) == 1

        # big selector, which data to use in model
        eval_rng = (course <= 88) & good_depth
        return eval_rng

    def check(self):
        assert self._av_start - DAY <= self.build_start < self.build_end < self.oos_start <= self._av_end

    def is_default(self):
        if self._av_end != self.oos_start:
            return False
        for k, v in self.defaults.items():
            if getattr(self, k) != v:
                return False
        return True


def cut_model(model, mask):
    '''Cuts the model to given mask'''
    newmodel = ModelParameters()
    attributes = model.__dict__.keys()
    for attr in attributes:
        value = model.__dict__[attr]
        if isinstance(value, np.ndarray):
            newmodel.__dict__[attr] = value[mask]
        else:
            newmodel.__dict__[attr] = value
    return newmodel
