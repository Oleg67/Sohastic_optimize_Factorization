
import numpy as np
from utils import DAY
from utils.accumarray import uaccum
from parameters import factor_build_end, factor_build_start
from ..tools.helpers import strata_scale_down


class ModelParameters(object):

    def __init__(self, av, build_start=None, build_end=None, oos_start=None, cut_mask=None, transformation_degree=0, depth=3, lmbd=10, verbose=False):
        self.build_start = build_start or factor_build_start
        self.build_end = build_end or factor_build_end
        self.oos_start = oos_start or np.nanmax(av.start_time)
        assert np.nanmin(av.start_time) - DAY <= self.build_start < self.build_end < self.oos_start <= np.nanmax(av.start_time)

        self.strata = strata_scale_down(av.event_id)
        self.run_id = av.run_id.copy()
        self.start_time = av.start_time.copy()
        self.result = av.result.copy()
        self.course = av.course.copy()

        self.valid = self.valid_mask(av.result, av.course, depth=depth)
        if cut_mask is not None:
            self.valid &= cut_mask
        self.build_mask = (self.build_start <= av.start_time) & (av.start_time < self.build_end) & self.valid
        self.is1, self.is2, self.oos = self.model_mask(self.valid, t0=self.build_end, t2=self.oos_start)
        self.model_mask = self.is1 | self.is2 | self.oos

        self.depth = depth
        self.lmbd = lmbd
        self.transformation_degree = transformation_degree
        self.verbose = verbose

    def model_mask(self, valid, t0=None, t1=None, t2=None):
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
