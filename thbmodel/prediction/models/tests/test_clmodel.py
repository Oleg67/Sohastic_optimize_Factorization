import numpy as np
import pytest

from utils.arrayview import ArrayView
from utils.containers import AttrDict

from .. import clmodel as cl
from ...tools.helpers import strata_scale_down


@pytest.fixture()
def mock():
    av = ArrayView.dummy(events=3, random_factors=3)
    strata = strata_scale_down(av.event_id[1:])
    return AttrDict(locals())


def test_pseudo_r2_and_likelihood(mock):
    mask = np.ones(len(mock.av)-1, dtype=bool)
    strata, prob, result = mock.strata, mock.av.true_prob[1:], mock.av.result[1:]
    ll  = cl._normalized_likelihood(strata, prob, result == 1, mask)
    pr2 = cl._pseudo_r2(strata, prob, result == 1, mask)

    np.testing.assert_almost_equal(np.mean(np.log(prob[result == 1])) * 1000, ll)
    ul = np.log(1/10.0) * 1000
    np.testing.assert_almost_equal(1 - ll / ul, pr2)
    

def test_run_cl_model():
    factors = np.array([[-1.057 , -1.2241, -1.2881,  0.3644, -1.0761,  0.0945, -0.6888,  1.5321,  0.8713, -0.197 ,  1.7776, -0.3369],
                       [-2.0017,  0.2381, -1.0427,  0.31  , -0.5714,  1.6635, -0.5881, -0.6335, -1.5251, -0.9937, -0.1551, -0.2515]], dtype=np.float32)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    is1 = is2 = oos = np.ones(len(strata), dtype=bool)
    result = np.array([1, 2, 3, 4, 3, 4, 2, 1, 3, 1, 4, 2])
    probs = cl.run_cl_model(factors, result, strata, is1, is2, oos)[1]
    sollprobs = np.array([ 0.8217,  0.0148,  0.137 ,  0.0264,  0.1718,  0.0057,  0.2103, 0.6123,  0.6944,  0.1693,  0.0932,  0.0431])
    np.testing.assert_array_almost_equal(probs, sollprobs, decimal=4)
    
    
def test_check_results_in_run_cl_model():
    factors = np.array([[-1.057 , -1.2241, -1.2881,  0.3644, -1.0761,  0.0945, -0.6888,  1.5321,  0.8713, -0.197 ,  1.7776, -0.3369],
                       [-2.0017,  0.2381, -1.0427,  0.31  , -0.5714,  1.6635, -0.5881, -0.6335, -1.5251, -0.9937, -0.1551, -0.2515]], dtype=np.float32)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    is1 = np.zeros(len(strata), dtype=bool)
    is1[:8] = True
    is2 = is1
    oos = np.zeros(len(strata), dtype=bool)
    oos[8:] = True
    result = np.array([1, 2, 3, 4, 3, 4, 2, 1, 0, 0, 0, 0])
    cl.run_cl_model(factors, result, strata, is1, is2, oos)
    result = np.array([0, 0, 0, 0, 3, 4, 2, 1, 3, 1, 4, 2])
    with pytest.raises(AssertionError):
        cl.run_cl_model(factors, result, strata, is1, is2, oos)
    
    
def test_student_t():
    hessian = np.array([[-1/4, 0], [0, -1/4]])
    coef = np.array([[0.5], [0.5]])
    stats = cl._student_t(hessian, coef)
    np.testing.assert_array_almost_equal(stats['coefse'], np.array([1, 1]))
    np.testing.assert_array_almost_equal(stats['t'], np.array([0.5, 0.5]))
    
    
def test_explosion_process():
    factors = np.array([[-1.057 , -1.2241, -1.2881,  0.3644, -1.0761,  0.0945, -0.6888,  1.5321,  0.8713, -0.197 ,  1.7776, -0.3369],
                       [-2.0017,  0.2381, -1.0427,  0.31  , -0.5714,  1.6635, -0.5881, -0.6335, -1.5251, -0.9937, -0.1551, -0.2515]], dtype=np.float32)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    result = np.array([3, 4, 1, 2, 1, 2, 3, 4, 3, 1, 4, 2], dtype=int)
    exploded_factors = np.concatenate((factors, factors[:, result > 1], factors[:, result > 2]), axis=1)
    exploded_strata = np.concatenate((strata, strata[result > 1]+3, strata[result > 2]+6))
    exploded_winners = np.concatenate((result == 1, result[result > 1] == 2, result[result > 2] == 3))
    f, s, w = cl._explosion_process(factors, strata, result, depth=3)
    np.testing.assert_array_almost_equal(exploded_factors, f)
    np.testing.assert_array_almost_equal(exploded_strata, s)
    np.testing.assert_array_almost_equal(exploded_winners, w)
    
    
def test_linalg_solve():
    np.random.seed(100)
    a = np.random.random((24, 24))
    b = np.random.random(24)
    assert np.sum(a) - 280 < 0.1
    x = np.linalg.solve(a, b.reshape((-1, 1)))
    assert np.abs(np.sum(x) - 0.94809770849006014) < 1e-6, "np.linalg.solve is probably buggy"
    
    
# def _test_local_maximum(func, x0, stepsize=0.001, nTests=100):
#     '''Tests whether <x0> locally maximizes <func> by making small steps of size <stepsize> in random directions
#     and looking whether <func> is smaller every time.'''
#     # if verbose:
#     #    print 'If all differences are positive, then we are at a local maximum.'
#     ll0 = func(x0)
#     for _ in xrange(nTests):
#         x = x0 + np.random.randn(len(x0)) * stepsize
#         ll = func(x)
#         try:
#             assert ll0 > ll
#         except AssertionError:
#             print 'x: ', x
#             raise AssertionError('Function became larger at the printed value, although local maximum was claimed.')
#     print 'Test successful.'

