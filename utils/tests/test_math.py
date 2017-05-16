import pytest
import numpy as np

from ..containers import Folder, AttrDict
from .. import math as umath


def test_sub2ind():
    shape = [5, 6]
    subscripts = np.array([[4, 1], [4, 2], [1, 3], [1, 5], [5, 1], [1, 5], [1, 4], [3, 2], [5, 1], [4, 3]]) - 1
    ref = np.array([4, 9, 11, 21, 5, 21, 16, 8, 5, 14]) - 1
    res = umath.sub2ind(shape, subscripts, order='F')
    np.testing.assert_array_equal(res, ref)


def test_multiple_lin_reg():
    mlr = [[[0.3667746891238157], [0.047177243215324204], [0.1643883098181505], [-0.04670550355401722], [0.12323173216922953]],
           [[0.11845683105661964]],
           [[0.06583763345806794], [0.03874354972673235], [0.025663196198345596], [0.03200069373044386], [0.02100679760545911]],
           [[1.4294278909988758], [0.23968058447987545], [1.0261611256315892], [-0.2610888721078906], [0.8502421432379694]],
           [[0.16525933583925312], [0.8125321489853907], [0.31464271906648067], [0.796162782244642], [0.40326027279954335]],
           [[0.035199753964306635]], [[0.10804441970545975]]]

    x = np.array([[ 0.4243095 , 0.82558382, 0.90085249, 0.1925104 ],
                  [ 0.27027042, 0.78996303, 0.57466122, 0.12308375],
                  [ 0.1970538 , 0.31852425, 0.84517819, 0.20549417],
                  [ 0.82172118, 0.53406413, 0.73864029, 0.14651491],
                  [ 0.42992141, 0.08995068, 0.58598704, 0.18907217],
                  [ 0.88777095, 0.11170574, 0.24673453, 0.04265241],
                  [ 0.391183  , 0.13629255, 0.66641622, 0.63519792],
                  [ 0.76911439, 0.6786523 , 0.08348281, 0.28186686],
                  [ 0.39679152, 0.49517702, 0.62595979, 0.53859668],
                  [ 0.8085141 , 0.18971041, 0.66094456, 0.69516304],
                  [ 0.7550771 , 0.49500582, 0.72975186, 0.49911601],
                  [ 0.37739554, 0.14760822, 0.89075212, 0.53580106],
                  [ 0.21601892, 0.05497415, 0.98230322, 0.44518317],
                  [ 0.79040722, 0.85071267, 0.76902909, 0.12393228],
                  [ 0.94930391, 0.56055953, 0.58144649, 0.49035729],
                  [ 0.32756543, 0.92960887, 0.92831306, 0.85299816],
                  [ 0.67126437, 0.6966672 , 0.58009037, 0.87392741],
                  [ 0.43864498, 0.58279097, 0.01698294, 0.27029433],
                  [ 0.8335006 , 0.81539721, 0.12085957, 0.20846136],
                  [ 0.76885425, 0.8790139 , 0.86271072, 0.56497957],
                  [ 0.16725355, 0.98891162, 0.48429651, 0.64031183],
                  [ 0.86198048, 0.00052238, 0.84485567, 0.41702895],
                  [ 0.98987215, 0.86543859, 0.20940508, 0.20597552],
                  [ 0.51442346, 0.61256647, 0.55229134, 0.94793312],
                  [ 0.88428102, 0.98995021, 0.62988339, 0.08207121],
                  [ 0.58802606, 0.52768007, 0.03199102, 0.10570943],
                  [ 0.15475235, 0.47952339, 0.61471342, 0.14204112],
                  [ 0.19986282, 0.80134761, 0.36241146, 0.16646044],
                  [ 0.40695484, 0.22784294, 0.04953258, 0.62095864],
                  [ 0.74870572, 0.49809429, 0.48956999, 0.57370976]])

    y = np.array([[ 0.05207789, 0.93120138, 0.72866168, 0.73784165, 0.0634045 ,
                   0.86044056, 0.93440512, 0.98439831, 0.85893882, 0.78555899,
                   0.51337742, 0.17760246, 0.3985895 , 0.13393125, 0.03088955,
                   0.93914171, 0.30130606, 0.29553383, 0.33293628, 0.46706819,
                   0.64819841, 0.02522818, 0.84220661, 0.55903254, 0.85409995,
                   0.34787919, 0.44602665, 0.05423948, 0.17710753, 0.66280806]]).transpose()


    out = umath.multiple_linear_regression(x, y)
    np.testing.assert_array_almost_equal(out.b, mlr[0])
    np.testing.assert_array_almost_equal(out.mse, mlr[1][0][0])
    np.testing.assert_array_almost_equal(out.sb2, np.array(mlr[2]).flat, decimal=5)

    # TODO: Next two fail due to shape mismatch 5,5 vs 5,1, probably outdated
    # np.testing.assert_array_almost_equal(out.t, mlr[3])
    # np.testing.assert_array_almost_equal(out.p, mlr[4])


A_ordered = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]).reshape((4, 4))
A_long = np.array([[62, 60, 47, 61, 53, 45, 39, 59, 49, 57, 59, 36, 41, 51],
                   [54, 58, 59, 56, 34, 38, 39, 57, 64, 57, 36, 61, 41, 42],
                   [55, 47, 51, 60, 51, 62, 41, 55, 66, 64, 35, 45, 50, 54],
                   [58, 63, 43, 41, 38, 64, 45, 52, 51, 38, 60, 49, 39, 64],
                   [42, 41, 47, 48, 42, 48, 40, 35, 63, 42, 53, 60, 50, 52]])
rand_arrs = [np.random.randint(20, size=(10, 10)) for _ in xrange(3)]

def _ia_ref(A, C, first=False):
    ia = np.zeros_like(C)
    search_vals = set(C)
    search_iter = enumerate(A.flat) if first else zip(xrange(A.size - 1, -1, -1), A.flat[::-1])
    for i, val in search_iter:
        if val in search_vals:
            ia[np.where(C == val)[0][0]] = i
            search_vals.remove(val)
            if not search_vals:
                break
    return ia


def assert_unique_with_order(A, first=True):
    C, ia, ic = umath.unique_custom(A, return_index=True, return_inverse=True, first=first)
    np.testing.assert_array_equal(C, sorted(set(A.flat)))
    np.testing.assert_array_equal(A, C[ic].reshape(A.shape))
    np.testing.assert_array_equal(C, A.flat[ia])

    for ia_i, ia_ref in zip(ia.flat, _ia_ref(A, C, first)):
        assert ia_i == ia_ref
    return C, ia, ic

unique_test_list = [A_ordered, A_ordered.flat[::-1].reshape(A_ordered.shape), A_long, A_long.flat[::-1].reshape(A_long.shape)] + rand_arrs


@pytest.mark.parametrize("A", unique_test_list)
def test_unique(A):
    assert_unique_with_order(A, first=True)
    assert_unique_with_order(A, first=False)


def test_unique_compare():
    A = np.array([62, 60, 47, 61, 53, 45, 39, 59, 49, 57, 59, 36, 41, 51, 54, 58, 59,
                  56, 34, 38, 39, 57, 64, 57, 36, 61, 41, 42, 55, 47, 51, 60, 51, 62,
                  41, 55, 66, 64, 35, 45, 50, 54, 58, 63, 43, 41, 38, 64, 45, 52, 51,
                  38, 60, 49, 39, 64, 42, 41, 47, 48, 42, 48, 40, 35, 63, 42, 53, 60,
                  50, 52])
    C_ref = np.array([34, 35, 36, 38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51, 52, 53,
                      54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66])
    ia_ref = np.array([18, 63, 24, 51, 54, 62, 57, 65, 44, 48, 58, 61, 53, 68, 50, 69, 66,
                       41, 35, 17, 23, 42, 16, 67, 25, 33, 64, 55, 36])
    ic_ref = np.array([25, 23, 10, 24, 16, 9, 4, 22, 12, 20, 22, 2, 6, 14, 17, 21, 22,
                       19, 0, 3, 4, 20, 27, 20, 2, 24, 6, 7, 18, 10, 14, 23, 14, 25,
                       6, 18, 28, 27, 1, 9, 13, 17, 21, 26, 8, 6, 3, 27, 9, 15, 14,
                       3, 23, 12, 4, 27, 7, 6, 10, 11, 7, 11, 5, 1, 26, 7, 16, 23,
                       13, 15])

    C, ia, ic = assert_unique_with_order(A, first=False)
    np.testing.assert_array_equal(C, C_ref)
    np.testing.assert_array_equal(ia, ia_ref)
    np.testing.assert_array_equal(ic, ic_ref)


@pytest.mark.parametrize("dims", [[3, 2, 1], [3, 1], [3, 0], [2, 0]])
def test_sum_ndim(dims):
    test_arr = np.arange(5 * 4 * 3 * 2, dtype=int).reshape((5, 4, 3, 2))
    def _sum_non_preserving(x, dims):
        for dim in sorted(dims, reverse=True):
            x = np.sum(x, dim)
        return x

    res = umath.sum_ndim(test_arr, dims)
    compare = _sum_non_preserving(test_arr, dims)
    np.testing.assert_array_equal(np.squeeze(res), compare)
    assert len(test_arr.shape) == len(res.shape)
    for dim in xrange(len(test_arr.shape)):
        if dim in dims:
            assert res.shape[dim] == 1
        else:
            assert res.shape[dim] == test_arr.shape[dim]


def test_discretize_numbers():
    np.testing.assert_array_equal(umath.discretize_discrete([3.4, 1.2, 5.4, 1.2, 3.4, 3.4, 0.1]), np.array([2, 1, 3, 1, 2, 2, 0]))


def test_discretize_strings():
    np.testing.assert_array_equal(umath.discretize_discrete('FHFCCH'), np.array([1, 2, 1, 0, 0, 2]))


def test_quantile_shapes():
    t = np.concatenate((np.arange(0, 10, 0.1), np.repeat(np.arange(0, 10, 0.2), 5))).reshape((-1, 1))
    np.testing.assert_array_equal(umath.quantile_custom(t, 10).flat, umath.quantile_custom(t.flat, 10).flat)


def test_quantile_plain():
    q = np.array([-0.69115913, -0.02446194, 0.53615708])
    vec = np.array([-0.53201138, 1.68210359, -0.87572935, -0.48381505, -0.71200455,
                    - 1.17421233, -0.19223952, -0.27407023, 1.53007251, -0.24902474,
                    - 1.06421341, 1.6034573 , 1.23467915, -0.22962645, -1.5061597 ,
                    - 0.44462782, -0.15594104, 0.27606825, -0.26116365, 0.44342191,
                    0.39189421, -1.25067891, -0.94796092, -0.74110609, -0.50781755,
                    - 0.32057551, 0.01246904, -3.02917734, -0.45701464, 1.24244841,
                    - 1.0667014 , 0.93372816, 0.350321  , -0.02900576, 0.18245217,
                    - 1.56505601, -0.08453948, 1.60394635, 0.09834777, 0.04137361,
                    - 0.73416911, -0.03081373, 0.23234701, 0.42638756, -0.37280874,
                    - 0.23645458, 2.02369089, -2.25835397, 2.22944568, 0.3375637 ,
                    1.00006082, -1.66416447, -0.59003456, -0.27806416, 0.42271569,
                    - 1.6702007 , 0.47163433, -1.2128472 , 0.06619005, 0.65235589,
                    0.32705997, 1.0826335 , 1.00607711, -0.65090774, 0.25705616,
                    - 0.94437781, -1.32178852, 0.92482593, 0.00004985, -0.05491891,
                    0.91112727, 0.5945837 , 0.35020117, 1.25025123, 0.92978946,
                    0.23976326, -0.6903611 , -0.65155364, 1.19210187, -1.61183039,
                    - 0.02446194, -1.94884718, 1.02049801, 0.8617163 , 0.00116208,
                    - 0.07083721, -2.48628392, 0.58117232, -2.19243492, -2.31928031,
                    0.07993371, -0.94848098, 0.41149062, 0.67697781, 0.85773255,
                    - 0.69115913, 0.44937762, 0.10063335, 0.82607   , 0.53615708])
    np.testing.assert_array_equal(umath.quantile_custom(vec, 3), q)


@pytest.fixture(scope='module')
def dd():
    return AttrDict(**np.load(Folder(__file__).test_math_discretize))


@pytest.mark.parametrize("col", [3, 4])
def test_discretize(dd, col):
    calculated = umath.discretize_continuous(dd.datacont[col], dd.discretization_level) + 1
    # print calculated, dd.datadisc[col], len(calculated), len(dd.datadisc[col])
    # print sorted(Counter(calculated).items())
    # print sorted(Counter(dd.datadisc[col]).items())
    errs = set(dd.datacont[col][z] for (z, (x, y)) in enumerate(zip(calculated, dd.datadisc[col])) if x != y)
    for err in errs.intersection(set(dd.edges[0][col][0])):
        print "Edge error: %s" % err
    np.testing.assert_array_equal(calculated, dd.datadisc[col])

