import pytest
import itertools
import numpy as np

from utils import MINUTE, HOUR, AttrDict, nbtools as nbt
from utils.accumarray import uaccum
from utils.arrayview import ArrayView, TimeseriesView
from utils.database.tests.test_bbdb import dummydb  # @UnusedImport for pytest

from prediction.models import prediction as ftp
from prediction.sim.jbettingsim import compute_probabilities, JSimCache
from ..probabilities import ProbabilitiesCL

from core.strategies.strategy import JEvalInput, JEvalCache

np.set_printoptions(suppress=True)

# def test_slicenum():
#     def slicenum_by_ttm_ref(ttm, ttm_slice=ftp.TTM_SLICE):
#         return np.maximum(np.searchsorted(ttm_slice, ttm, side='right') - 1, 0)
#     for ttm_base in ftp.TTM_SLICE:
#         for offset in [-0.1, 0, 0.1]:
#             ttm = ttm_base + offset
#             assert ftp.slicenum_by_ttm_jitted(ttm) == slicenum_by_ttm_ref(ttm)
#
#
# @pytest.fixture(scope='module')
# def tsav():
#     av = ArrayView.dummy(events=5)
#     ts = TimeseriesView.dummy_from_av(av, steps=5, nonrunner_chance=0)
#     return AttrDict(av=av, ts=ts)
#
#
# def test_raw_probs_unavailable(tsav):
#     for sl in tsav.ts.slices():
#         tsl = tsav.ts[sl]
#         with np.errstate(invalid='ignore'):
#             coefs = ftp.step2coefs[ftp.slicenum_by_ttm_jitted(25 * HOUR)]
#             res = ftp.factors_to_probs(coefs, [1 / tsl.back_price, 1 / tsl.lay_price, np.ones_like(tsl.back_price) * np.nan],
#                                        logarithmize=[True, True, True])
#         assert np.all(np.isnan(res)), "Values created even if no probs were available"
#         assert len(tsl) == len(res)
#
#
# def test_calculation_2stepped(tsav):
#     for sl in tsav.ts.slices():
#         tsl = tsav.ts[sl]
#         probs = tsav.av.step1probs[tsav.av.row_lookup(tsl.run_id)]
#         with np.errstate(invalid='ignore'):
#             coefs = ftp.step2coefs[ftp.slicenum_by_ttm_jitted(10 * MINUTE)]
#             res = ftp.factors_to_probs(coefs, [1 / tsl.back_price, 1 / tsl.lay_price, probs],
#                                        logarithmize=[True, True, True])
#         assert not np.any(np.isnan(res)), "Result contains nan values"
#         assert len(tsl) == len(res)
#         res.sort()
#         assert np.all(np.diff(res)), "Prediction contains equal values"


@pytest.fixture(scope='module')
def data():
    factors = np.array([[ 0.6804414392, 0.7315325141, 0.1297504902, 0.5850496888,
             0.963049829 , 0.0313311517, 0.6775838137, 0.4759249389,
             0.0303387605, 0.4591667056],
           [ 0.8866195083, 0.6097097993, 0.4404956996, 0.9557273984,
             0.698363483 , 0.641972959 , 0.6053719521, 0.0471264571,
             0.6279424429, 0.8474975824],
           [ 0.1887788177, 0.540828228 , 0.1532587409, 0.213797614 ,
             0.1520240009, 0.104133971 , 0.0727923214, 0.4385688305,
             0.4826028049, 0.960827589 ],
           [ 0.2890259027, 0.4476162493, 0.1575058103, 0.5281149745,
             0.4142432213, 0.7384464145, 0.2450834662, 0.8196508288,
             0.9600215554, 0.5643634796],
           [ 0.538415134 , 0.3247479498, 0.1229971573, 0.726606667 ,
             0.2595875263, 0.8001584411, 0.711612165 , 0.2296330184,
             0.1166569889, 0.7500155568],
           [ 0.6163023114, 0.2094457895, 0.9835837483, 0.6799826026,
             0.4840016663, 0.1933582127, 0.8576578498, 0.5463132262,
             0.0381744429, 0.7833901048],
           [ 0.7378384471, 0.7208883762, 0.0577742197, 0.6045768261,
             0.9935530424, 0.5992178321, 0.567887485 , 0.2873377502,
             0.1146442667, 0.7961348295],
           [ 0.8959595561, 0.0599570461, 0.9558240771, 0.1562099755,
             0.5809192061, 0.642357707 , 0.3195770383, 0.7953734398,
             0.466463089 , 0.1003348008],
           [ 0.1680349708, 0.2468525022, 0.592169404 , 0.1078730747,
             0.0930862054, 0.1579206735, 0.9349926114, 0.1049669385,
             0.0485154763, 0.0207382217],
           [ 0.0924504995, 0.5910742879, 0.8041478395, 0.9828779697,
             0.1027580649, 0.2041040212, 0.9846781492, 0.7149704099,
             0.8873822689, 0.8275128603],
           [ 0.1383778453, 0.1992287338, 0.3582637608, 0.652751863 ,
             0.3127159178, 0.4666547477, 0.5012752414, 0.4002668858,
             0.235883072 , 0.7820643783],
           [ 0.0975357741, 0.0852985978, 0.3783034682, 0.1693547964,
             0.0593044683, 0.589998126 , 0.8958294392, 0.5522723198,
             0.8169540167, 0.6859343648],
           [ 0.0504906997, 0.0918064937, 0.093665272 , 0.7853648067,
             0.1024331301, 0.0040999772, 0.68576473  , 0.1451767236,
             0.3586080074, 0.9714791179],
           [ 0.8165338635, 0.9760597944, 0.8417123556, 0.8371312618,
             0.405321002 , 0.0992325768, 0.0567413419, 0.8183228374,
             0.0966112688, 0.642106235 ],
           [ 0.7839481235, 0.0246551447, 0.4016645849, 0.812780261 ,
             0.759144485 , 0.486305654 , 0.2404744923, 0.708199203 ,
             0.0388902389, 0.5058290362],
           [ 0.6798810959, 0.2346500009, 0.2413725257, 0.0434444062,
             0.4990884066, 0.8091067672, 0.1666328758, 0.1728008538,
             0.8600120544, 0.5165944695],
           [ 0.5704731345, 0.5170003772, 0.4003915787, 0.3735976517,
             0.25887537  , 0.0331901498, 0.6243562102, 0.1344421059,
             0.4575660825, 0.2958861589],
           [ 0.9129708409, 0.1947141588, 0.6725312471, 0.2691006958,
             0.3654453754, 0.0653966963, 0.3057087064, 0.1966750175,
             0.41234079  , 0.6262252927],
           [ 0.3427227437, 0.2284743488, 0.9914050102, 0.6353655457,
             0.3919232786, 0.0244871862, 0.7474147081, 0.3653643727,
             0.6395674348, 0.5736994147],
           [ 0.458745569 , 0.6432455778, 0.5146945119, 0.835639298 ,
             0.8622787595, 0.5871462822, 0.6222221851, 0.4667364955,
             0.5244920254, 0.9125919938],
           [ 0.4380195141, 0.519518733 , 0.7491316199, 0.7767777443,
             0.5401954055, 0.9871909022, 0.5944755673, 0.8068993092,
             0.4150748253, 0.9717165828],
           [ 0.9200573564, 0.7369067669, 0.7167891264, 0.6626301408,
             0.0991018265, 0.9400367737, 0.6221994162, 0.9256386757,
             0.3707442582, 0.5732507706],
           [ 0.5593857169, 0.2155069709, 0.0240942016, 0.3341625631,
             0.1802887619, 0.2554274797, 0.4624139369, 0.3886986673,
             0.4294630587, 0.6034608483],
           [ 0.3219775259, 0.2564212084, 0.1462819278, 0.0030913879,
             0.8274508119, 0.1649028659, 0.6492583156, 0.9413766265,
             0.0659168288, 0.1878880858],
           [ 0.6848681569, 0.5569753051, 0.4331966639, 0.1858922988,
             0.6753236055, 0.9783237576, 0.2503733337, 0.204095006 ,
             0.6033113599, 0.8169184923],
           [ 0.9130730033, 0.1380343884, 0.7622727752, 0.6154682636,
             0.6504223347, 0.6661753058, 0.7993372083, 0.9762411118,
             0.6073663235, 0.9408424497],
           [ 0.2041170448, 0.3450306058, 0.4077080488, 0.4401132464,
             0.3336454928, 0.8582446575, 0.1776751578, 0.9311425686,
             0.6704463959, 0.7141986489],
           [ 0.7133659124, 0.4880221784, 0.3623979092, 0.3144557178,
             0.228438288 , 0.4551691115, 0.5841957331, 0.3084807396,
             0.725297749 , 0.2665587664],
           [ 0.8228152394, 0.8363196254, 0.9537758827, 0.0587875508,
             0.8289164901, 0.9367877841, 0.5569429398, 0.9772160649,
             0.8940296173, 0.915307343 ],
           [ 0.4616784751, 0.0461071841, 0.3922715187, 0.9343735576,
             0.8178443313, 0.1130584255, 0.4161815941, 0.7873948216,
             0.2733545601, 0.5832310319],
           [ 0.4739804864, 0.4580272734, 0.257750392 , 0.7353748083,
             0.9701277018, 0.0762523636, 0.8054041862, 0.4987441003,
             0.0042147362, 0.2732792497],
           [ 0.4739296734, 0.9344451427, 0.4699712694, 0.5227239728,
             0.0916686431, 0.0725754499, 0.3094133735, 0.5490608215,
             0.9331766367, 0.6974585056],
           [ 0.5361135602, 0.6972301006, 0.4546348751, 0.3311187625,
             0.7222708464, 0.4287602305, 0.1074622869, 0.0529883541,
             0.3672127128, 0.3973879218],
           [ 0.1389008164, 0.9797748327, 0.1616417915, 0.9155862927,
             0.99648875  , 0.5357746482, 0.9492073059, 0.0946694314,
             0.0175810307, 0.8019135594],
           [ 0.1834013313, 0.4251140654, 0.4831991792, 0.2233883739,
             0.6030051708, 0.3060545921, 0.531077981 , 0.0323349722,
             0.7547754645, 0.5136209726],
           [ 0.708123982 , 0.3274577558, 0.7557792664, 0.3557890356,
             0.7830867171, 0.022611659 , 0.2904182076, 0.1272953004,
             0.6369996071, 0.0590347238],
           [ 0.472138375 , 0.5541568398, 0.1835210174, 0.1641996205,
             0.1253886819, 0.5944344401, 0.9237655401, 0.8362447619,
             0.8502215147, 0.8410412669],
           [ 0.8033889532, 0.9419650435, 0.8771784902, 0.5312062502,
             0.9604707956, 0.7002931237, 0.8841437697, 0.7528274059,
             0.5803533792, 0.8754288554],
           [ 0.7158628106, 0.9136520624, 0.573141396 , 0.7509226203,
             0.28907004  , 0.0828049704, 0.9367245436, 0.4110031128,
             0.090073131 , 0.3754843771],
           [ 0.0217951331, 0.8744844198, 0.2244863361, 0.0336801708,
             0.8130235076, 0.7798473239, 0.791218996 , 0.9983395934,
             0.8196367621, 0.8900091052],
           [ 0.1658981591, 0.2108718753, 0.7750349045, 0.5194205642,
             0.3783816993, 0.9421516657, 0.7717604041, 0.4078520238,
             0.9527984858, 0.1900920719],
           [ 0.7673528194, 0.8704245687, 0.4786103368, 0.3305361867,
             0.0819948241, 0.1384977251, 0.0470337421, 0.5653733611,
             0.2070083767, 0.4439738393],
           [ 0.6488838792, 0.6152307391, 0.5077576041, 0.1564115286,
             0.1066650227, 0.9341997504, 0.8661872745, 0.5997076631,
             0.6698511243, 0.3707255125],
           [ 0.0752729774, 0.3262940049, 0.0425854921, 0.3099196553,
             0.3923858106, 0.8408157229, 0.7272081971, 0.1002204418,
             0.5804655552, 0.8147228956],
           [ 0.9375543594, 0.5420121551, 0.7837378979, 0.697376132 ,
             0.4672240019, 0.3688814938, 0.2063729763, 0.869564712 ,
             0.5802010894, 0.642808497 ],
           [ 0.9124888182, 0.0608549416, 0.775300622 , 0.1991192698,
             0.0624658018, 0.5802386403, 0.9403300285, 0.5885403156,
             0.852987349 , 0.1873560995],
           [ 0.9781572223, 0.7401968837, 0.2729397714, 0.8818359375,
             0.0731307343, 0.9736234546, 0.1922616661, 0.5726189017,
             0.5769479871, 0.6003339291],
           [ 0.7847774625, 0.8401702642, 0.1918340027, 0.9513038397,
             0.6985769868, 0.5672914982, 0.4257274866, 0.4198988378,
             0.1807788014, 0.2101880312],
           [ 0.6137747169, 0.3729198575, 0.0236831028, 0.4089342654,
             0.8526545763, 0.135226205 , 0.1293704361, 0.3803837895,
             0.1975028664, 0.0903235823],
           [ 0.2751413584, 0.3291722536, 0.3339657784, 0.2993099689,
             0.0237522721, 0.6652520895, 0.4109492898, 0.9273298383,
             0.5844188333, 0.1762014329],
           [ 0.9618358612, 0.9879864454, 0.7858480215, 0.9274845719,
             0.3538906276, 0.4186618626, 0.6629226208, 0.485627085 ,
             0.3369825482, 0.260864526 ],
           [ 0.4523414075, 0.1158154681, 0.2269242108, 0.2658584416,
             0.2258625925, 0.6369449496, 0.5855545998, 0.5764486194,
             0.3614463806, 0.4968906641],
           [ 0.7806646824, 0.5313808322, 0.9994851351, 0.7760959268,
             0.052997198 , 0.6783127189, 0.6968601346, 0.9484143853,
             0.4238670468, 0.8696386814],
           [ 0.1651779264, 0.281306684 , 0.9493367672, 0.3390839398,
             0.4442096353, 0.3792173564, 0.2495399415, 0.7691108584,
             0.6497272253, 0.1708332896],
           [ 0.4277083278, 0.32663697  , 0.2421234697, 0.7544845343,
             0.0773213059, 0.5866292715, 0.0015424612, 0.0431322642,
             0.0439459868, 0.5010769367],
           [ 0.0276513211, 0.1526308507, 0.8730093837, 0.700078547 ,
             0.7015269995, 0.5248752832, 0.9730814099, 0.5405934453,
             0.782644093 , 0.3655663431],
           [ 0.5761326551, 0.39641276  , 0.1907742023, 0.276640743 ,
             0.5300590396, 0.7704452276, 0.5865975618, 0.8367540836,
             0.9717620611, 0.3731811345]], dtype=np.float32)


    ei = JEvalInput()
    ei.nruns = 10
    ei.probs[:ei.nruns] = np.zeros(10, dtype=np.float32)
    ei.bprice[:ei.nruns] = np.array([ 12.5, 8.3299999237, 9.0900001526, 3.2300000191,
             8.3299999237, np.nan, 10., 14.2899999619,
            14.2899999619, 14.2899999619], dtype=np.float32)
    ei.lprice[:ei.nruns] = np.array([ 12.5, 9.0900001526, 10., 3.5699999332,
             9.0900001526, np.nan, 11.1099996567, 14.2899999619,
            14.2899999619, 14.2899999619], dtype=np.float32)
    
    bp, lp = ei.bprice[:ei.nruns].astype(np.float64), ei.lprice[:ei.nruns].astype(np.float64)

    coefs = np.array([[ 0.          , 0.8754401013, 0.0205621759, -0.0202165931,
           - 0.0180567007, -0.0073436342, 0.0070844472, 0.0192662404,
            0.0187478662, 0.1004781956, 0.0213397372, 0.0324847821,
            0.0237588167, 0.006566073 , -0.0031102451, -0.0319664079,
           - 0.0235860253, 0.0118362105, 0.1701131274, -0.0027646623,
           - 0.0243635866, -0.0108858578, 0.0063932816, 0.0593538438,
           - 0.0323119907, 0.0213397372, -0.0308432638, 0.0041469935,
           - 0.055034059 , 0.0123545847, 0.006566073 , 0.0160695996,
            0.0042333892, -0.0050109504, -0.0110586492, 0.0066524687,
            0.0215125286, 0.0038878064, 0.0599586137, 0.0291153499,
            0.100650987 , -0.0020734967, -0.0366317755, 0.0220309027,
            0.0173655351, 0.0184022835, -0.0069980515, -0.0391372507,
           - 0.0026782666, -0.0356814229, -0.0038878064, 0.001987101 ,
            0.0210805501, 0.0266962704, 0.0239316081, 0.0140824986,
            0.1469590806, 0.1009101741, 0.0013823312]])
    return AttrDict(locals())



def prepare_step1(coefs, strata, factors):
    # TODO: This doesn't contain any nan check
    step1probs = np.full((coefs.shape[0], factors.shape[1]), np.nan)
    for sl in xrange(coefs.shape[0]):
        step1probs[sl] = ProbabilitiesCL._compute(coefs[sl, 2:], factors, strata)
    return step1probs


def predict_step1(bprice, lprice, factors, coefs):
    """ New reference function for creating step1 probs with all factors weighted in directly.
        The result should still be fed through step2 ftp.
    """
    try:
        all_factors = np.concatenate((np.log(1 / bprice.astype(np.float64).reshape((1, -1))),
                                      np.log(1 / lprice.astype(np.float64).reshape((1, -1))),
                                      factors.astype(np.float64)), axis=0)
    except ValueError as e:
        raise ValueError(e.message + '\n' + '\n    '.join((str(bprice), str(lprice), str(factors))))
    V = np.dot(coefs, all_factors)
    V -= nbt.nanmean(V)
    expV = np.exp(V)
    return (expV / nbt.nansum(expV)).astype(np.float32)


def predict_step2(bprice, lprice, probs_step1, coefs):
    """ New reference function for mixing prices and raw probabilities together according to
        given coefficients grouped by market. Needs further speedup by replacing probs_cl
        with a faster and more strapped down version.
    """
    if np.all(np.isnan(probs_step1)):
        return probs_step1
    try:
        factors = np.log(np.concatenate((1 / bprice.reshape((1, -1)),
                                         1 / lprice.reshape((1, -1)),
                                         probs_step1.reshape((1, -1))), axis=0))
    except ValueError as e:
        raise ValueError(e.message + '\n' + '\n    '.join((str(bprice), str(lprice), str(probs_step1))))
    V = np.dot(coefs, factors)
    V -= nbt.nanmean(V)
    expV = np.exp(V)
    return expV / nbt.nansum(expV)



def test_probability_computation(data):
    ei = data.ei
    slicenum = 0
    factors = [1 / data.bp, 1 / data.lp, data.factors]
    py_probs = ftp.factors_to_probs(data.coefs[slicenum], factors, logarithmize=[True, True])

    sc = JSimCache()
    sc.row_id[:ei.nruns] = np.arange(ei.nruns)

    strength = np.dot(data.coefs[slicenum, 2:], data.factors)
    step1probs = np.exp(strength) / np.sum(np.exp(strength))
    coefs = np.ones(3)
    coefs[:2] = data.coefs[slicenum, :2]
    ec = JEvalCache()
    compute_probabilities(ei, ec, sc, step1probs, coefs)
    np.testing.assert_array_almost_equal(py_probs, ei.probs[:ei.nruns], decimal=10)

def test_prepare_step1(data):
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    coefs = np.random.randn(11, data.factors.shape[0] + 2)
    probs1 = prepare_step1(coefs, strata, data.factors)
    probs2 = ftp.factors_to_probs(coefs[:, 2:], data.factors, strata)
    np.testing.assert_array_almost_equal(probs1.astype(np.float32), probs2, decimal=15)

def test_predict_step1(data):
    probs1 = predict_step1(data.bp, data.lp, data.factors, data.coefs[0, :])
    factors = [1 / data.bp, 1 / data.lp, data.factors]
    probs2 = ftp.factors_to_probs(data.coefs[0, :], factors, logarithmize=[True, True])
    np.testing.assert_array_almost_equal(probs1, probs2, decimal=15)

def test_predict_step2(data):
    coefs = np.random.randn(3)
    probs1 = predict_step2(data.bp, data.lp, np.exp(data.factors[0, :]), coefs)
    factors = [1 / data.bp, 1 / data.lp, np.exp(data.factors[0, :])]
    probs2 = ftp.factors_to_probs(coefs, factors, logarithmize=[True, True, True])
    np.testing.assert_array_almost_equal(probs1.astype(np.float32), probs2, decimal=15)

import pdb

def test_sizes(data):
    nSlices = np.random.randint(20) + 2
    nFactors = data.factors.shape[0]
    coefs = np.random.randn(nSlices, nFactors)
    probs = ftp.factors_to_probs(coefs, data.factors)
    assert probs.shape == (nSlices, data.factors.shape[1])

    # factor matrices dont have equal length
    with pytest.raises(ValueError):
        factors = [1 / data.bp[:5], 1 / data.lp, data.factors]
        probs = ftp.factors_to_probs(coefs, factors)

    # strata length does not equal factor length
    with pytest.raises(ValueError):
        event_id = np.array([10, 10, 10, 10, 1, 1, 1, 1, 1], dtype=int)
        probs = ftp.factors_to_probs(coefs, data.factors, event_id)

    # coefs matrix does not fit factor matrix
    with pytest.raises(ValueError):
        coefs = np.random.randn(nSlices, nFactors + 1)
        probs = ftp.factors_to_probs(coefs, data.factors)


# @pytest.mark.parametrize(["bad_num", "content"], itertools.product([0, 1], [np.nan, np.inf]))
@pytest.mark.parametrize("content", [np.nan, np.inf])
def test_nans(data, content):
    coefs = np.random.randn(7, data.factors.shape[0])
    i, j = np.random.randint(coefs.shape[0]), np.random.randint(coefs.shape[1])
    coefs[i, j] = content
    with pytest.raises(ValueError):
        ftp.factors_to_probs(coefs, data.factors)

@pytest.mark.parametrize("logarithmize", itertools.product([False, True], repeat=3))
def test_logarithmize(data, logarithmize):
    nFactors = data.factors.shape[0]
    coefs = np.random.randn(5, nFactors + 2)
    factors = [1 / data.bp.reshape((1,-1)), 1 / data.lp.reshape((1,-1)), np.exp(data.factors)]
    for i, f in enumerate(factors):
        if logarithmize[i]:
            i, j = np.random.randint(f.shape[0]), np.random.randint(f.shape[1])
            f[i, j] = -np.random.rand() if np.random.rand() > 0.5 else 0.0
    if any(logarithmize):
        with pytest.raises(ValueError):
            ftp.factors_to_probs(coefs, factors, logarithmize=list(logarithmize))
    else:
        ftp.factors_to_probs(coefs, factors, logarithmize=list(logarithmize))

def test_large_entries(data):
    for _ in xrange(100):
        coefs = np.random.randn(5, data.factors.shape[0])
        i, j = np.random.randint(data.factors.shape[0]), np.random.randint(data.factors.shape[1])
        data.factors[i, j] = 1000.0
        with pytest.raises(ValueError):
            ftp.factors_to_probs(coefs, data.factors)

def test_probabilities_reproduced():
    ''' When coefficients are 1, probabilities are reproduced by the softmax '''
    coefs = np.array([[1.0]])
    probs0 = np.array([0.1, 0.4, 0.45, 0.05])
    probs = ftp.factors_to_probs(coefs, probs0, logarithmize=[True])
    np.testing.assert_array_almost_equal(probs0, probs, decimal=7)

def test_sum_to_one():
    for _ in xrange(10):
        n = 1000
        event_id = np.zeros(n, dtype=int)
        val = 52
        for i in xrange(n):
            val = np.random.randint(50) + 600 if np.random.rand() < 0.1 else val
            event_id[i] = val
        factors = []
        nC = 0
        for i in xrange(np.random.randint(5) + 3):
            nF = np.random.randint(30) + 1
            factors += [np.random.randn(nF, n)]
            # sprinkle some nans
            for _ in xrange(5):
                x, y = np.random.randint(factors[-1].shape[0]), np.random.randint(factors[-1].shape[1])
                factors[-1][x, y] = np.nan
            nC += nF
        coefs = np.random.randn(6, nC)
        probs = ftp.factors_to_probs(coefs, factors, event_id=event_id)
        for sl in xrange(coefs.shape[0]):
            assert np.all((probs[sl] > 0) | np.isnan(probs[sl]))
            sum_probs = uaccum(event_id, probs[sl], func='nansum')
            ONES = np.ones(n)
            if np.any(sum_probs == 0):
                assert np.all(np.isnan(probs[sl][sum_probs == 0]))
                ONES[sum_probs == 0] = 0
            np.testing.assert_array_almost_equal(sum_probs, ONES, decimal=7)

def test_particular_io():
    coefs = np.array([[1, 2]])
    p1 = np.array([0.2, 0.7, 0.1])
    p2 = np.array([0.13, 0.27, 0.6])
    S = p1 * p2 ** 2
    probs0 = S / np.sum(S)
    probs1 = ftp.factors_to_probs(coefs, [p1, p2], logarithmize=[True, True])
    np.testing.assert_array_almost_equal(probs0, probs1, decimal=7)

