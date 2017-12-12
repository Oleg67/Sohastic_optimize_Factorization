from numpy import array
from ...utils import timestamp
# earlier than this, the data are considered too old even for factor pre-processing and building
factor_build_start = float(timestamp('2013-01-01'))
# factor build and pre-processing shall not touch data later than this. After out of sample start, the data is used for fitting and simulation.
factor_build_end = float(timestamp('2016-04-01'))

ep = 93.97


step1coefs = {'-log(back_price)': array([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
 '-log(lay_price)': array([ 0.73337606, 0.70244162, 0.68127266, 0.64607716, 0.53733987,
        0.45989563, 0.30011993, 0.19482324, 0.17518893, 0.10886678,
        0.10886678]),
 'weight': array([-0.01398177, -0.01441507, -0.01650105, -0.01940143, -0.02498306,
       - 0.02702062, -0.03011127, -0.03328317, -0.03333333, -0.03005818,
       - 0.03005818]),
 'z027f9f0f5': array([ 0.01853364, 0.01438764, 0.00445251, -0.00770873, -0.05173254,
       - 0.08180293, -0.16010166, -0.22521977, -0.23908705, -0.28026401,
       - 0.28026401]),
 'z0f50ce7c0': array([ 0.00295896, 0.00095748, -0.0009486 , -0.00276245, -0.0060378 ,
       - 0.00925057, -0.01247309, -0.01330918, -0.01263178, -0.01294066,
       - 0.01294066]),
 'z106842cce': array([ 0.00225839, 0.00159316, 0.00068818, -0.0025585 , -0.00541499,
       - 0.00789123, -0.01142752, -0.01135605, -0.01258845, -0.01187626,
       - 0.01187626]),
 'z1b0bd4349': array([ 0.02652573, 0.02606195, 0.02635924, 0.02729972, 0.03253703,
        0.03373955, 0.0355099 , 0.03572044, 0.03614221, 0.03525122,
        0.03525122]),
 'z215129aeb': array([ 0.01954761, 0.02122982, 0.02090767, 0.02062026, 0.0243331 ,
        0.02610505, 0.02678355, 0.02702537, 0.02760891, 0.03035267,
        0.03035267]),
 'z245159235': array([ 0.02026984, 0.0212801 , 0.02390507, 0.02840604, 0.0335607 ,
        0.03744197, 0.04053403, 0.04542833, 0.04808194, 0.05121821,
        0.05121821]),
 'z28e932d84': array([ 0.03659475, 0.03767423, 0.04192753, 0.04291097, 0.05077957,
        0.06024461, 0.0647409 , 0.07283241, 0.06306308, 0.0659419 ,
        0.0659419 ]),
 'z34b78e584': array([ 0.01842852, 0.0184861 , 0.02226201, 0.02805016, 0.03779837,
        0.03753074, 0.05351077, 0.06104321, 0.06285592, 0.06414143,
        0.06414143]),
 'z36b8bffe0': array([ 0.02030691, 0.01953749, 0.01936319, 0.02175104, 0.02384416,
        0.027502  , 0.03417063, 0.04160033, 0.04513327, 0.04612478,
        0.04612478]),
 'z37a191205': array([ 0.00377316, 0.00639011, 0.0055042 , 0.00907895, 0.01262829,
        0.01645953, 0.01514888, 0.01910299, 0.01874677, 0.02147948,
        0.02147948]),
 'z3d11336f4': array([-0.01337996, -0.01593644, -0.01531443, -0.01699528, -0.01840763,
       - 0.01895212, -0.02614768, -0.03037777, -0.03354134, -0.03664839,
       - 0.03664839]),
 'z412893062': array([-0.0352695 , -0.0357007 , -0.03730382, -0.04022379, -0.04180205,
       - 0.04398923, -0.04432421, -0.04553236, -0.04686655, -0.04788416,
       - 0.04788416]),
 'z44cf2d196': array([-0.04812321, -0.04669041, -0.05217076, -0.05421792, -0.05421822,
       - 0.06340516, -0.05120935, -0.06493162, -0.04300824, -0.01491332,
       - 0.01491332]),
 'z4a6755a99': array([ 0.04178261, 0.04155722, 0.04385257, 0.04206458, 0.04593595,
        0.05384256, 0.05531746, 0.06322484, 0.05291812, 0.0555907 ,
        0.0555907 ]),
 'z4cac73de3': array([ 0.03038473, 0.02997698, 0.0290965 , 0.02937526, 0.03731822,
        0.03997148, 0.03435538, 0.03643049, 0.03985839, 0.03702539,
        0.03702539]),
 'z4fdabfabc': array([-0.0116419 , -0.00798087, -0.00688386, -0.00562388, -0.0072228 ,
       - 0.00434426, -0.00140795, 0.00090064, 0.00071301, 0.00298509,
        0.00298509]),
 'z50810021e': array([-0.01462643, -0.01435154, -0.01654369, -0.01705297, -0.02075728,
       - 0.02179327, -0.0252631 , -0.02727583, -0.02952272, -0.0286884 ,
       - 0.0286884 ]),
 'z5873666ab': array([ 0.016493  , 0.01836656, 0.01772467, 0.01715197, 0.01592666,
        0.01604083, 0.01574877, 0.0175532 , 0.01989852, 0.02069896,
        0.02069896]),
 'z58de9ec3e': array([ 0.00940407, 0.00941904, 0.01064241, 0.01166428, 0.01364835,
        0.01337289, 0.016423  , 0.02006235, 0.02218176, 0.02418985,
        0.02418985]),
 'z5981b9f89': array([ 0.04208196, 0.04129713, 0.04315315, 0.04607007, 0.0480604 ,
        0.056123  , 0.0601208 , 0.06180192, 0.06069227, 0.06061111,
        0.06061111]),
 'z6809c316d': array([-0.02965651, -0.03017713, -0.03217376, -0.03496467, -0.04144039,
       - 0.04836512, -0.05418738, -0.05723164, -0.05632997, -0.06041566,
       - 0.06041566]),
 'z6c84925d1': array([ 0.01508801, 0.01462374, 0.01507491, 0.01976426, 0.02006275,
        0.02021234, 0.02242236, 0.02407989, 0.02069698, 0.02343284,
        0.02343284]),
 'z6f11029f7': array([-0.00909254, -0.00679196, -0.00706701, -0.00591383, -0.00617176,
       - 0.00492556, -0.00111153, 0.00342577, 0.00435622, 0.00567272,
        0.00567272]),
 'z6f1eaa94d': array([-0.00532671, -0.00631163, -0.0072081 , -0.00735152, -0.01048123,
       - 0.015215  , -0.01301808, -0.01445549, -0.01374706, -0.01439388,
       - 0.01439388]),
 'z7081bf371': array([-0.03606534, -0.03235735, -0.03351033, -0.03646559, -0.02721714,
       - 0.02732785, -0.02215712, -0.02105638, -0.02093743, -0.01920413,
       - 0.01920413]),
 'z7603edce1': array([ 0.01056966, 0.01219165, 0.01256476, 0.01399089, 0.01356859,
        0.01505053, 0.01824606, 0.0213015 , 0.01973557, 0.01928048,
        0.01928048]),
 'z779f698c0': array([-0.00094877, -0.00145751, -0.00048408, 0.00185766, 0.00501983,
        0.01028092, 0.01600935, 0.01857738, 0.01728745, 0.01739372,
        0.01739372]),
 'z77c9cc0a5': array([ 0.01599556, 0.01647089, 0.01825612, 0.02142432, 0.02821367,
        0.03031961, 0.03688143, 0.04053058, 0.04124386, 0.04150415,
        0.04150415]),
 'z81f2598ec': array([-0.00411711, -0.00333494, -0.00256945, -0.00236468, -0.00254088,
       - 0.0019773 , -0.00146623, -0.00085923, -0.00004823, -0.00211218,
       - 0.00211218]),
 'z8ca2ea7f2': array([ 0.00993451, 0.00664331, 0.00380978, 0.00317206, 0.00063891,
        0.0011814 , -0.00580899, -0.0133322 , -0.01737277, -0.02070108,
       - 0.02070108]),
 'z8d2862a46': array([-0.00494151, -0.00606758, -0.00446515, -0.00371954, -0.000501  ,
       - 0.00532829, -0.01356512, -0.01471381, -0.01408535, -0.00962505,
       - 0.00962505]),
 'z953916863': array([-0.00914251, -0.00847402, -0.01015036, -0.01186656, -0.01148855,
       - 0.01410394, -0.02024689, -0.02450272, -0.0209913 , -0.0221563 ,
       - 0.0221563 ]),
 'z999f82560': array([-0.01120522, -0.01103034, -0.01225557, -0.01148841, -0.0161372 ,
       - 0.01539091, -0.02077746, -0.01951788, -0.01908078, -0.02011694,
       - 0.02011694]),
 'za5c443236': array([-0.0088923 , -0.00664776, -0.00694338, -0.00393002, -0.00149087,
        0.00255254, 0.00992163, 0.00944533, 0.00981569, 0.01622263,
        0.01622263]),
 'za5cf91a5d': array([ 0.01061796, 0.01656413, 0.02058163, 0.02273885, 0.0259268 ,
        0.02964675, 0.03674298, 0.0430653 , 0.04314226, 0.04665642,
        0.04665642]),
 'zac38414de': array([ 0.00089771, 0.00295578, 0.00187365, 0.00454574, 0.01595511,
        0.01233489, 0.03315919, 0.03320115, 0.04977455, 0.07634472,
        0.07634472]),
 'zae03d13a5': array([ 0.03764984, 0.03815264, 0.0413071 , 0.0409694 , 0.04679654,
        0.05544077, 0.05864529, 0.06663599, 0.05694314, 0.05969346,
        0.05969346]),
 'zb392bb74a': array([ 0.04537009, 0.04035516, 0.03863203, 0.03286713, 0.01171909,
        0.00323882, -0.03109093, -0.04892706, -0.05040985, -0.07137273,
       - 0.07137273]),
 'zb85d3de89': array([ 0.01719632, 0.00975618, 0.00976718, 0.01156619, 0.00729867,
       - 0.00388535, 0.03295767, 0.05393482, 0.06381449, 0.06488308,
        0.06488308]),
 'zbba3f4349': array([ 0.00785727, 0.00965866, 0.0096892 , 0.01342158, 0.01288293,
        0.01254088, 0.01566357, 0.01765382, 0.01918941, 0.0218213 ,
        0.0218213 ]),
 'zbe307f294': array([-0.0018649 , -0.0029455 , -0.00342089, -0.0069396 , -0.00856773,
       - 0.01145463, -0.00679941, -0.00682267, -0.00527772, -0.00303687,
       - 0.00303687]),
 'zc31456152': array([-0.03172879, -0.03146546, -0.03235122, -0.03786847, -0.04338776,
       - 0.04955895, -0.04869538, -0.05474657, -0.0558146 , -0.05564485,
       - 0.05564485]),
 'zc6614c5ee': array([-0.00141371, 0.00055172, 0.00368748, 0.00787384, 0.01360893,
        0.01732458, 0.02655069, 0.03130183, 0.03301101, 0.03258848,
        0.03258848]),
 'zce609d953': array([-0.02286366, -0.02513889, -0.02806941, -0.02931415, -0.02952803,
       - 0.03167621, -0.03958788, -0.04254805, -0.04384446, -0.0473035 ,
       - 0.0473035 ]),
 'zd393eb79d': array([-0.00436816, -0.0024202 , -0.00262323, -0.00168581, -0.00385411,
       - 0.00396756, -0.00356142, -0.00059704, -0.00176734, 0.00032636,
        0.00032636]),
 'zd7cd94e4c': array([-0.00531156, -0.00560649, -0.00425954, 0.00021203, 0.00426499,
        0.00706234, 0.01476164, 0.01835069, 0.02020776, 0.02049577,
        0.02049577]),
 'zd81bcd220': array([ 0.01488296, 0.01583705, 0.01638604, 0.01785304, 0.01742087,
        0.01890196, 0.01549269, 0.01448951, 0.01488445, 0.01254468,
        0.01254468]),
 'ze4c91eac0': array([ 0.05606445, 0.06350665, 0.06834966, 0.0806113 , 0.12349771,
        0.14513226, 0.17002231, 0.17062136, 0.17593909, 0.18540507,
        0.18540507]),
 'zec0c22a48': array([ 0.0492848 , 0.05533939, 0.06206334, 0.06588114, 0.07479224,
        0.07974179, 0.08808293, 0.0910731 , 0.09414136, 0.09883114,
        0.09883114]),
 'zec3f8f8b9': array([ 0.01555283, 0.01556462, 0.01363982, 0.01522947, 0.01726221,
        0.02292814, 0.02438664, 0.02840253, 0.03097887, 0.03152944,
        0.03152944]),
 'zef5a850eb': array([ 0.04059919, 0.04052605, 0.04270267, 0.0409687 , 0.04478461,
        0.05298374, 0.05457335, 0.06249212, 0.05228232, 0.05499367,
        0.05499367]),
 'zf47a136cc': array([ 0.0412899 , 0.04336642, 0.0461987 , 0.04680499, 0.0550587 ,
        0.06523518, 0.06850094, 0.07468013, 0.06539884, 0.06788359,
        0.06788359]),
 'zf50212bfb': array([ 0.00789983, 0.00878831, 0.00905648, 0.00926321, 0.01055519,
        0.01020571, 0.0168493 , 0.01597437, 0.01682025, 0.01729239,
        0.01729239])}



step2coefs = array([[ 0.        , 0.2963376 , 0.87626779],
       [ 0.        , 0.27997078, 0.89350916],
       [ 0.        , 0.26446144, 0.91391804],
       [ 0.        , 0.25945029, 0.92575432],
       [ 0.        , 0.22608803, 0.96356145],
       [ 0.        , 0.1870958 , 1.0138497 ],
       [ 0.        , 0.15368183, 1.05704772],
       [ 0.        , 0.07890085, 1.15380544],
       [ 0.        , 0.0677619 , 1.16452484],
       [ 0.        , 0.03329594, 1.20859099],
       [ 0.        , 0.03329594, 1.20859099]])
