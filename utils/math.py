from __future__ import division
import sys
from itertools import combinations
from contextlib import contextmanager
import numpy as np
import scipy.stats

from .containers import AttrDict


def sleep(x=None):
    """ Dummy for running non-async """


@contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def logarithmize(x, count=1):
    """Fix high kurtosis numbers by logarithmizing."""
    assert count >= 1, 'Logarithmize at least once'
    for _ in xrange(count):
        x = np.sign(x) * np.log(1 + np.abs(x))
    return x


def pick(x):
    """Pick random value from array"""
    return x[np.random.randint(len(x))]


def set_round(x):
    import ctypes
    libm = ctypes.CDLL('libm.so.6')
    rounding = dict(TONEAREST=0x0000, DOWNWARD=0x0400, UPWARD=0x0800, TOWARDZERO=0x0c00)[x.upper()]
    libm.fesetround(rounding)


def find(x):
    return np.where(x)[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sel_len(sel, total):
    if isinstance(sel, slice):
        sta, sto, ste = sel.indices(total)
        return int(np.ceil((sto - sta) / ste))
    elif isinstance(sel, np.ndarray):
        if sel.dtype.type is np.bool_:
            return np.count_nonzero(sel[:total])
        elif issubclass(sel.dtype.type, np.integer):
            return len(sel)
    elif isinstance(sel, list):
        return len(sel)
    elif isinstance(sel, int):
        return 1
    raise ValueError("Invalid selector supplied: %s" % sel)


def princomp(A, numpc=0):
    """
    Performs principal components analysis (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables. 
    
    Returns :  
     coeff :
       is a p-by-p matrix, each column containing coefficients 
       for one principal component.
     score : 
       the principal component scores; that is, the representation 
       of A in the principal component space. Rows of SCORE 
       correspond to observations, columns to components.
     latent : 
       a vector containing the eigenvalues 
       of the covariance matrix of A.
    """

    # Computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # Subtract the mean (along columns)
    latent, coeff = np.linalg.eig(np.cov(M))
    p = np.size(coeff, axis=1)
    idx = np.argsort(latent)  # Sorting the eigenvalues
    idx = idx[::-1]  # In ascending order
    # Sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:, idx]
    latent = latent[idx]  # Sorting eigenvalues
    if numpc < p or numpc >= 0:
        coeff = coeff[:, range(numpc)]  # Cutting some PCs
    score = np.dot(coeff.T, M)  # Projection of the data in the new space
    return coeff, score, latent


def quantile(a, prob=None, alphap=0.4, betap=0.4, **kwargs):
    if prob is None:
        prob = [0.25, 0.5, 0.75]
    elif isinstance(prob, (int, float, np.floating, np.integer)):
        prob = [(x + 1) / (prob + 1) for x in xrange(prob)]
    return scipy.stats.mstats.mquantiles(a, prob, alphap=alphap, betap=alphap, **kwargs)


def quantile_custom(a, prob, sorted=False):
    """ Y = quantile(X,N) returns quantiles for N evenly spaced cumulative 
        probabilities (1/(N + 1), 2/(N + 1), ..., N/(N + 1)) for integer N>1
    """
    if isinstance(a, np.ndarray):
        a = a if sorted else np.sort(a.flat)
    else:
        # Needed for fancy indexing a[1,2,3]
        a = np.array(a)
        if not sorted:
            a.sort()
    # edges = (1:N)/(N+1)
    edges = np.array([(x + 1) / (prob + 1) for x in xrange(prob)]) + 1e-10
    # Y = sortedX(round(length(X) * edges))
    indices = np.round(a.size * edges).astype(int) - 1
    return a[indices]


def sum_ndim(x, dims, preserve_axes=False):
    """ Summation on multiple axes
        sum(sum(sum(x, dim[0]), dim[1]), dim[2])
    """
    return np.apply_over_axes(np.sum, x, dims)


def sub2ind(shape, subscripts, order='C'):
    """ Converts subscript vector to linear index.
        Works also for an array of subscripts simultaneously.
    """
    try:
        return np.ravel_multi_index(subscripts.transpose(), shape, order=order)
    except ValueError as e:
        raise ValueError("%s\nshape=%s\nsubscripts=%s" % (e, shape, subscripts))


def ind2sub(shape, indices, order='C'):
    try:
        return np.unravel_index(indices, shape, order='C')
    except ValueError as e:
        raise ValueError("%s\nshape=%s\nindices=%s" % (e, shape, indices))


def discretize_discrete(iterable, insample=None):
    """ Schleife ueber alle Faktoren, die bereits diskret vorliegen:
        Bei Zahlen verwandle alle in Integer, die bei 0 beginnen und lueckenlos
        hochzaehlen (das ist der dritte Output von <unique>,
        z.B. [3.4, 1.2, 5.4, 1.2, 3.4, 3.4, 0.1] ==> [2, 1, 3, 1, 2, 2, 0] 
        analog fuer Strings: 'FHFCCH' ==> [1, 2, 1, 0, 0, 2]
    """
    assert len(iterable) > 0, "discretization data can not be empty"
    if insample is not None:
        iterable_insample = iterable[insample]
        assert len(iterable_insample) > 0, "discretization data can not be empty after applying insample range"
    else:
        iterable_insample = iterable

    if isinstance(iterable, np.ndarray):
        if iterable.dtype.name == 'object':
            uniques = sorted(set(i[0] for i in iterable_insample.flat))
        else:
            uniques = np.unique(iterable_insample)
    else:
        uniques = sorted(set(iterable_insample))

    tmp = dict((x, i) for i, x in enumerate(uniques))
    newvalue = max(tmp.itervalues()) + 1
    if isinstance(iterable, np.ndarray):
        if iterable.dtype.name == 'object':
            iterable_iter = (i[0] for i in iterable.flat)
        else:
            iterable_iter = iterable.flat

        ret = np.zeros_like(iterable, dtype=int)
        for i, x in enumerate(iterable_iter):
            if x in tmp:
                ret[i] = tmp[x]
            else:
                ret[i] = newvalue
        return ret
    else:
        return np.array([tmp[x] for x in iterable])


def discretize_continuous(iterable, levels, insample=None):
    """ Schleife ueber alle kontinuierlichen Faktoren:
        Unterteile jeden Faktor in <discretization_level> Quantile (also Bins, so
        dass in jedem Bin die gleiche Anzahl an Elementen sind)
        Der zweite Output von histc nimmt den Faktor und die Quantilgrenzen
        <edges>, und nummeriert die Bins einfach durch. Dann wird jedem Eintrag
        dis Faktors einfach seine Binnummer zugewiesen.
    """
    edges = quantile_custom(iterable if insample is None else iterable[insample], levels - 1).flat
    return np.searchsorted(edges, iterable, 'right')  # Kein inf benoetigt, wird bereits passend behandelt


def triangular_upper_idx(k, m):
    # Quite slow, don't use this in some tight loop
    # n = fac(m) / fac(k) / fac(m - k)
    return np.array(list(combinations(xrange(m), k)))


def unique_custom(ar, return_index=False, return_inverse=False, first=None):
    # TODO: Further speedups with np.unique1d

    if first is None:
        return np.unique(ar, return_index=return_index, return_inverse=return_inverse)

    c, ia, ic = np.unique(ar, return_index=True, return_inverse=True)
    if return_index:
        args = np.array(ar.flat).argsort()
        A_sort = ar.flat[args]
        group_sorts = np.lexsort((args, A_sort))
        if first:
            diff = np.ediff1d(A_sort, to_begin=[1])
        else:
            diff = np.ediff1d(A_sort, to_end=[1])
        ia = args[group_sorts[np.where(diff)]]

    if not return_index and not return_inverse:
        return c
    ret = [c]
    if return_index:
        ret.append(ia)
    if return_inverse:
        ret.append(ic)
    return tuple(ret)


def multiple_linear_regression(x, y, se=None, lmbd=0):
    """ 
    Result with following params: [b sb2 t p mse B C SSR]
    Input format: x.shape = (n,m)   y.shape = (n,) with n >> m
    lmbd: Tikhanov regularization parameter
    """

    out = AttrDict()
    n = len(y)

    if se is None:
        se = np.ones(y.shape)

    if len(x.shape) == 1:
        x0 = np.concatenate((np.ones((n, 1)), x.reshape((-1, 1))), axis=1)
    else:
        x0 = np.concatenate((np.ones((n, 1)), x), axis=1)
    k = x0.shape[1] - 1
    x1 = x0 / np.tile(se.reshape((-1, 1)), (1, k + 1))
    y1 = y / se

    out.n = n
    out.k = k
    # Get coefficients
    C = np.linalg.inv(x1.transpose().dot(x1) + lmbd * np.eye(k + 1))
    out.b = C.dot(x1.transpose()).dot(y1)

    # Mean squared error, don't wonder about the n - k - 1
    out.mse = np.sum((y1 - x1.dot(out.b)) ** 2) / (n - k - 1)

    # Variance estimate of the coefficients
    out.sb2 = out.mse * np.abs(C[xrange(0, len(C), k + 1)]).squeeze()

    # Bestimmtheitsmass. B = R^2, Quadrat des Korrelationskoeffizienten
    out.B = 1 - np.sum((y1 - x1.dot(out.b)) ** 2) / np.sum((y1 - np.mean(y1)) ** 2)
    out.SSR = np.sum((y1 - np.mean(y1)) ** 2) - np.sum((y1 - x1.dot(out.b)) ** 2)

    # t-statistics and p-value
    out.t = out.b / np.sqrt(out.sb2)
    out.p = (1 - scipy.stats.t.cdf(np.abs(out.t), n - k - 1)) * 2

    # Prediction
    out.pred = x0.dot(out.b)
    return out


def intersect(a, b):
    """
    MATLABs intersect function with additional ia and ib outputs: intsect, ia, ib = intersect(a, b)
    In [500]: a
    Out[500]: array([3, 8, 5, 4])
    
    In [501]: b
    Out[501]: array([4, 9, 7, 3, 5])
    
    In [502]: intersect1d(a,b)
    Out[502]: array([3, 4, 5])
    
    In [503]: ia
    Out[503]: array([0, 3, 2])
    
    In [504]: ib
    Out[504]: array([3, 0, 4])
    
    In [505]: a[ia]
    Out[505]: array([3, 4, 5])
    
    In [506]: b[ib]
    Out[506]: array([3, 4, 5])
    """
    inter = np.intersect1d(a, b)

    in_inter = np.in1d(a, inter)
    si = np.argsort(a[in_inter])
    ia = np.where(in_inter)[0][si]

    in_inter = np.in1d(b, inter)
    si = np.argsort(b[in_inter])
    ib = np.where(in_inter)[0][si]

    return inter, ia, ib


def randsample(values, probabilities, size):
    """Generate a random sample of size=size of values drawn from discrete probability distribution"""
    bins = np.add.accumulate(probabilities / np.sum(probabilities))
    return values[np.digitize(np.random.random_sample(size), bins)]


def create_combinations(av, names, disc_count=4):
    """Takes factors in string array <names> and enumerates every combination of discrete values.
    If factor is a float, then a quantile-wise discretization is performed first."""
    unique_count = np.zeros((len(names), len(av)), dtype=int)
    for i, n in enumerate(names):
        if (type(av[n][0]) == np.string_) or (type(av[n][0]) == np.int64):
            unique_count[i, :] = unique_custom(av[n], return_inverse=True)[1]
        elif type(av[n][0]) == np.float64:
            d = av[n][~np.isnan(av[n])]
            edges = quantile_custom(d, disc_count - 1)
            for j in xrange(len(edges) - 1):
                unique_count[i, (av[n] >= edges[j]) & (av[n] < edges[j + 1])] = j + 1
            unique_count[i, (av[n] >= edges[-1])] = len(edges)
    confounder = np.math.pi ** (1 + np.arange(len(names)))
    combs = unique_custom(np.dot(confounder, unique_count), return_inverse=True)[1]
    return combs, unique_count

