import numpy as np
from manipulate import *

def regression_data(y=[], Xnum=[], Xnum_transform=[], Xcat=[], strata = None):
    """Remove rows with missing data, standardize, add indicator variables and return features as matrix"""
    n_samples = y[0].shape[0]
    missing = np.zeros(n_samples, dtype=bool)
    for v in y+Xnum+Xnum_transform+Xcat:
        missing = missing | find_missing(v)
    if len(Xnum_transform) > 0:
        Xnum_transform = [standardize(v, strata) for v in Xnum_transform]
    if len(Xcat) > 0:
        Xcat = [dummies(v) for v in Xcat]
        X = np.concatenate([cbind_1d(Xnum + Xnum_transform)] + Xcat, axis=1)
    else:
        X = cbind_1d(Xnum + Xnum_transform)
    y = y[0]
    print "\nNumber of observations: %d\nMissing: %d\nValid: %d" % (n_samples, np.sum(missing), np.sum(~missing))
    return X, y, missing

def dummies(x):
    """Replace a categorical variable by a set of indicator variables"""
    missing = find_missing(x)
    uvals = np.unique(x[~missing])
    indicators = np.zeros((len(x), len(uvals)-1))
    for j in range(indicators.shape[1]):
        indicators[x == uvals[j], j] = 1
    return indicators

# Function that returns Root Mean Squared Error
def rmse(error):
    return np.sqrt(np.mean(error**2))
