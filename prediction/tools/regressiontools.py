import numpy as np
import statsmodels.formula.api as smf
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

def eval_models(models, evalScheme, trainSize, k_CV, targetVar, cost):
    """
    Evaluate the performance of one or more models
    :param models:
    :param evalScheme:
    :param trainSize:
    :param k_CV:
    :param targetVar:
    :param cost:
    :return:
    """


def dummies(x):
    """Replace a categorical variable by a set of indicator variables"""
    missing = find_missing(x)
    uvals = np.unique(x[~missing])
    indicators = np.zeros((len(x), len(uvals)-1))
    for j in range(indicators.shape[1]):
        indicators[x == uvals[j], j] = 1
    return indicators

def rmse(error):
    """Function that returns Root Mean Squared Error"""
    return np.sqrt(np.mean(error**2))

def modelcompare_test(test_data, models, measure='rmse'):
    """Split data into train and test sets and evaluate """
    fits = models.keys()
    criterion = np.zeros(len(fits))
    for i in range(len(fits)):
        criterion[i] = eval(measure)(test_data['speed'] - models[fits[i]].predict(test_data))
    return criterion, fits

def fit_lme(models, data):
    """Fit mixed linear effect models"""
    model_fits = {}
    for m in models.keys():
        model = models[m]
        model_fits[m] = smf.mixedlm(model['formula'], data, groups=data[model['groups']],
                                    re_formula=model['re_formula']).fit()
    return model_fits



