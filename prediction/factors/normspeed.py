import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.arrayview import ArrayView
from prediction.models.model_parameters import ModelParameters
from utils import YEAR
from prediction.tools.regressiontools import regression_data, modelcompare_test, fit_lme
from target import FactorTarget

class NormSpeed(FactorTarget):
    provided = ['norm_speed']
    required = ['speed', 'distance', 'wind_speed', 'temperature', 'sex', 'obstacle', 'going', 'course',
                'start_time', 'date_of_birth']

    def run(self, av, sel, verbose=False, *args, **kwargs):
        age = np.floor((av.start_time - av.date_of_birth) / YEAR)
        X, y, missing = regression_data(y=[av['speed']],
                                        Xnum=[age, av['distance'], av['wind_speed'], av['temperature']],
                                        Xcat=[av['sex'], av['obstacle'], av['going']])
        y = pd.DataFrame(y, columns=['speed'])
        fixed = pd.DataFrame(X[:, 0:6], columns=['age', 'distance', 'wind_speed', 'temperature', 'sex1', 'sex2'])
        random = pd.DataFrame({'obstacle': av['obstacle'], 'going': av['going']})
        data = pd.concat([y, fixed, random], axis=1)[~missing]
        train_data, test_data = train_test_split(data, test_size=0.3)

        models = {'mod1': {'formula': "speed ~ age + distance + wind_speed + temperature + sex1 + sex2",
                           'groups': 'obstacle', 're_formula': "~age"},
                  'mod2': {'formula': "speed ~ age + distance + wind_speed + temperature + sex1 + sex2",
                           'groups': 'going', 're_formula': "~age"},
                  'mod3': {'formula': "speed ~ age + distance + wind_speed + temperature + sex1 + sex2",
                           'groups': 'going', 're_formula': "~distance"},
                  'mod4': {'formula': "speed ~ age + distance + wind_speed + temperature + sex1 + sex2",
                           'groups': 'going', 're_formula': "~age+distance"}
                  }

        model_fits = fit_lme(models, train_data)
        model_rmse, model_codes = modelcompare_test(test_data, models=model_fits, measure='rmse')
        best_model = model_codes[np.argmin(model_rmse)]

        best_fit = fit_lme({'best_model': models[best_model]}, data)
        fitted = best_fit['best_model'].fittedvalues

        speed_estimate = np.zeros(len(av.speed))
        speed_estimate[~missing] = fitted
        speed_estimate[missing] = np.nan
        av.norm_speed[sel] = av.speed[sel] - speed_estimate[sel]

