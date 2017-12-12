import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from utils.arrayview import ArrayView
from prediction.models.model_parameters import ModelParameters
from prediction.models.parameters import factor_build_end
from prediction.tools.plotting import varinfo
from utils import timestamp, YEAR
from prediction.tools.regressiontools import regression_data, rmse
from prediction.factors.normspeed import NormSpeed

av = ArrayView.from_file('../datadev/brain_final2cut.av.bcolz')
av_w = ArrayView.from_file('../datadev/weather.av.bcolz')
pars = ModelParameters(av, oos_start=factor_build_end+YEAR, depth=3, lmbd=10, verbose=True)

df = pd.read_csv("../roll_normspeed.csv")
av.prediction = df.prediction.values.flatten()
av.residuals = df.residuals.values.flatten()
av_new = av['prediction', 'residuals']
av_new.dump('../datadev/roll_normspeed.av.bcolz')


# Extract variables
vars = {'speed': av.speed, 'distance': av.distance, 'sex': av.sex, 'obstacle': av.obstacle,
        'going': av.going, 'course': av.course, 'wind_speed': av_w.wind_speed, 'temp': av_w.temperature,
        'wind_direction': av_w.wind_direction, 'humidity': av_w.humidity, 'pressure': av_w.pressure,
        'condition': av_w.condition, 'start_time': av.start_time, 'norm_speed': av.norm_speed,
        'age': np.floor((av.start_time - av.date_of_birth) / YEAR), 'trainer': av['trainer'],
        'jockey': av['jockey'], 'runner': av['runner_id']}

'''After removing the missing values the first level of the going variable (AW) does not appear at all in the data so 
I will remove it before I start the analysis. Wind speed has negative values and I will remove them.'''
vars['going'][vars['going'] == 'AW'] = ''
vars['wind_speed'][vars['wind_speed'] < 0] = np.nan

# Prepare data for regression
X, y, missing = regression_data(y=[vars['speed']], Xnum=[vars['distance'], vars['norm_speed'], vars['trainer'],
                                                         vars['jockey'], vars['runner'], vars['start_time']],
                                Xcat=[vars['obstacle'], vars['going']])
df = pd.DataFrame({k: vars[k] for k in ('speed', 'distance',  'obstacle', 'going', 'norm_speed', 'trainer', 'jockey',
                                        'runner', 'start_time')})
df['build'] = ~missing & pars.build_mask
df['test'] = ~missing & (pars.is1 | pars.oos)
df_build, df_test = df[~missing & pars.build_mask], df[~missing & (pars.is1 | pars.oos)]

df.to_csv('../data.csv', index=False)
df_build.to_csv('../build.csv', index=False)
df_test.to_csv('../test.csv', index=False)

# Fit the full linear regression model
lm_full = sm.OLS(y_build, X_build).fit()
print(lm_full.summary())
#plt.scatter(range(len(lm_full.resid)), lm_full.resid)
#plt.show()

# Fit linear regression model without going
lm_wg = sm.OLS(y_build, X_build[:, 0:6]).fit()
print(lm_wg.summary())
#plt.scatter(range(len(lm_wg.resid)), lm_wg.resid)
#plt.show()

# Fit a regression model only with numeric variables
lm_num = sm.OLS(y_build, X_build[:, 0:2]).fit()
print(lm_num.summary())
#plt.scatter(range(len(lm_num.resid)), lm_num.resid)
#plt.show()

# Fit a lme model with obstacle as a random effect variable
lme_obst = smf.mixedlm("speed ~ distance + age", df_build, groups=df_build["obstacle"]).fit()
print(lme_obst.summary())
plt.scatter(range(len(lme_obst.resid)), lme_obst.resid)
plt.show()

# Fit a lme model with going as a random effect variable
lme_go = smf.mixedlm("speed ~ distance + age", df_build, groups=df_build["going"]).fit()
print(lme_go.summary())
plt.scatter(range(len(lme_go.resid)), lme_go.resid)
plt.show()

# Compare RMSE of prediction for the three models
rmse_full = rmse(y_test - lm_full.predict(X_test))
rmse_wg = rmse(y_test - lm_wg.predict(X_test[:, 0:6]))
rmse_num = rmse(y_test - lm_num.predict(X_test[:, 0:2]))
rmse_obst = rmse(y_test - lme_obst.predict(df_test))
rmse_go = rmse(y_test - lme_go.predict(df_test))
print """RMSE full model: %.3f
RMSE without going: %.3f
RMSE only numeric: %.3f
RMSE obstacle as re: %.3f
RMSE going as re: %.3f""" %(rmse_full, rmse_wg, rmse_num, rmse_obst, rmse_go)

'''We see that the model with going as a random intercept has the smallest RMSE. 
Next I fit the linear mixed effect models and add the weather variables. 
I will run a rolling forecast for several models with a step of 20 000 observations.
The python LME library has a limitation in that the model can have only one random effect variable.
In this setup it was not possible to use going or course as random effects because some of the levels appear rarely, 
leaving us with different levels in the smaller build and test samples. 
So the only random effect variable I will try is obstacle'''

# Prepare data for regression adding the weather variables
lme_data = pd.DataFrame({k: vars[k] for k in ('speed', 'distance', 'age', 'wind_speed', 'temp', 'wind_direction', 'obstacle')})
X_w, y_w, missing_w = regression_data(y=[vars['speed']],
                                      Xnum=[vars['distance'], vars['age'], vars['wind_speed'], vars['temp'], vars['wind_direction']],
                                      Xcat=[vars['sex'], vars['obstacle'], vars['going']])
lme_build = lme_data.ix[~missing_w & pars.build_mask, :]
X_w_build, y_w_build = X[~missing_w & pars.build_mask, :], y[~missing_w & pars.build_mask]


split = range(0, len(y_w_build), 20000)

# Run the models and save prediction RMSE
RMSE = np.zeros((len(split)-2, 6))
for i in range(len(split) - 2):
        y_b, x_b, lme_b = y_w_build[split[i]:split[i+1]], X_w_build[split[i]:split[i+1], :], lme_build[split[i]:split[i+1]]
        y_t, x_t, lme_t = y_w_build[split[i+1]:split[i+2]], X_w_build[split[i+1]:split[i+2], :], lme_build[split[i+1]:split[i+2]]
        lm1 = sm.OLS(y_b, x_b[:, 0:2]).fit()  # includes distance and age features
        lm2 = sm.OLS(y_b, x_b[:, 0:5]).fit()  # lm1 + the weather variables
        lm3 = sm.OLS(y_b, x_b).fit()  # full linear model
        """Random intercept"""
        lme1 = smf.mixedlm("speed ~ distance + age + wind_speed + temp + wind_direction",
                          lme_b, groups=lme_b["obstacle"]).fit()  # random intercept
        """Random intercept + slope"""
        lme2 = smf.mixedlm("speed ~ distance + age + wind_speed + temp + wind_direction",
                        lme_b, groups=lme_b["obstacle"], re_formula="~distance + age").fit()
        lme3 = smf.mixedlm("speed ~ distance + age + wind_speed + temp + wind_direction",
                          lme_b, groups=lme_b["obstacle"], re_formula="~age").fit()
        RMSE[i, 0] = rmse(y_t - lm1.predict(x_t[:, 0:2]))
        RMSE[i, 1] = rmse(y_t - lm2.predict(x_t[:, 0:5]))
        RMSE[i, 2] = rmse(y_t - lm3.predict(x_t))
        RMSE[i, 3] = rmse(y_t - lme1.predict(lme_t))
        RMSE[i, 4] = rmse(y_t - lme2.predict(lme_t))
        RMSE[i, 5] = rmse(y_t - lme3.predict(lme_t))
RMSE = pd.DataFrame(RMSE, columns=['lm1', 'lm2', 'lm3', 'lme1', 'lme2', 'lme3'])
plt.figure()
RMSE.plot()







