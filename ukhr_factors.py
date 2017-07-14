import pandas as pd
import numpy as np
from sources.ukhr import Ukhr
from utils import timestamp, YEAR
from utils.arrayview import ArrayView
from prediction.tools.helpers import spread_from_av
import sys
from prediction.tools.plotting import varinfo


ukhr_data = Ukhr(basepath="../")
ukhr_data.update_archive()
ukhr_data.match("../run_id_match.csv")


def match_ukhr():
    runs = pd.read_csv("../run_id_match_V1.csv")
    dates = runs.date.unique()
    for d in dates[1:]:
        print(d)
        year = d.split('-')[0]
        fpath = "../ukhr/" + year + "/ukhr_daily_" + d + ".csv"
        ukhr_daily = pd.read_csv(fpath)
        ukhr_daily['run_id'] = np.nan
        cols = [key.strip().replace(' ', '').replace('&', '') for key in list(ukhr_daily.columns)]
        ukhr_daily.columns = cols
        daily_runs = runs[runs.date == d]
        for i in range(ukhr_daily.shape[0]):
            is_in = daily_runs.horse_name.isin([ukhr_daily.Horse[i]])
            if (np.any(is_in)):
                runid = daily_runs.Run_ID[daily_runs.horse_name == ukhr_daily.Horse[i]].values
                if len(runid) > 1:
                    start_times = daily_runs.Start_time[daily_runs.horse_name == ukhr_daily.Horse[i]].values
                    uk_st = float(timestamp(d + ' ' + ukhr_daily['Time24Hour'][i]))
                    runid = runid[start_times == uk_st]
                if len(runid) == 1:
                    ukhr_daily.set_value(i, 'run_id', runid)
        if d == '2013-01-01':
            ukhr_data = ukhr_daily
        else:
            ukhr_data = ukhr_data.append(ukhr_daily)
    ukhr_data.to_csv("../ukhr_data.csv")

match_ukhr()


ukhr_data = pd.read_csv("../ukhr_data.csv")
av = ArrayView.from_file('../datadev/brain_final2cut.av.bcolz')

df_av = pd.DataFrame({'run_id': av.run_id})
ukhr_data = df_av.merge(ukhr_data, on='run_id', how='left', copy=False)

cols = list(ukhr_data.columns)
for col in cols:
    col_save = col
    if '/' in col:
        col_save = '_'.join(col.split('/'))
    ukhr_data[col].to_csv("../ukhr_avs/" + col_save + ".csv", index=False)







