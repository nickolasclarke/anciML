# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import csv
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
from shapely.affinity import scale
import matplotlib.pyplot as plt

import glob
import os
import datetime

# %% [markdown]
# # Weather_Assumptions_thru_2016-2019

# %%
#loading all data and concatenating
path = r'/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/Data--ERCOT/Weather_Assumptions_thru_2016-2019'
all_files = glob.glob(path + "/*.csv")

dfs = []
bad_files = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)
        
df = pd.concat(dfs, ignore_index=True)


# %%
df.head()
#TODO: HOUR_ENDING RUNS FROM 0:23 SO I CANT JUST SUBTRACT


# %%
df['hr_end'] = df['HourEnding'].apply(lambda x: int(x[:2]))
df['hr_beg'] = df['hr_end'] - 1
df.drop(columns=['hr_end'],inplace=True)


# %%
max(df['HourEnding'])


# %%
dupes = df[df['DSTFlag']=='Y'] #i now remember from the old days of parsing ERCOT data
#that there's always some weird thing at 3pm with duplicate data


# %%
dupe_avgs = dupes.groupby(by=['DeliveryDate','HourEnding']).mean()


# %%
dupe_avgs


# %%
df.drop_duplicates(subset=['DeliveryDate','HourEnding'], keep="first", inplace=True)


# %%
df.reset_index(inplace=True)


# %%
df.rename(columns={'DeliveryDate': 'date'}, inplace=True)


# %%
df.drop(columns=['index','HourEnding'], inplace=True)


# %%
df.drop(columns=['DSTFlag'], inplace=True)


# %%
df.to_csv('weather_forecast_ercot.csv', index=False)

# %% [markdown]
# # WPP_Hrly_AVG_Actual_Forecasted_2016-2018

# %%
#loading all data and concatenating
path = r'/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/Data--ERCOT/WPP_Hrly_AVG_Actual_Forecasted_2016-2018'
all_files = glob.glob(path + "/*.csv")

dfs = []
bad_files = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)
        
df_wind = pd.concat(dfs, ignore_index=True)


# %%
df_wind.head()


# %%
df_wind.columns

# %% [markdown]
# Seems like we actually care about actual and STWPF (short-term wind power forecast), less WGRPP (wind gen resource power potential) or COP HSL (current operating plan high sustainable limit) http://www.ercot.com/glossary/h

# %%
df_wind_sub = df_wind[['DELIVERY_DATE','HOUR_BEGINNING','HOUR_ENDING','ACTUAL_LZ_NORTH', 'ACTUAL_LZ_SOUTH_HOUSTON', 'ACTUAL_LZ_WEST',
       'ACTUAL_SOUTH_HOUSTON', 'ACTUAL_SYSTEM_WIDE', 'ACTUAL_WEST_NORTH','STWPF_LZ_NORTH', 'STWPF_LZ_SOUTH_HOUSTON', 'STWPF_LZ_WEST',
       'STWPF_SOUTH_HOUSTON', 'STWPF_SYSTEM_WIDE', 'STWPF_WEST_NORTH']]


# %%
df_wind_sub.shape


# %%
df_wind_sub.head()


# %%
df_wind_sub['hr_beg'] = df_wind_sub['HOUR_ENDING'] - 1


# %%
df_wind_sub.rename(columns={'DELIVERY_DATE':'date'}, inplace=True)


# %%
df_wind_sub['date'] = pd.to_datetime(df_wind_sub['date']).dt.date


# %%
df_wind_sub.drop(columns=['HOUR_BEGINNING','HOUR_ENDING'], inplace=True)


# %%
df_wind_sub.drop_duplicates(keep='first',inplace=True)


# %%
df_wind_sub.to_csv("df_wind.csv", index=False)

# %% [markdown]
# why is this dataframe SO LONG...oh--I bet it's because I assumed their headers were all the same, but they weren't...

# %%
hr = df_wind_sub['hr_beg']
df_wind_sub.drop(labels=['hr_beg'], axis=1,inplace = True)
df_wind_sub.insert(0, 'hr_beg', hr)
df_wind_sub.head()


# %%
df_wind_sub[df_wind_sub.duplicated(subset=['date','hr_beg'],keep='first')]


# %%


