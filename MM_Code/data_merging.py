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


# %%
path = '/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/FinalData/'

bids = pd.read_csv(path+'df_as_bid_aggregated_data.csv')
plans = pd.read_csv(path+'df_as_plan.csv')
energy_prices = pd.read_csv(path+'df_energy_price.csv') #this was an error in Nick's; was aggregated data
price_vol = pd.read_csv(path+'df_AS_price_vol.csv')
generation = pd.read_csv(path+'df_generation.csv')
weather = pd.read_csv(path+'df_weather_forecast_ercot.csv')

data = {'bids':bids,
        'plans':plans,
        'energy_prices':energy_prices,
        'price_vol':price_vol,
        'generation':generation,
        'weather':weather,
       }


# %%
#should note to nick i dropped duplicates in energy prices, and fixed the weather -1 thing


# %%
#Converting date columns to datetime format
for key in ['energy_prices','price_vol','generation']:
    data[key]['dt'] = pd.to_datetime(data[key]['date'] + " " + data[key]['hr_beg'])
    data[key].drop(columns=['date','hr_beg'], inplace=True)
    
for key in ['bids','plans','weather']:
    single_dig_hrs = (data[key]['hr_beg'] >= 0) & (data[key]['hr_beg'] < 10)
    data[key]['time'] = data[key]['hr_beg'].astype(str)
    data[key]['time'] = data[key]['time'] + ":00"
    data[key].loc[single_dig_hrs,'time'] = "0" + data[key].loc[single_dig_hrs,'time']
    
    data[key]['dt'] = pd.to_datetime(data[key]['date'] + " " + data[key]['time'])
    data[key].drop(columns=['date','hr_beg','time'], inplace=True)
    


# %%
#Getting starting and ending date to create daterange for merged df
mindate = []
maxdate = []
for key in data.keys():
    mindate.append(min(data[key]['dt']))
    maxdate.append(max(data[key]['dt']))
    
date_range = pd.date_range(min(mindate), max(maxdate),freq='H',)


# %%
#Merging dfs
df = pd.DataFrame(date_range, columns=["dt"]) #wanted to just make this an index, but wasn't working w merge

for key in data.keys():
    df = df.merge(data[key], how='left', on='dt', left_index=True)
    df.reset_index(inplace=True,drop=True)

df.set_index('dt')
df.head()


# %%
df.to_csv("merged_df.csv") #180 MB...worth saving? Pickle instead to keep datetime features?

