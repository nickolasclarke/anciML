# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import csv
import pandas as pd
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
# # Bid data (Aggregated Ancillary Service Offer Curve_thru_2016-2019)

# %%
#loading all data and concatenating
path = r'/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/Data--ERCOT/Aggregated Ancillary Service Offer Curve_thru_2016-2019'
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


# %%
df.shape


# %%
np.sum(df.duplicated(subset=['AncillaryType','DeliveryDate','HourEnding','Price','Quantity'],keep='first'))


# %%
df.drop_duplicates(subset=['AncillaryType','DeliveryDate','HourEnding','Price','Quantity'],
                   keep='first', inplace=True)


# %%
df.head()


# %%
df['hr_end'] = df['HourEnding'].apply(lambda x: int(x[:2]))
df['hr_beg'] = df['hr_end'] - 1
df.drop(columns=['hr_end'],inplace=True)


# %%
df['date'] = pd.to_datetime(df['DeliveryDate']).dt.date


# %%
df.head()


# %%
df = df[df['date']>datetime.date(2013,12,31)]


# %%
"""full_dt_hrbeg = []
for d, t in zip(df['date'], df['hr_end']):
    full_dt_hrbeg.append(datetime.datetime.combine(d,datetime.time(t-1)))"""


# %%
df.drop(columns=['DeliveryDate','HourEnding'], inplace=True)

# %% [markdown]
# https://towardsdatascience.com/pandas-groupby-aggregate-transform-filter-c95ba3444bbb is great

# %%
grouped = df.groupby(['AncillaryType','date','hr_beg'])#this is a good one


# %%
aggregation = {
    'Unweighted Average Price': pd.NamedAgg(column='Price', aggfunc='mean'),
    'Max Price': pd.NamedAgg(column='Price', aggfunc='max'),
    'Min Price': pd.NamedAgg(column='Price', aggfunc='min'),
    'Total Quantity': pd.NamedAgg(column='Quantity', aggfunc='sum'),
    'Number of Bids': pd.NamedAgg(column='Price', aggfunc='size')
}
grouped.agg(**aggregation)


# %%
#want weighted average price
def wavg(group, avg_name, weight_name):
    """ https://pbpython.com/weighted-average.html
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


# %%
x = pd.Series(grouped.apply(wavg, "Price", "Quantity"), name="Weighted Avg Price")


# %%
grouped_data = pd.concat([grouped.agg(**aggregation), x], axis=1)


# %%
grouped_data


# %%
#grouped_data.to_csv("as_bid_aggregated_data.csv")


# %%
products = df['AncillaryType'].unique()
output = grouped_data.loc[(products[0]),:]
output.columns = [products[0] + "_" + str(col) for col in output.columns]

for prod in products[1:]:
    x = grouped_data.loc[(prod),:]
    x.columns = [prod + "_" + str(col) for col in x.columns]
    output = pd.concat([output, x], axis=1)


# %%
output


# %%
output.to_csv("df_as_bid_aggregated_data.csv")


# %%
#ok now you need to check actual project requirements to see if you walk through your thought process in 
#a meaningful way or just need to guess your way to the outcome
#like we should test for correlation between things rigorously if we need to be able to justify
#our process

# %% [markdown]
# # DAM AS Plan_2016-2019

# %%
#loading all data and concatenating
path = r'/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/Data--ERCOT/DAM AS Plan_2016-2019'
all_files = glob.glob(path + "/*.csv")

dfs = []
bad_files = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)
        
df_plan = pd.concat(dfs, ignore_index=True)


# %%
#testing that the insane number of reported duplicates is real...
df_plan.loc[(df_plan['DeliveryDate']=='02/05/2016') &
           (df_plan['HourEnding']=='12:00'),:]


# %%
df_plan.drop_duplicates(subset=['DeliveryDate','HourEnding','AncillaryType','Quantity'], 
                        keep="first", inplace=True)


# %%
df_plan.reset_index(inplace=True, drop=True)


# %%
df_plan.head()


# %%
df_plan['hr_end'] = df_plan['HourEnding'].apply(lambda x: int(x[:2]))
df_plan['hr_beg'] = df_plan['hr_end'] - 1
df_plan.drop(columns=['hr_end'],inplace=True)


# %%
df_plan.drop(columns=['HourEnding','DSTFlag'],inplace=True)


# %%
df_plan.rename(columns={'DeliveryDate':'date'},inplace=True)


# %%
#df_plan.pivot_table(values=['Quantity'],index=['date','hr_beg'],columns=['AncillaryType'])


# %%
products = df_plan['AncillaryType'].unique()
output = df_plan.loc[df_plan['AncillaryType']==products[0],['date','hr_beg','Quantity']]
output.rename(columns={'Quantity':products[0]+"_"+'Quantity'}, inplace=True)

for prod in products[1:]:
    x = df_plan.loc[df_plan['AncillaryType']==prod, ['date','hr_beg','Quantity']]
    output = output.merge(x, how='outer', on=['date','hr_beg'])
    output.rename(columns={'Quantity':prod+"_"+'Quantity'}, inplace=True)


# %%
output.to_csv("df_as_plan.csv", index=False)


# %%


