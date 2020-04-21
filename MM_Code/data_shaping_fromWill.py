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


# %%
path = '/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/data_dump_will/'

# %% [markdown]
# # AS_price
# %% [markdown]
# EDA observations:
# * ISO: ERCOT
# * Markets = DAH, RTH
# * Date range = 1/1/14 to 1/1/19
# * Price type = Down Regulation, Non-Spinning Reserve, Responsive Reserve, Up Regulation
# * There's only one price node ID
# * KEEP: Market, Price Type (sep tables?), Date, Time, Price, Volume

# %%
df_as = pd.read_csv(path+'AS_price 2.csv') #no longer need first AS file


# %%
df_as.head(1)


# %%
df_as['dt'] = pd.to_datetime(df_as['Local Datetime (Hour Beginning)'])


# %%
df_as['date'] = df_as['dt'].dt.date
df_as['time_beginning'] = df_as['dt'].dt.time


# %%
type(df_as['dt'][0])


# %%
type(df_as['date'][0])


# %%
df_as = df_as[['Market','Price Type','date','time_beginning','Price $/MWh','Volume MWh']]
df_as.columns = ['market','product','date','hr_beg','price','volume']


# %%
#Separating markets and products
as_dict = {}
for mark in df_as['market'].unique():
    for prod in df_as['product'].unique():
        as_dict[mark+prod] = df_as.loc[(df_as['market']==mark) 
                                       & (df_as['product']==prod),'date':].reset_index(drop=True)
        as_dict[mark+prod].sort_values(by=['date','hr_beg']).reset_index(drop=True)


# %%
df_as.head()


# %%
df_as = df_as.drop_duplicates(keep='first')


# %%
as_output = df_as.loc[:,'date':'hr_beg']


# %%
products = ['Down Regulation', 'Non-Spinning Reserve', 'Responsive Reserve',
       'Up Regulation']
new_products = ['REGDN','NSPIN','RRS','REGUP']

as_output = df_as.loc[:,'date':'hr_beg']

for market in ['DAH']:
    for i, prod in enumerate(products[0:2]):
        subset = df_as.loc[(df_as['market']==market) & (df_as['product']==prod),['date','hr_beg','price','volume']].rename(columns={'price':'price'+"_"+market+"_"+new_products[i],
                                                                                                                    'volume':'vol'+"_"+market+"_"+new_products[i]})
        as_output = as_output.merge(subset, how="outer", on=['date','hr_beg'])


# %%
#this didn't work, so i resorted to the hack below
as_output


# %%
#as_output.to_csv("df_AS_price_vol.csv", index=False)


# %%
market = 'DAH'
prod = 'Down Regulation'
x = df_as.loc[(df_as['market']==market) & 
          (df_as['product']==prod),['date','hr_beg','price','volume']].rename(columns={'price':'price'+"_"+market+"_"+prod,
                                                                      'volume':'vol'+"_"+market+"_"+prod})


# %%
prod = 'Up Regulation'
y = df_as.loc[(df_as['market']==market) & 
          (df_as['product']==prod),['date','hr_beg','price','volume']].rename(columns={'price':'price'+"_"+market+"_"+prod,
                                                                      'volume':'vol'+"_"+market+"_"+prod})
x = x.merge(y, how="outer", on=['date','hr_beg'])


# %%
prod = 'Non-Spinning Reserve'
y = df_as.loc[(df_as['market']==market) & 
          (df_as['product']==prod),['date','hr_beg','price','volume']].rename(columns={'price':'price'+"_"+market+"_"+prod,
                                                                      'volume':'vol'+"_"+market+"_"+prod})
x = x.merge(y, how="outer", on=['date','hr_beg'])


# %%
prod = 'Responsive Reserve'
y = df_as.loc[(df_as['market']==market) & 
          (df_as['product']==prod),['date','hr_beg','price','volume']].rename(columns={'price':'price'+"_"+market+"_"+prod,
                                                                      'volume':'vol'+"_"+market+"_"+prod})
x = x.merge(y, how="outer", on=['date','hr_beg'])


# %%
x


# %%
x.to_csv("df_AS_price_vol.csv", index=False)


# %%



# %%



# %%



# %%


# %% [markdown]
# ### Checking out autocorrelation and cross-correlation between products

# %%
#Autocorrelation
outputs = pd.DataFrame(index=[0,1,2,3,4,5])
for as_product in as_dict.keys():
    timesteps = {}
    timesteps["t0"] = as_dict[as_product]['price'][5:].reset_index(drop=True)
    timesteps["t1"] = as_dict[as_product]['price'][4:-1].reset_index(drop=True)
    timesteps["t2"] = as_dict[as_product]['price'][3:-2].reset_index(drop=True)
    timesteps["t3"] = as_dict[as_product]['price'][2:-3].reset_index(drop=True)
    timesteps["t4"] = as_dict[as_product]['price'][1:-4].reset_index(drop=True)
    timesteps["t5"] = as_dict[as_product]['price'][0:-5].reset_index(drop=True)
    
    for i, step in enumerate(timesteps.keys()):
        outputs.loc[i,as_product] = np.corrcoef(timesteps['t0'], timesteps[step])[0,1]

# %% [markdown]
# Regulation price not very correlated from timestep to timestep, whereas reserves (more predictably) are more correlated....50% correlation means if we can do better than a coinflip? or no, that's 0 correl?

# %%
outputs

# %% [markdown]
# *Now checking out correlation of price and volume in DAH*
# Wow, interesting--almost no correlation of price and volume in DAH (except for DAHdown). I wonder if there's autocorrelation of volume...(for DAHdown)
# 
# I should also remember this is just from hour to hour, and has nothing to do with the same hour on subsequent days. That next.

# %%
for product in as_dict.keys():
    print(product + ": " + str(np.corrcoef(as_dict[product]['price'], as_dict[product]['volume'])[0,1]))


# %%
#Autocorrelation among volumes
vol_outputs = pd.DataFrame(index=[0,1,2,3,4,5])
for as_product in as_dict.keys():
    timesteps = {}
    timesteps["t0"] = as_dict[as_product]['volume'][5:].reset_index(drop=True)
    timesteps["t1"] = as_dict[as_product]['volume'][4:-1].reset_index(drop=True)
    timesteps["t2"] = as_dict[as_product]['volume'][3:-2].reset_index(drop=True)
    timesteps["t3"] = as_dict[as_product]['volume'][2:-3].reset_index(drop=True)
    timesteps["t4"] = as_dict[as_product]['volume'][1:-4].reset_index(drop=True)
    timesteps["t5"] = as_dict[as_product]['volume'][0:-5].reset_index(drop=True)
    
    for i, step in enumerate(timesteps.keys()):
        vol_outputs.loc[i,as_product] = np.corrcoef(timesteps['t0'], timesteps[step])[0,1]


# %%
vol_outputs
#Highly correlated from one hour to the next...wonder if this could be leveraged to 
#better predict DAH down, at least?

# %% [markdown]
# #OK--now trying to see correlation of given hour on different days (at some point also have to account for seasonality, of course!)

# %%
ddown = as_dict['DAHDown Regulation']
ddown.head()


# %%
#Same hour on each subsequent day
hourly_price = {}

for hour in ddown['hr_beg'].unique():
    hourly_price[hour] = ddown.loc[ddown['hr_beg']==hour,'price'].reset_index(drop=True)


# %%
day2day_output = pd.DataFrame()

for hour in hourly_price.keys():
    timesteps = {}
    timesteps["t0"] = hourly_price[hour][5:].reset_index(drop=True)
    timesteps["t1"] = hourly_price[hour][4:-1].reset_index(drop=True)
    timesteps["t2"] = hourly_price[hour][3:-2].reset_index(drop=True)
    timesteps["t3"] = hourly_price[hour][2:-3].reset_index(drop=True)
    timesteps["t4"] = hourly_price[hour][1:-4].reset_index(drop=True)
    timesteps["t5"] = hourly_price[hour][0:-5].reset_index(drop=True)
    
    for i, step in enumerate(timesteps.keys()):
        day2day_output.loc[hour,i] = np.corrcoef(timesteps['t0'], timesteps[step])[0,1]


# %%
day2day_output ##oy vey...not incredibly good here either...i maen, i guess; 
#should probably look at that book. this is for regulation down

# %% [markdown]
# Now trying for regulation up

# %%
dup = as_dict['DAHUp Regulation']


# %%
#Same hour on each subsequent day
hourly_price = {}

for hour in dup['hr_beg'].unique():
    hourly_price[hour] = dup.loc[dup['hr_beg']==hour,'price'].reset_index(drop=True)
    
day2day_output_up = pd.DataFrame()

for hour in hourly_price.keys():
    timesteps = {}
    timesteps["t0"] = hourly_price[hour][5:].reset_index(drop=True)
    timesteps["t1"] = hourly_price[hour][4:-1].reset_index(drop=True)
    timesteps["t2"] = hourly_price[hour][3:-2].reset_index(drop=True)
    timesteps["t3"] = hourly_price[hour][2:-3].reset_index(drop=True)
    timesteps["t4"] = hourly_price[hour][1:-4].reset_index(drop=True)
    timesteps["t5"] = hourly_price[hour][0:-5].reset_index(drop=True)
    
    for i, step in enumerate(timesteps.keys()):
        day2day_output_up.loc[hour,i] = np.corrcoef(timesteps['t0'], timesteps[step])[0,1]


# %%
day2day_output_up #fascinating...6AM has no correlation, but on peak in afternoon
#(at least from 12-5) is pretty good

# %% [markdown]
# I should obviously visualize the data...

# %%
plt.plot(dup['price'][1000:1240]) #so, obvious pattern of some sort, but not daily...morn/eve peaks?
#honestly this looks extremely predictable, we just have to figure out when those spikes are
#may be that we can predict some hours well and some hours not well...[but obvi we care about price spikes most]


# %%
plt.plot(ddown['price'][1000:1240])


# %%
plt.plot(dup['price'][0:240])
plt.plot(ddown['price'][0:240]) #inversely varying? yes, duh

# %% [markdown]
# Why are there a million rows in DAH and only like 1/40 as many in RTH?

# %%
plt.hist(as_dict['DAHDown Regulation']['date'])


# %%
plt.hist(as_dict['RTHDown Regulation']['hr_beg']) #so weird!! just missing a bunch of days/hours???
#or maybe just only rebid sometimes in real time? this makes the most sense

# %% [markdown]
# # energy_prices

# %%
df_energy = pd.read_csv(path+'energy_price.csv')


# %%
df_energy.head()


# %%
df_energy['dt'] = pd.to_datetime(df_energy['Local Datetime (Hour Ending)'])


# %%
df_energy['dt_beg'] = df_energy['dt'].apply(lambda x: x - datetime.timedelta(hours=1))


# %%
df_energy['date'] = df_energy['dt_beg'].dt.date
df_energy['hr_beg'] = df_energy['dt_beg'].dt.time


# %%
df_energy = df_energy[['Price Node Name','Price Type','Market','date','hr_beg','Price $/MWh']].reset_index(drop=True)
df_energy.columns = ['node','price_type','market','date','hr_beg','price']


# %%
df_energy.head()


# %%
plt.hist(df_energy['date'])
plt.show()


# %%
for node in df_energy['node'].unique():
    df_energy[df_energy['node']==node].to_csv("energy_prices_"+node+".csv", index=False)


# %%
nodes = ['HB_BUSAVG', 'HB_HOUSTON', 'HB_HUBAVG', 'HB_NORTH', 'HB_SOUTH',
       'HB_WEST']
newnodes = ['busavg','houston','hubavg','N','S','W']


# %%
markets = ['DAH', 'RT15AVG']
newmarkets = ['DAH','RT15']


# %%
df_energy.head()


# %%
energy_output = df_energy.loc[:,'date':'hr_beg']

for i, market in enumerate(markets):
    for j, node in enumerate(nodes):
        subset = df_energy.loc[(df_energy['market']==market) & (df_energy['node']==node),['date','hr_beg','price']].rename(columns={'price':'price'+"_"+newmarkets[i]+"_"+newnodes[j],
                                                                                                                    })
        energy_output = energy_output.merge(subset, on=['date','hr_beg'], how="outer")


# %%
energy_output.to_csv("df_energy_price.csv", index=False)


# %%
energy_output

# %% [markdown]
# # Weather
# %% [markdown]
# New data from near Houston, Dallas, and San Antonio

# %%
#loading all data and concatenating
path = r'/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/data_dump_will/weather'
all_files = glob.glob(path + "/*.csv")

dfs = []
bad_files = []

for file in all_files:
    try:
        x = pd.read_csv(file)
        name = file.replace(path,"")
        name = name.replace('.csv',"")
        x['Name'] = name
        dfs.append(x)
    except pd.errors.EmptyDataError:
        bad_files.append(file)
        
df = pd.concat(dfs, ignore_index=True)


# %%
df['Name'].unique()


# %%
df['coord1'] = df['Name'].apply(lambda s: s[1:s.find('_-')])
df['coord2'] = df['Name'].apply(lambda s: s[8:s.find('_2')])


# %%
df['Name'] = df['Name'].apply(lambda s: s[:-5])


# %%
df.head(50)


# %%
realnames = {'/29.48_-98.3': 'AustinSA',
             '/32.82_-97.05': 'Dallas',
             '/30.96_-103.35': 'WestTX',
            '/29.63_-96.08': 'Houston'}


# %%
#this is stupid
for key in realnames.keys():
    df.loc[df['Name']==key,'Name'] = realnames[key]


# %%
df.sort_values(by=['Year','Month','Day','Hour','Minute','coord1','coord2'], inplace=True)


# %%
df.reset_index(inplace=True, drop=True)


# %%
df.head()


# %%
df.to_csv("weather_data.csv")

# %% [markdown]
# # Generator data

# %%
path = '/Users/margaretmccall/Documents/2020 Spring/CE 295/0 - Final Project/data_dump_will/'
df_gen = pd.read_csv(path+'ERCOT_hourly_by_BA_v5.csv')


# %%
df_gen.head()

# %% [markdown]
# The datetimes are weird--local time and UTC seem to be the same? datetime column seems most likely to be central time

# %%
df_gen['dt1'] = pd.to_datetime(df_gen['local_time_cems'])
df_gen['dt2'] = pd.to_datetime(df_gen['utc'])
df_gen['dt3'] = pd.to_datetime(df_gen['datetime'])


# %%
df_gen.columns


# %%
plt.plot(df_gen.loc[:24,'dt2'], df_gen.loc[:24,'load_naked'], label='starts at 8am')


# %%
plt.plot(df_gen.loc[:24,'dt3'], df_gen.loc[:24,'load_naked'], label='starts at 2am') #this is more reasonable


# %%
df_gen['date'] = df_gen['dt3'].dt.date
df_gen['hr_beg'] = df_gen['dt3'].dt.time


# %%
df_gen.head()


# %%
df_gen.columns


# %%
df_gen.drop(columns=['dt1','dt2','dt3','hour'], inplace=True)


# %%
df_gen.drop(columns=['local_time_cems','utc','datetime'], inplace=True)


# %%
df_gen['year'] = df_gen['date'].apply(lambda x: x.year)


# %%
df_gen = df_gen[df_gen['year'] > 2013]


# %%
df_gen.drop(columns=['year'], inplace=True)


# %%
df_gen.to_csv('df_generation.csv', index=False)


# %%


