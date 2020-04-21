# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import csv
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
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
# # Loading dataframes

# %%
df_as_bids = pd.read_csv('FinalData/df_as_bid_aggregated_data.csv')
df_as_plan = pd.read_csv('FinalData/df_as_plan.csv')
df_as_prices = pd.read_csv('FinalData/df_AS_price_vol.csv')
df_energy_prices = pd.read_csv('FinalData/df_energy_price.csv')
df_gen = pd.read_csv('FinalData/df_generation.csv')
df_weather = pd.read_csv('FinalData/df_weather_forecast_ercot.csv')


# %%
df_dict = {}
dfs = [df_as_bids, df_as_plan, df_as_prices, df_energy_prices, df_gen, df_weather]
names = ["df_as_bids", "df_as_plan", "df_as_prices", "df_energy_prices", "df_gen", "df_weather"]
for i,df in enumerate(dfs):
    df_dict[names[i]] = df


# %%
for key in df_dict.keys():
    df_dict[key]['date'] = pd.to_datetime(df_dict[key]['date']).dt.date


# %%
for key in df_dict.keys():
    df_dict[key] = df_dict[key].sort_values(by=['date','hr_beg']).reset_index(drop=True)

# %% [markdown]
# # Intro plotting
# * Just a week of data, to get used to what it looks like: 5/1/17 - 5/8/17
# * Just regulation (up and down)

# %%
df_dict['df_as_bids'].head(1)


# %%
df_dict['df_as_plan'].head(1)


# %%
df_dict['df_as_prices'].head(1)

# %% [markdown]
# ## Plotting prices, part 1

# %%
fig = plt.figure(figsize = (15,10))
plt.subplot(311)
for year in [2014,2015,2016,2017,2018]:
    plot_prices = df_dict['df_as_prices'].loc[(df_dict['df_as_prices']['date']>=datetime.date(year,1,1)) &
                                     (df_dict['df_as_prices']['date']<=datetime.date(year,12,31)),:]
    plt.plot(plot_prices['date'], plot_prices.iloc[:,2], label="year")
    plt.title("Reg Down Prices")

plt.subplot(312)
for year in [2014,2015,2016,2017,2018]:
    plot_prices = df_dict['df_as_prices'].loc[(df_dict['df_as_prices']['date']>=datetime.date(year,1,1)) &
                                     (df_dict['df_as_prices']['date']<=datetime.date(year,12,31)),:]
    plt.plot(plot_prices['date'], plot_prices.iloc[:,4], label="year")
    plt.title("Reg Up Prices")
    plt.ylim([0,150])
    
plt.subplot(313)
for year in [2014,2015,2016,2017,2018]:
    plot_prices = df_dict['df_as_prices'].loc[(df_dict['df_as_prices']['date']>=datetime.date(year,1,1)) &
                                     (df_dict['df_as_prices']['date']<=datetime.date(year,12,31)),:]
    plt.plot(plot_prices['date'], np.log(plot_prices.iloc[:,4]), label="year")
    plt.title("Reg Up Prices (ln)")

# %% [markdown]
# ## Plotting bids

# %%
plot_prices = df_dict['df_as_prices'].loc[(df_dict['df_as_prices']['date']>=datetime.date(2017,5,1)) &
                                     (df_dict['df_as_prices']['date']<datetime.date(2017,5,8)),:]
plot_prices = plot_prices.loc[:,plot_prices.columns[np.r_[0,1,2,4]]].reset_index(drop=True)


# %%
plot_bids = df_dict['df_as_bids'].loc[(df_dict['df_as_bids']['date']>=datetime.date(2017,5,1)) &
                                     (df_dict['df_as_bids']['date']<datetime.date(2017,5,8)),:]


# %%
plot_bids_idx = plot_bids.loc[:,['date','hr_beg']]


# %%
plot_bids = plot_bids.loc[:,plot_bids.columns[14:26]]


# %%
plot_bids = pd.concat([plot_bids_idx, plot_bids], axis=1).reset_index(drop=True)


# %%
plot_bids['REGDN_Total Quantity'] = plot_bids['REGDN_Total Quantity']/10000
plot_bids['REGUP_Total Quantity'] = plot_bids['REGUP_Total Quantity']/10000


# %%
plot_bids['REGUP_Max Price'] = plot_bids['REGUP_Max Price']/1000


# %%
len(plot_bids)


# %%
fig = plt.figure(figsize = (15,10))

plt.subplot(211)
for col in plot_bids.columns[3:8]:
    plt.plot(plot_bids[col], label=col)
plt.plot(plot_prices['price_DAH_Down Regulation'], marker='o', label='DAH ACTUAL PRICE')
plt.legend()

plt.subplot(212)
for col in plot_bids.columns[9:14]:
    plt.plot(plot_bids[col], label=col)
plt.plot(plot_prices['price_DAH_Up Regulation'], label='DAH ACTUAL PRICE')
plt.legend()

# %% [markdown]
# * read up on how to de-trend
# * still need to check for seasonal component (ie, longer-term)
# * make persistence model
# * make avg model
# * look at stats model
# %% [markdown]
# Notes:
# * t is the x-variable, you dummy! and y is the regup. so your sin function has t in it
# * we care about harmonic regression--sum of sine waves (p. 26). maybe just one period, or maybe 2 for reg_dn (regup less clear)
# * to remove trend, may sometimes need to apply prelim transformation to data
# * autocovariance vs autocorrelation (p. 28)...calculating these functions seems important, given how striking the wine thing is, but a little confused... *but doing it for a lag of 40 shows the cyclical nature*--so do this (more lag)
# * is there a reason to take a log in our data?
# * exponential smoothing (moving average of past values) often used for forecasting--** try a moving average (if you can figure out the point--as proxy for fxn?) but is there one that will let a sine wave pass through?
# * or differencing? not sure how we'd forecast with it, but...
# * (but let's keep in mind that forecasting is literally just applying OLS! this is same as duncan's ;) )
# * plot out your residuals when you do anything!
# * they deal with seasonality by saying "transform--do classical fit"?! maybe statsmodels has something like that?!
# * too bad that down reg is easier to predict but up reg is where the money is
# %% [markdown]
# ## Plotting prices: persistence model (#1)
# https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/

# %%
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# %%
prices = df_dict['df_as_prices'].loc[:, df_dict['df_as_prices'].columns[np.array([0,1,2,4])]]


# %%
prices.rename(columns={'price_DAH_Down Regulation':'price_down',
                      'price_DAH_Up Regulation':'price_up'}, inplace=True)


# %%
def persistence_model(x):
    return x

# %% [markdown]
# Across whole dataset (5 years), MSE for down regulation is 83.61.

# %%
Y = prices['price_down'][1:]
X = prices['price_down'][:-1]

test_size = 1
test_cutoff = int((1-test_size)*len(Y))
Y_train = Y.iloc[:test_cutoff]
X_train = X.iloc[:test_cutoff]
Y_test = Y.iloc[test_cutoff:]
X_test = X.iloc[test_cutoff:]

assert len(Y_train) == len(X_train)
assert len(Y_test) == len(X_test)

Y_hat = []
for X in X_test:
    Y_hat.append(persistence_model(X))
    
print(mean_squared_error(Y_test, Y_hat))
fig = plt.figure(figsize = (15,3))
plt.plot(Y_hat)
plt.plot(Y_test)
plt.show() #graphing is pointless on this timescale

# %% [markdown]
# Trying a year at a time. Results are pretty interesting: 2014 was much more volatile than 2018 (each subsequent year is actually less volatile).

# %%
prices.head(1)


# %%
for year in [2014,2015,2016,2017,2018]:
    price_sub = prices.loc[(prices['date']>=datetime.date(year,1,1)) &
                           (prices['date']<=datetime.date(year,12,31)),'price_down']

    Y = price_sub[1:]
    X = price_sub[:-1]

    test_size = 1
    test_cutoff = int((1-test_size)*len(Y))
    Y_train = Y.iloc[:test_cutoff]
    X_train = X.iloc[:test_cutoff]
    Y_test = Y.iloc[test_cutoff:]
    X_test = X.iloc[test_cutoff:]

    assert len(Y_train) == len(X_train)
    assert len(Y_test) == len(X_test)

    Y_hat = []
    for X in X_test:
        Y_hat.append(persistence_model(X))

    print("MSE in "+str(year)+": "+str(mean_squared_error(Y_test, Y_hat)))

# %% [markdown]
# OK, finally, a month at a time! Seems like we should be referencing a persistence model specific to whatever timeframe we're trying to predict over.
# 
# Results are also interesting, maybe helpful. Most interesting is that MSEs for summer months (I'm too lazy to graph this) are quite low compared to winter and spring--persistence model works well.

# %%
for year in [2014,2015,2016,2017,2018]:
    for month,day in zip(np.arange(1,13),[31,28,31,30,31,30,31,31,30,31,30,31]):
        price_sub = prices.loc[(prices['date']>=datetime.date(year,month,1)) &
                               (prices['date']<=datetime.date(year,month,day)),'price_down']

        Y = price_sub[1:]
        X = price_sub[:-1]

        test_size = 1
        test_cutoff = int((1-test_size)*len(Y))
        Y_train = Y.iloc[:test_cutoff]
        X_train = X.iloc[:test_cutoff]
        Y_test = Y.iloc[test_cutoff:]
        X_test = X.iloc[test_cutoff:]

        assert len(Y_train) == len(X_train)
        assert len(Y_test) == len(X_test)

        Y_hat = []
        for X in X_test:
            Y_hat.append(persistence_model(X))

        print("MSE in "+str(year)+", month "+str(month)+": "+str(mean_squared_error(Y_test, Y_hat)))

# %% [markdown]
# ## Trying to apply a harmonic function (#2) to deal w/ periodicity
# %% [markdown]
# First I'm going to try to figure out the period of the down-reg seasonal oscillation...by applying a moving average? Sure, why not.
# 
# https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788

# %%
import statsmodels

# %% [markdown]
# Actually, let's come back to that. Let's do a daily thing first, instead.

# %%
price_sub = prices.loc[(prices['date']>=datetime.date(2016,4,1)) &
                               (prices['date']<=datetime.date(2016,4,10)),:'price_down'].reset_index(drop=True)


# %%
plt.plot(price_sub['hr_beg'],price_sub['price_down'], linestyle='none', marker='o')

# %% [markdown]
# It's surprisingly hard to tell what the period is of this data--a day? Less? It seems to be a day, ish, with a peak every night (around hour 0/24). Maybe I'll just try to fit a sine curve, and then move on to some smoothing with statsmodels.

# %%
t = price_sub.index.to_list()
plt.plot(price_sub['price_down'])
plt.plot(10*np.sin(np.divide(t,3)+5)+10)

# %% [markdown]
# We're getting there. this is pure guesswork, though. is there a way to fit a sine curve without so much guessing? I guess figure out mathematically where each of the spikes is--then you've got the period, at least, and could bump up the whole thing to align the bottom with zero, and then use OLS to get the other parameter...
# %% [markdown]
# ## Exploring statsmodels
# Based on this: https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788
# %% [markdown]
# An autocorrelation (ACF) plot represents the autocorrelation of the series with lags of itself.
# A partial autocorrelation (PACF) plot represents the amount of correlation between a series and a lag of itself that is not explained by correlations at all lower-order lags.
# Ideally, we want no correlation between the series and lags of itself. [I assume this is if you have successfully made the data stationary]
# 
# Trying it out--woah, cool. Also, what?! I was thinking that every 30 days there's serious autocorrelation...but it's like every 25 days?! This is baffling me a bit...

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(prices['price_down'].dropna(), ax=ax[0], lags=120)
ax[1] = plot_pacf(prices['price_down'].dropna(), ax=ax[1], lags=120)

# %% [markdown]
# Phew--just checking if random numbers would be autocorrelated due to something I'm not understanding about the math. No, duh.

# %%
test_rand = np.random.normal(size=500)

fig, ax = plt.subplots(2, figsize=(12,3))
ax[0] = plot_acf(test_rand, ax=ax[0], lags=120)


# %%
corr = pd.DataFrame()
corr['acf'] = statsmodels.tsa.stattools.acf(prices['price_down'], nlags=120)


# %%
corr['price'] = np.flip(prices['price_down'][:121]).reset_index(drop=True)


# %%
plt.plot(corr['acf']*10, label="autocorrelation function")
plt.plot(corr['price'], label='price')
plt.xlabel("timesteps back from 0")
plt.legend()

# %% [markdown]
# I guess I don't really understand why this is not at all lined up with the price history. Sense I'm missing something. At any rate, could get local mins and use that as the period for my sine function.
# %% [markdown]
# ### Now actually exploring rolling means and de-trending
# https://towardsdatascience.com/time-series-in-python-exponential-smoothing-and-arima-processes-2c67f2a52788

# %%
prices_sub = prices.loc[:8760,:'price_down']


# %%
prices_sub


# %%
prices_sub['z_data'] = (prices_sub['price_down'] - prices_sub['price_down'].rolling(window=12).mean()) / prices_sub['price_down'].rolling(window=12).std()
prices_sub['zp_data'] = prices_sub['z_data'] - prices_sub['z_data'].shift(12)


# %%
prices_sub.rename(columns={'price_down':'data'}, inplace=True)


# %%
def plot_rolling(df):
    fig, ax = plt.subplots(3,figsize=(12, 9))
    ax[0].plot(df.index, df.data, label='raw data')
    ax[0].plot(df.data.rolling(window=12).mean(), label="rolling mean");
    ax[0].plot(df.data.rolling(window=12).std(), label="rolling std (x10)");
    ax[0].legend()

    ax[1].plot(df.index, df.z_data, label="de-trended data")
    ax[1].plot(df.z_data.rolling(window=12).mean(), label="rolling mean");
    ax[1].plot(df.z_data.rolling(window=12).std(), label="rolling std (x10)");
    ax[1].legend()

    ax[2].plot(df.index, df.zp_data, label="12 lag differenced de-trended data")
    ax[2].plot(df.zp_data.rolling(window=12).mean(), label="rolling mean");
    ax[2].plot(df.zp_data.rolling(window=12).std(), label="rolling std (x10)");
    ax[2].legend()

    plt.tight_layout()
    fig.autofmt_xdate()


# %%
plot_rolling(prices_sub)

# %% [markdown]
# Testing if data is stationary. Now this is really bizarre, that it says the original data is stationary. Does that mean it's just random...?

# %%
from statsmodels.tsa.stattools import adfuller

df = prices_sub

print(" > Is the data stationary ?")
dftest = adfuller(df.data, autolag='AIC')
print("Test statistic = {:.3f}".format(dftest[0]))
print("P-value = {:.3f}".format(dftest[1]))
print("Critical values :")
for k, v in dftest[4].items():
    print("\t{}: {} - The data is {} stationary with {}% confidence".format(k, v, "not" if v<dftest[0] else "", 100-int(k[:-1])))
    

# %% [markdown]
# ### Seasonal decomposition, specifically
# https://towardsdatascience.com/time-series-in-python-part-2-dealing-with-seasonal-data-397a65b74051
# 
# with some help from: 
# https://www.cbcity.de/timeseries-decomposition-in-python-with-statsmodels-and-pandas.
# 
# Cool! This worked pretty well. Still so much randomness it's unbelievable, but hey.

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
decomp_freq = 24*365
decomp_annual = seasonal_decompose(prices.loc[:,'price_down'], model='additive', freq=decomp_freq)
decomp_annual.plot()
plt.show()


# %%
decomp_daily = seasonal_decompose(prices.loc[:,'price_down'], model='additive', freq=24)
decomp_daily.plot()
plt.show()


# %%
plt.plot(decomp_daily.seasonal)
plt.xlim([0,200])


# %%
decomp_annual2 = seasonal_decompose(decomp_daily.resid.dropna(),model='additive',freq=8760)
decomp_annual2.plot()
plt.show()


# %%
plt.plot(decomp_annual2.seasonal)
plt.xlim([0,8760])


# %%


