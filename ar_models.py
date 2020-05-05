# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm 
from statsmodels.tsa.tsatools import lagmat

# %%
df = pd.read_csv('data/intersect.csv', index_col='dt', parse_dates=True)
# %%
# set up our Y,Xs

y1,y2,y3,y4 = df['price_DAH_REGDN'],df['price_DAH_NSPIN'],df['price_DAH_RRS'],df['price_DAH_REGUP']
x1,x2,x3,x4 = df.drop(columns=['price_DAH_REGDN']),df.drop(columns=['price_DAH_NSPIN']),\
              df.drop(columns=['price_DAH_RRS']),  df.drop(columns=['price_DAH_REGUP'])
X = df.drop(columns=['price_DAH_REGDN','price_DAH_NSPIN','price_DAH_RRS','price_DAH_REGUP'])

#TODO include frequency in the dt
#index of last hour of 2017
train_range = 35127
#we are only concerned about REG_down prices
y_train = df['price_DAH_REGDN'][:train_range]
y_test  = df['price_DAH_REGDN'][train_range:]
X_train = X[:train_range]
X_test  = X[train_range:]
#The features determined to be most relevant, per Jess C. 
#TODO include a lag matrix of all of these.
x_exog = X[['REGDN_Unweighted Average Price_y',
            'REGDN_Total Quantity_y',
            'REGDN_Weighted Avg Price_y',
            'REGUP_Unweighted Average Price_y',
            'REGUP_Number of Bids_y',
            'REGUP_Weighted Avg Price_y',
            'NSPIN_Quantity',
            'REGDN_Quantity',
            'REGUP_Quantity',
            'RRS_Quantity',
            'Coal',
            'NGCC',
            'load_net',
            'ramp',
            'Solar',
            'Wind',
            'ACTUAL_SYSTEM_WIDE',
            'STWPF_SYSTEM_WIDE'
           ]]

#create a lag mat of non-forecast datapoints
exog_lagged = lagmat(x_exog,72, use_pandas=True) #72 hour lag perhaps? Lag all the way back?
lag_extract = 'L.24|L.25|L.26|L.48|L.49|L.50|L.72|L.73|L.74'
#TODO only 126 cols, should be ~162 since x_exog 18 cols and we want 9 lags each.
lag_cols = exog_lagged.columns[exog_lagged.columns.str.contains(lag_extract)]
exog_lagged = exog_lagged[lag_cols]
exog_forecast = x_exog[['NSPIN_Quantity','REGDN_Quantity','REGUP_Quantity',
                        'RRS_Quantity','ACTUAL_SYSTEM_WIDE','STWPF_SYSTEM_WIDE'
                      ]]
exog = pd.concat([exog_forecast,exog_lagged], axis=1)
#%%
#TODO doesnt quite work yet
def plot_model(y_true,y_hat,figsize=(15,4)):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(y_true,label='y',c='b',)
    ax.plot(y_hat,label='y_hat',c='r')
    plt.tight_layout()
    plt.legend()
    plt.show()

# fig, ax = plt.subplots(1,1,figsize=(15, 4))
# ax.plot(y1)
# ax.plot(ar_model.fittedvalues)

#%%
#Run some univariate models, just to get a feel.
ar_model = sm.AutoReg(y1,3,exog=exog).fit()
print('AR Summary')
ar_model.summary()
plot_model(y1,ar_model.fittedvalues)
#%%
ar_model = sm.AutoReg(y1,[24,25,26,48,49,50,72,73,74],exog=exog).fit()
print('AR Summary')
ar_model.summary()
plot_model(y1,ar_model.fittedvalues, title='')
ar_yhat = y_hat = ar_model.predict(start=y_test.index[0], end=y_test.index[-1])
plot_model(y1[y_test.index[0]:],ar_yhat)
# %%
#TODO need to define the p,q vals
# arma_model = sm.ARMA(y1,?,exog=exog).fit()
# print('ARMA Summary')
# arma_model.summary()
# plot_model(y1,arma_model.fittedvalues)
# %%
#TODO is our data stationary? 
#TODO need to define the p,q vals
# arima_model = sm.ARIMA(y1,exog=exog).fit()
# print('ARMIA Summary')
# arima_model.summary()
# plot_model(y1,arima_model.fittedvalues)
#%%
sarimax_model = sm.SARIMAX(y1,exog=exog).fit()
print('SARIMAX Summary:')
sarimax_model.summary()
plot_model(y1,sarimax_model.fittedvalues)
#%%
#Run multivariate models
#Removed since we are only concerned with
# var_model = sm.VAR(Y).fit()
# print('VAR Summary:')
# var_model.summary()

# varmax_model = sm.VARMAX(Y).fit()
# print('VARMAX Summary:')
# varmax_model.summary()

# %%
#Basic Attempt at Markov Chain
markov_model = sm.MarkovRegression(y1, k_regimes=3, trend='nc', switching_variance=True).fit()
markov_model.summary()
#%%
#plot markov
fig, axes = plt.subplots(2, figsize=(20,7))
axes[0].plot(markov_model.filtered_marginal_probabilities[0])
axes[1].plot(markov_model.smoothed_marginal_probabilities[0])
# %% 
# attempt a RNN
# https://www.tensorflow.org/tutorials/structured_data/time_series

# %%
df_nan = df.describe().loc['count'] < df.shape[0]
df_nan = df.describe().loc['count'][df_nan].sort_values()
df_na = df.isna().any(axis=1)



# %%
