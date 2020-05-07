git stausgit# %%
import numpy as np
import pandas as pd
#import tensorflow as tf

import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm 
from statsmodels.tsa.tsatools import lagmat

# %%
df = pd.read_csv('data/intersect.csv', index_col='dt', parse_dates=True)
df_dates = pd.Series(df.index.values,name='dt')
df = df.reset_index()
#index of last hour of 2017
train_range = 35058

# set up input (x) and output (y) data for both training (2014-2017) and testing (2018)
y_train = df['price_DAH_REGDN'][:train_range]
y_test  = df['price_DAH_REGDN'][train_range:]
x_exog = df[['70th Pctl Bid',
            '60th Pctl Bid',
            '50th Pctl Bid',
            '30th Pctl Bid',
            'REGDN_Quantity',
            'REGUP_Quantity',
            'RRS_Quantity',
            'vol_DAH_REGDN',
            'vol_DAH_RRS',
            'vol_DAH_REGUP',
            'Coal',
            'NGCC',
            'Load.MW',
            'Total',
            'UPV',
            'solar_ISO',
            'wind',
            'load_naked',
            'load_net',
            'Solar',
            'Wind',
            'ng_price',
            'STWPF_SYSTEM_WIDE'
           ]]

#create a lag mat of non-forecast datapoints
exog_lagged = lagmat(x_exog,72, use_pandas=True) 
lag_extract = 'L.23|L.24|L.25|L.47|L.48|L.49|L.71|L.72|L.73'
lag_cols = exog_lagged.columns[exog_lagged.columns.str.contains(lag_extract)]
exog_lagged = exog_lagged[lag_cols]
exog_forecast = x_exog[['REGDN_Quantity','REGUP_Quantity',
                        'RRS_Quantity','STWPF_SYSTEM_WIDE'
                      ]]
#create matrices of exogenous variables for training and testing data
exog = pd.concat([exog_forecast,exog_lagged], axis=1)
exog_train = exog[:train_range]
exog_test = exog[train_range:]
#%%
#
def plot_model(y_true,y_hat,title,figsize=(15,4)):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(y_true,label='Actual Price',c='b',)
    ax.plot(y_hat,label='Modeled Price',c='r')
    plt.tight_layout()
    plt.title(title)
    plt.ylabel('Price')
    plt.xlabel('Hour')
    plt.legend()
    plt.show()
#%%
# autoregressive model
ar_model = sm.AutoReg(y_train,3,exog=exog_train).fit()
ar_y_hat = ar_model.predict(start=35058, end=43816,exog_oos=exog_test)
print('AR Summary')
ar_model.summary()
#%%
# which coeffs are most influential?
ar_params = ar_model.params.abs().sort_values(ascending=False)
print(ar_params)
#%%
#clean up y_hats for export as csv
ar_yhat_dt = pd.concat([pd.Series(ar_y_hat, name='y_hat'), df_dates],axis=1).dropna()
ar_yhat_dt = ar_yhat_dt.set_index(['dt'])
ar_yhat_dt.to_csv('data/ar_preds.csv')
# plot autoregressive model
plot_model(y_train,ar_model.fittedvalues,'Autoregressive Model (2014-2017 Modeled Values vs Training Data)')
plot_model(y_test,ar_y_hat,'Autoregressive Model (2018 Predictions vs Test Data)')

#%%
# RMSE analysis of autoregressive model
rmse = np.sqrt(((y_test - ar_model.predict(start=35058, end=43816,exog_oos=exog_test))**2).mean())
#mean_squared_error(y_test,ar_model.predict(start=35058, end=43816,exog_oos=exog_test))
print('Root Mean Squared Error of Predicted Price:' + str(rmse))

#%%
# spike analysis of autoregressive model
column_names = ['Actual Spike','Predicted Spike','Both Spike','Neither Spike']
df_2018 = pd.DataFrame(columns=column_names)
spike = np.mean(y_test) + np.std(y_test)
df_2018['Actual Spike'] = y_test >= spike
df_2018['Predicted Spike'] = ar_model.predict(start=35058, end=43816,exog_oos=exog_test) >= spike
df_2018['Both Spike'] = (y_test >= spike) & (ar_model.predict(start=35058, end=43816,exog_oos=exog_test) >= spike)
df_2018['Neither Spike'] = (y_test < spike) & (ar_model.predict(start=35058, end=43816,exog_oos=exog_test) < spike)
spikes_correct = sum(df_2018['Both Spike']) / sum(df_2018['Actual Spike'])
overall_correct = (sum(df_2018['Both Spike']) + sum(df_2018['Neither Spike'])) / len(df_2018)
print("Spikes correctly predicted, 2018: " + str(spikes_correct))
print("Overall hours correctly predicted, 2018: " + str(overall_correct))

#%%
#what is the difference between this and the one above
ar_model = sm.AutoReg(y_train,[23,24,25,47,48,49,71,72,73],exog=exog_train).fit()

plot_model(y_train,ar_model.fittedvalues,'Autoregressive Model?')
ar_yhat = y_hat = ar_model.predict(start=y_test.index[0], end=y_test.index[-1])
plot_model(y1[y_test.index[0]:],ar_yhat,'Autoregressive Model?')

print('AR Summary')
ar_model.summary()
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

# %%
#Basic Attempt at Markov Chain
markov_model = sm.MarkovRegression(y1, k_regimes=3, trend='nc', switching_variance=True).fit()
markov_model.summary()
#%%
#plot markov
fig, axes = plt.subplots(2, figsize=(20,7))
axes[0].plot(markov_model.filtered_marginal_probabilities[0])
axes[1].plot(markov_model.smoothed_marginal_probabilities[0])
