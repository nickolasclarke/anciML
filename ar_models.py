# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm 


# %%
df = pd.read_csv('data/intersect.csv', index_col='dt', parse_dates=True)
# %%
# set up our Y,Xs
#TODO train, test, split
y1,y2,y3,y4 = df['price_DAH_REGDN'],df['price_DAH_NSPIN'],df['price_DAH_RRS'],df['price_DAH_REGUP']
x1,x2,x3,x4 = df.drop(columns=['price_DAH_REGDN']),df.drop(columns=['price_DAH_NSPIN']),\
              df.drop(columns=['price_DAH_RRS']),  df.drop(columns=['price_DAH_REGUP'])

Y = df[['price_DAH_REGDN','price_DAH_NSPIN','price_DAH_RRS','price_DAH_REGUP']]
X = df.drop(columns=['price_DAH_REGDN','price_DAH_NSPIN','price_DAH_RRS','price_DAH_REGUP'])

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
           ]]

#%%
#TODO doesnt quite work yet
def plot_model(model,figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    ax = plt.plot(y1)
    ax = model.plot_predict()
    plt.tight_layout()
    plt.show()

#%%
#Run some univariate models, just to get a feel.
ar_model = sm.AutoReg(y1,3, exog=x_exog).fit()
print('AR Summary')
ar_model.summary()
plot_model(ar_model,(5,2))
#%%
ar_model = sm.AutoReg(y1,[24,25,26],exog=x_exog).fit()
print('AR Summary')
ar_model.summary()
# %%
arma_model = sm.ARMA(y1,exog=x_exog).fit()
print('ARMA Summary')
arma_model.summary()
# %%
arima_model = sm.ARIMA(y1,exog=x_exog).fit()
print('ARMIA Summary')
arima_model.summary()

#%%
sarimax_model = sm.SARIMAX(y1,exog=x_exog).fit()
print('SARIMAX Summary:')
sarimax_model.summary()
sarimax_model.plot_model()
#%%
#Run multivariate models
var_model = sm.VAR(Y).fit()
print('VAR Summary:')
var_model.summary()

varmax_model = sm.VARMAX(Y).fit()
print('VARMAX Summary:')
varmax_model.summary()

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
