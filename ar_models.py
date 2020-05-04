# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm 


# %%
df = pd.read_csv('data/intersect.csv', index_col='dt', parse_dates=True)
# %%
# set up our Y,Xs
y1,y2,y3,y4 = df['price_DAH_REGDN'],df['price_DAH_NSPIN'],df['price_DAH_RRS'],df['price_DAH_REGUP']
x1,x2,x3,x4 = df.drop(columns=['price_DAH_REGDN']),df.drop(columns=['price_DAH_NSPIN']),\
              df.drop(columns=['price_DAH_RRS']),  df.drop(columns=['price_DAH_REGUP'])

Y = df[['price_DAH_REGDN','price_DAH_NSPIN','price_DAH_RRS','price_DAH_REGUP']]
X = df.drop(columns=['price_DAH_REGDN','price_DAH_NSPIN','price_DAH_RRS','price_DAH_REGUP'])

# x_exog = x[['RE']
# ]
#%%
#Run some univariate models, just to get a feel.
ar_model = sm.AutoReg(y1,3).fit()
print('AR Summary')
ar_model.summary()
#%%
ar_model = sm.AutoReg(y1,[24,25,26]).fit()
print('AR Summary')
ar_model.summary()
#%%
sarimax_model = sm.SARIMAX(y1).fit()
print('SARIMAX Summary:')
sarimax_model.summary()

#%%
#Run multivariate models
var_model = sm.VAR(Y).fit()
print('VAR Summary:')
var_model.summary()

varmax_model = sm.VARMAX(Y).fit()
print('VARMAX Summary:')
varmax_model.summary()

# %%
df_nan = df.describe().loc['count'] < df.shape[0]
df_nan = df.describe().loc['count'][df_nan].sort_values()
df_na = df.isna().any(axis=1)



# %%
