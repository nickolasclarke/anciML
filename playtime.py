# %%
import numpy as np
import pandas as pd

bids = pd.read_csv('data/as_bid_aggregated_data.csv')
plans = pd.read_csv('data/as_plan.csv')
energy_prices = pd.read_csv('data/as_bid_aggregated_data.csv')
price_vol = pd.read_csv('data/as_price_vol.csv')
generation = pd.read_csv('data/generation.csv')
weather = pd.read_csv('data/weather_forecast_ercot.csv')

data = {'bids':bids,
        'plans':plans,
        'energy_prices':energy_prices,
        'price_vol':price_vol,
        'generation':generation,
        'weather':weather,
       }

# %%
# process weather seperately, as hour 23 is indexed as hour -1 on the next day.
def process_weather(df):
    weather_negative = df[df['hr_beg'] == -1]
    #change value to 23rd hour
    weather_negative['hr_beg'] = 23
    weather_negative['date'] = pd.to_datetime(df['date'])
    #change date to one day in the past
    weather_negative['date'] = pd.to_datetime(weather_negative['date'])-pd.Timedelta('1 days')
    #TODO results in funky dt objects for 23rd hour, no idea why.
    df[df['hr_beg'] == -1] = weather_negative
    df['date'] = pd.to_datetime(df.date)
    df['hr_beg'] = pd.to_datetime(df['hr_beg'],format='%H').dt.time
    #TODO HAXXXXXY
    df = df.set_index(df.apply(lambda x: pd.Timestamp.combine(x['date'], x['hr_beg']), axis=1))
    df = df.drop(columns=['date','hr_beg'])
    return df

weather = process_weather(weather)
# %%
def create_dt(input_df,date_col,hr_col):
    #TODO runs quite slow, optimize if possible.
    '''create a datetime index for a dataframe from multiple cols
    '''
    df = input_df.copy()
    #TODO raise exceptions for multiple dt string formats in a column
    if df[hr_col].astype(str).str.len().nunique() == 1:
        #hr_col is probably already in a inferable dt string format
        df[hr_col] = pd.to_datetime(df[hr_col]).dt.time
    else:    
        df[hr_col] = pd.to_datetime(df[hr_col],format='%H').dt.time
    df = df.set_index(pd.to_datetime(df[date_col].astype(str)+'-'+df[hr_col].astype(str)))
    df = df.drop(columns=[date_col,hr_col])
    #TODO this fails
    df = df.asfreq('H')
    return df

# %%
#process the rest of the dfs
#TODO this extremely simple loop doesn't work. Why? 
# for df in data:
#     df = create_dt(df,'date','hr_beg')
# %%
bids = create_dt(bids, 'date', 'hr_beg')
plans = create_dt(plans, 'date', 'hr_beg')
energy_prices = create_dt(energy_prices, 'date', 'hr_beg')
price_vol = create_dt(price_vol, 'date', 'hr_beg')
generation = create_dt(generation, 'date', 'hr_beg')
# weather = create_dt(weather, 'date', 'hr_beg')
# %%
for key, df in data.items():
    nan_cols = df.describe().loc['count'] < generation.shape[0]
    nan_cols = df.describe().loc['count'][nan_cols].sort_values()
    na = df.isna().any(axis=1)
    res = df[na]
    print(f'{key} Summary:\n  Rows with NaN: {res.shape[0]}\n'\
          f'                  Cols with NAn: {nan_cols}\n'
          f'{df.describe()}'
          )

# %%
#exploratory joins
joined_df = bids.join(weather, how='inner')
joined_df = joined_df.asfreq('H')
joined_df.describe().loc['count']
#this should be ~35,158 hours if continuous
time_gaps = joined_df.index - joined_df.index.shift(1)
time_gaps.plot()

# %%
