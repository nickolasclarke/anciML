# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


bids = pd.read_csv('data/as_bid_aggregated_data.csv')
plans = pd.read_csv('data/as_plan.csv')
energy_prices = pd.read_csv('data/as_bid_aggregated_data.csv')
price_vol = pd.read_csv('data/as_price_vol.csv')
#generation = pd.read_csv('data/generation.csv')
weather = pd.read_csv('data/weather_forecast_ercot.csv')

data = {'bids':bids,
        'plans':plans,
        'energy_prices':energy_prices,
        'price_vol':price_vol,
        #'generation':generation,
        'weather':weather,
       }
#clean up df-specific bits
#data['generation'].fillna(0)
#%%
# process weather seperately, as hour 23 is indexed as hour -1 on the next day.
# def process_weather(df):
#     #df = df.drop_duplicates()
#     weather_negative = df[df['hr_beg'] == -1]
#     #change value to 23rd hour
#     weather_negative['hr_beg'] = 23
#     weather_negative['date'] = pd.to_datetime(df['date'])
#     #change date to one day in the past
#     weather_negative['date'] = pd.to_datetime(weather_negative['date'])-pd.Timedelta('1 days')
#     #TODO results in funky dt objects for 23rd hour, no idea why.
#     df[df['hr_beg'] == -1] = weather_negative
#     df['date'] = pd.to_datetime(df.date)
#     df['hr_beg'] = pd.to_datetime(df['hr_beg'],format='%H').dt.time
#     #TODO HAXXXXXY
#     df = df.set_index(df.apply(lambda x: pd.Timestamp.combine(x['date'], x['hr_beg']), axis=1))
#     df = df.drop(columns=['date','hr_beg'])
#     return df

# data['weather'] = process_weather(weather)

# %%
def create_dt(input_df,date_col,hr_col,tz ='America/Chicago'):
    #TODO runs quite slow, optimize if possible.
    '''create a datetime index for a dataframe from multiple cols
    '''
    #TODO this would allow idempotence, but it in turn doesnt allow modifying the global from data[key]
    #df = input_df.copy()
    #input_df = input_df.drop_duplicates()
    if isinstance(input_df.index, pd.DatetimeIndex):
        return input_df
    #TODO raise exceptions for multiple dt string formats in a column
    if input_df[hr_col].astype(str).str.len().nunique() == 1:
        #hr_col is probably already in a inferable dt string format
        input_df[hr_col] = pd.to_datetime(input_df[hr_col]).dt.time
    else:    
        input_df[hr_col] = pd.to_datetime(input_df[hr_col],format='%H').dt.time
    dt_index = pd.to_datetime(input_df[date_col].astype(str)+'T'+input_df[hr_col].astype(str))
    #dt_index = pd.DatetimeIndex(dt_index,ambiguous='NaT',tz=tz,freq='H',)
    input_df = input_df.set_index(dt_index)
    input_df = input_df.drop(columns=[date_col,hr_col])
    #TODO this fails
    #input_df = input_df.asfreq('H')
    return input_df

# %%
#process the rest of the dfs. NOTE: modified dfs only present in `data` dict
#TODO https://stackoverflow.com/questions/61377110/updating-dict-value-does-not-update-global-var-it-references
for key, df in data.items():
    data[key] = create_dt(data[key],'date','hr_beg')

# %%
#TODO refactor into function
times = []
for key, df in data.items():
    start_ts = df.index.sort_values()[0]
    end_ts = df.index.sort_values()[-1]
    #TODO can this be done in single line?
    times.append(start_ts)
    times.append(end_ts)
    mask = df.index.to_series().diff() > pd.Timedelta('01:00:00')
    missing_ts = df[mask].index
    nan_cols = df.describe().loc['count'] < df.shape[0]
    nan_cols = df.describe().loc['count'][nan_cols].sort_values()
    na = df.isna().any(axis=1)
    res = df[na]
    print(f'{key} Summary:\n'
          f"ts range: {start_ts} - {end_ts}\n"
          f'Total Rows: {df.shape[0]}\n'
          f'Rows NaN count: {res.shape[0]}\n'
          f'Cols with NaN: \n{nan_cols}\n'
          f'position of ts gaps: \n{missing_ts}\n'
          f'Partial Describe:\n{df.describe().iloc[:,:3]}\n'
          )

# %%
#attempt to join on full data
date_range = pd.date_range(min(times), max(times),freq='H',)#tz='America/Chicago')
joined_df = pd.DataFrame(date_range, columns=["dt"])
for key in data.keys():
    joined_df = joined_df.merge(data[key], how='left',left_on='dt',
                             right_on=data[key].index,left_index=True)
    joined_df.reset_index(inplace=True,drop=True)

joined_df = joined_df.set_index('dt')
join_nan = joined_df.describe().loc['count'] < df.shape[0]
join_nan = joined_df.describe().loc['count'][join_nan].sort_values()
joined_na = joined_df.isna().any(axis=1)


# %%
sequence = np.array(joined_df.dropna(how='any').index)    
longest_seq = max(np.split(sequence, np.where(np.diff(sequence) != 1)[0]+1), key=len)
longest_seq = pd.DatetimeIndex(longest_seq)
#fails, says out of indexers are out of bounds
joined_df.iloc[longest_seq]

# %%
