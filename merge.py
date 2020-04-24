# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {'bids':pd.read_csv('data/as_bid_aggregated_data.csv'),
        'plans':pd.read_csv('data/as_plan.csv'),
        'energy_prices':pd.read_csv('data/as_bid_aggregated_data.csv'),
        'price_vol':pd.read_csv('data/as_price_vol.csv'),
        'generation':pd.read_csv('data/generation.csv'),
        'weather':pd.read_csv('data/weather_forecast_ercot.csv'),
       }

#clean up df-specific bits
data['generation'].fillna(0)
# %%
def create_dt(input_df,date_col,hr_col,tz ='America/Chicago'):
    #TODO runs quite slow, optimize if possible.
    '''create a datetime index for a dataframe from multiple cols
    '''
    #TODO this would allow idempotence, but it in turn doesnt allow modifying the global from data[key]
    input_df = input_df.copy()
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
#process the rest of the dfs.
for key, df in data.items():
    data[key] = create_dt(data[key],'date','hr_beg')

# %%
#TODO refactor into function
start_time = []
end_time = []
for key, df in data.items():
    start_ts = df.index.sort_values()[0]
    end_ts = df.index.sort_values()[-1]
    #TODO can this be done in single line?
    start_time.append(start_ts)
    end_time.append(end_ts)
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

date_range = pd.date_range(min(start_time), max(end_time),freq='H')
union_dates = pd.date_range(max(start_time), min(end_time),freq='H')
intersect_df = pd.DataFrame(date_range, columns=["dt"])
union_df = pd.DataFrame(union_dates, columns=["dt"])

for key in data.keys():
    intersect_df = intersect_df.merge(data[key], how='left',left_on='dt',
                             right_on=data[key].index,left_index=True)
    union_df = union_df.merge(data[key], how='left',left_on='dt',
                             right_on=data[key].index,left_index=True)
    intersect_df.reset_index(inplace=True,drop=True)
    union_df.reset_index(inplace=True,drop=True)

intersect_df = intersect_df.set_index('dt')
union_df = union_df.set_index('dt')

intersect_df.to_csv('data/intersect.csv')
union_df.to_csv('data/union.csv')

# %%
intersect_nan = intersect_df.describe().loc['count'] < df.shape[0]
intersect_nan = intersect_df.describe().loc['count'][intersect_nan].sort_values()
intersect_na = intersect_df.isna().any(axis=1)
# %%
