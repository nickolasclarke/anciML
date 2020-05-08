import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean



def InputsSolarMarket():

    # ------------------------ Inputs ------------------------------#

    # Day Ahead Energy Market 2018
    dataset_E = pd.read_csv('in/all_ercot_profiles_hourly_2018.csv')
    E_price = dataset_E.iloc[0:8762, 4].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('in/AS_price.csv')
    ASM_price = dataset_AS.iloc[70080:78840, 9].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Solar CF
    dataset_solar = pd.read_csv('in/all_ercot_profiles_hourly_2018.csv')  # Reading the dataset of solar gen CF
    solar_cf = dataset_solar.iloc[0:8762, 1].values  # Creating the solar generation Vector, start 1/1/2018 00:00 (CST)

    return E_price, solar_cf, ASM_price



def InputsSolarUncertainMul(eta):
    # ---------Imports ------#

    data = pd.DataFrame(pd.read_csv('in/all_ercot_profiles_hourly.csv', sep=';'))  # Reading the dataset of solar gen CF
    dataset_solar = data.loc[:, ['year', 'CF_model_solar']]

    data_raw2015 = pd.DataFrame(pd.read_csv('in/solar_TX.csv', sep=';'))  # Reading the dataset of solar gen CF
    data2015 = data_raw2015.iloc[0:8760, 5].values.tolist()

    df_years = pd.DataFrame({'CF_2015': data2015, 'CF_2016': dataset_solar.loc[
        dataset_solar['year'] == 2016, 'CF_model_solar'].values.tolist(), 'CF_2017': dataset_solar.loc[
        dataset_solar['year'] == 2017, 'CF_model_solar'].values.tolist(), 'CF_2018': dataset_solar.loc[
        dataset_solar['year'] == 2018, 'CF_model_solar'].values.tolist()})
    df = df_years.stack()

    # --------Summary statistics - annual average day repeated ---#

    df_years['Av_CF'] = df_years.mean(axis=1)
    df_years['Std_CF'] = df_years.std(axis=1)
    mean_cf = np.array(df_years['Av_CF'])
    std_cf = np.array(df_years['Std_CF'])

    # Inverse cdf for average year
    #inv_cdf = stat.mean([np.percentile(df_years['CF_2015'], eta), np.percentile(df_years['CF_2016'], eta), np.percentile(df_years['CF_2017'], eta), np.percentile(df_years['CF_2018'], eta)])
    #Above is for the stacked version - no!

    #inv_cdf = np.percentile(df_years['Av_CF'], eta)
    inv_cdf_raw = np.percentile(df_years['Av_CF'], eta)
    inv_cdf = np.array([inv_cdf_raw for i in range(8760)])

    # --------Create plots of cdf --------------#
    num_bins = int(np.ceil(np.sqrt(8760)))
    data = df_years['CF_2015']
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf / cdf[-1], color='darkcyan', label='2015')

    data = df_years['CF_2016']
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf / cdf[-1], color='powderblue', label='2016')

    data = df_years['CF_2017']
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf / cdf[-1], color='darkturquoise', label='2017')

    data = df_years['CF_2018']
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf / cdf[-1], color='yellowgreen', label='2018')

    data = df_years['Av_CF']
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf / cdf[-1], color='black', label='Av')

    data = df
    counts, bin_edges = np.histogram(data, bins=num_bins)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf / cdf[-1], color='red', label='Stack')

    plt.xlabel('Solar Capacity Factor', fontsize=10)
    plt.ylabel('CDF', fontsize=10)
    plt.title('Multi-year and average solar capacity factor', fontsize=12)
    plt.legend()
    plt.show()

    return mean_cf, std_cf, inv_cdf


def InputsSolarUncertainHourly(eta, seasonal): #not used right now

    # ---------Imports ------#

    data = pd.DataFrame(pd.read_csv('in/all_ercot_profiles_hourly.csv', sep=';'))
    dataset_solar_raw = data.loc[:, ['local_time_hb', 'CF_model_solar']]

    data_raw2015 = pd.DataFrame(pd.read_csv('in/solar_TX.csv', sep=';'))  # Reading the dataset of solar gen CF
    data2015 = pd.DataFrame(data_raw2015.iloc[0:8760, 5])
    data2015['local_time_hb'] = dataset_solar_raw.loc[0:8759, ['local_time_hb']] #change the six hours difference

    #consolidate data for 4 years

    dataset_solar = data2015.append(dataset_solar_raw, ignore_index=True)

    # --------Parse data----#
    dataset_solar.loc[:, 'dates-to-parse'] = pd.to_datetime(dataset_solar['local_time_hb'])

    dataset_solar.loc[:, 'month-of-year'] = pd.to_numeric(dataset_solar.loc[:, 'dates-to-parse'].dt.month)
    dataset_solar.loc[:, 'week-of-year'] = pd.to_numeric(dataset_solar.loc[:, 'dates-to-parse'].apply(lambda x: str(x.isocalendar()[1]).zfill(2)))
    dataset_solar.loc[:, 'day-of-week'] = pd.to_numeric(dataset_solar.loc[:, 'dates-to-parse'].apply(lambda x: str(x.isocalendar()[2]).zfill(2)))
    dataset_solar.loc[:, 'hour-of-day'] = pd.to_numeric(dataset_solar.loc[:, 'dates-to-parse'].apply(lambda x: str(x.hour).zfill(2)))

    dataset_solar = dataset_solar.drop('dates-to-parse', axis=1)

    if seasonal == True:
        winter = dataset_solar[(dataset_solar['month-of-year'] == 1) | (dataset_solar['month-of-year'] == 2) | (dataset_solar['month-of-year'] == 3)].groupby('hour-of-day')['CF_model_solar'].apply(list).reset_index(name='CF_model_solar')
        spring = dataset_solar[(dataset_solar['month-of-year'] == 4) | (dataset_solar['month-of-year'] == 5) | (dataset_solar['month-of-year'] == 6)].groupby('hour-of-day')['CF_model_solar'].apply(list).reset_index(name='CF_model_solar')
        summer = dataset_solar[(dataset_solar['month-of-year'] == 7) | (dataset_solar['month-of-year'] == 8) | (dataset_solar['month-of-year'] == 9)].groupby('hour-of-day')['CF_model_solar'].apply(list).reset_index(name='CF_model_solar')
        fall =  dataset_solar[(dataset_solar['month-of-year'] == 10) | (dataset_solar['month-of-year'] == 11) | (dataset_solar['month-of-year'] == 12)].groupby('hour-of-day')['CF_model_solar'].apply(list).reset_index(name='CF_model_solar')

        inv_win = [np.percentile(winter['CF_model_solar'][i], eta) for i in range(0, len(winter))] * 90
        inv_spring = [np.percentile(spring['CF_model_solar'][i], eta) for i in range(0, len(spring))] * 91
        inv_summer = [np.percentile(summer['CF_model_solar'][i], eta) for i in range(0, len(summer))] * 92
        inv_fall = [np.percentile(fall['CF_model_solar'][i], eta) for i in range(0, len(fall))] * 92

        inv_cdf = inv_win + inv_spring + inv_summer + inv_fall
        print('Typical seasonal day cdf', inv_cdf)


    else:

        # --------Parse data for hourly changes----#

        hourly_cf = dataset_solar.groupby('hour-of-day')['CF_model_solar'].apply(list).reset_index(name='CF_model_solar')
        #print(len(hourly_cf)) #this has 1460=365*4 hours x per row


        #Inverse cdf for each hour
        data = hourly_cf['CF_model_solar']
        plt.figure()

        # for i in range(0, len(data)):
        #
        #     # ------Create cdf--------#
        #
        #     if data[i] == [0] * len(data[i]):
        #         continue
        #     else:
        #         # Plot empirical cumulative distribution using Matplotlib and Numpy.
        #         num_bins = 20
        #         counts, bin_edges = np.histogram(data[i], bins=num_bins)
        #         cdf = np.cumsum(counts)
        #         n = 25
        #         colors = plt.cm.gnuplot2(np.linspace(0, 1, n))
        #         plt.plot(bin_edges[1:], cdf / cdf[-1], label=i, color=colors[i])
        #         plt.xlabel('Solar Capacity Factor', fontsize=10)
        #         plt.ylabel('CDF', fontsize=10)
        #         plt.title('Cumulative distribution of hourly solar CF', fontsize=15)
        #         plt.legend()
        #         plt.show()


        #typical day cdfs
        inv_cdf = [np.percentile(data[i], eta) for i in range(0, len(data))]* 365
        print('Typical day cdf', inv_cdf)

    return inv_cdf





def InputsMarketML():


    # ------------------------ ML Results - Inputs ------------------------------#

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('in/persistence_results_48.csv')
    ASM_price_pers = dataset_AS.iloc[0:8762,3].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('in/random_forest_results_48.csv')
    ASM_price_rr = dataset_AS.iloc[0:8762,5].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('in/markov_chain_results.csv')
    ASM_price_markov = dataset_AS.iloc[0:8762,3].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('in/markov_chain_results.csv')
    ASM_price_markov = dataset_AS.iloc[0:8762, 3].values  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST

    # Day Ahead Market - AS Down Regulation ERCOT hourly prices 2018
    dataset_AS = pd.read_csv('in/ar_preds_central.csv', sep=';')
    ASM_price_arx_raw = dataset_AS.iloc[0:8762, 1]
    ASM_price_arx = np.array([0 if i < 0 else i for i in ASM_price_arx_raw])  # Creating the price Vector $/MWh, start at 1/1/2018 00:00 CST


    return ASM_price_markov, ASM_price_pers, ASM_price_rr, ASM_price_arx





