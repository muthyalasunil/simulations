import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import random
import datetime as dt
from scipy import stats
import statistics
from hurst import compute_Hc

import warnings

warnings.filterwarnings("ignore")

DATA_FOLDER = 'data'


def load_data(filename):
    _data_df = pd.read_csv(DATA_FOLDER + '\\' + filename)
    return _data_df


def run_simuation(simulations, iterations, threshold, value_list, r_data_df, close_val):
    print(simulations, iterations, threshold)
    index_value = []
    for g in range(iterations):
        sim_results = []
        '''
        for sim in range(simulations):
            random_num = abs(random.choice(value_list))
            if math.isnan(random_num):
                random_num = 0
            sim_results.append(random_num)

        sim_rand_val = abs(np.mean(sim_results))
        if math.isnan(sim_rand_val):
            sim_rand_val = abs(np.median(sim_results))
        '''
        random_num = abs(random.choice(value_list))
        number = int(np.random.uniform() * 100)  # get a random number to see who wins
        if number > threshold:
            index_value.append(-random_num)
        else:
            index_value.append(random_num)

    r_data_df['sim_diff'] = np.array(index_value).tolist()
    r_data_df['sim_values'] = r_data_df['sim_diff'].cumsum()
    r_data_df['sim_values'] = r_data_df['sim_values'] + close_val

    return r_data_df


def run_simuations(simulations, iterations, threshold, value_list, r_data_df, close_val):
    # print(simulations, iterations, threshold)
    sim_results = []
    for sim in range(simulations):
        index_values = []

        for ite in range(iterations):
            random_num = abs(random.choice(value_list))
            if math.isnan(random_num):
                random_num = 0
            else:
                number = int(np.random.uniform() * 100)  # get a random number to see who wins
                if number > threshold:
                    random_num = -random_num
            index_values.append(random_num)
        sim_results.append(np.array(index_values).tolist())

    i = 1
    for index_list in sim_results:
        col_name = 'sim' + str(i)
        r_data_df[col_name] = index_list
        r_data_df[col_name] = r_data_df[col_name].cumsum()
        r_data_df[col_name] = r_data_df[col_name] + close_val
        simH, simC, data = compute_Hc(r_data_df[col_name].tolist(), kind='price', simplified=True)
        # print("simH={:.4f}, simC={:.4f} col_name={:s}".format(simH, simC, col_name))
        i += 1
    return r_data_df


def evaluate_HC(c_data_df, years, span):
    data_df = c_data_df.iloc[-(span * years):]
    cls_data = data_df['Close'].tolist()
    H, c, data = compute_Hc(cls_data, kind='price', simplified=True)
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data)), cls_data)
    stddev = statistics.stdev(cls_data)

    return H, c, cls_data, stddev, slope


def project_forward(c_data_df, span):
    data_df = c_data_df.iloc[-(span * 2):].head(span)
    r_data_df = c_data_df.iloc[-span:]

    print("base data begin={:s}, end={:s}".format(str(data_df.iloc[0]['Date']), str(data_df.iloc[-1]['Date'])))
    close_val = data_df.iloc[-1]['Close']
    print("last Close={:.4f}".format(close_val))

    c_data_df['diff'] = c_data_df['Close'].diff()
    c_data_df['trend'] = np.where(c_data_df['diff'] > 0, 1, 0)

    value_list = data_df['Close'].diff().tolist()  # c_data_df['diff'].tolist()
    value_list = [0 if math.isnan(x) else x for x in value_list]
    value_list = set(value_list)
    # convert the set to the list
    value_list = (list(value_list))

    threshold = int(c_data_df['trend'].sum() / c_data_df['trend'].count() * 100)
    iterations = r_data_df['Close'].count()
    ret_data_df = run_simuations(1000, iterations, threshold, value_list, r_data_df, close_val)
    return ret_data_df


def plot_corr(r_data_df, r1=20, r2=50, plot=False):
    cols = r_data_df.columns.values
    r_data_df_20 = r_data_df.copy()
    for col in cols:
        r_data_df_20[col] = r_data_df_20[col].rolling(r1).std()
    for col in cols:
        r_data_df[col] = r_data_df[col].rolling(r2).std()

    r_data_df = r_data_df.dropna()
    r_data_df_20 = r_data_df_20.dropna()

    corr20 = r_data_df_20.corr()[['Close']].sort_values(by='Close', ascending=False)
    corr50 = r_data_df.corr()[['Close']].sort_values(by='Close', ascending=False)
    corr20.rename(columns={'Close': 'corr_20'}, inplace=True)
    corr50.rename(columns={'Close': 'corr_50'}, inplace=True)

    if plot:
        # fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # sns.heatmap(corr, ax=ax[0], annot=True)
        sns.heatmap(corr20, ax=ax, annot=True)

    return corr20.join(corr50, how="inner")


def project_close_values(c_data_df, span=250):
    print("data_df beg={:s}, data_df end={:s}".format(str(c_data_df.iloc[0]['Date']), str(c_data_df.iloc[-1]['Date'])))

    H3, c3, cls_data3, stddev3, slp3 = evaluate_HC(c_data_df, 3, span)
    print("std 3yr ={:.4f}, H={:.4f}".format(stddev3, H3))
    H5, c5, cls_data5, stddev5, slp5 = evaluate_HC(c_data_df, 5, span)
    print("std 5yr ={:.4f}, H={:.4f}".format(stddev5, H5))
    H7, c7, cls_data7, stddev7, slp7 = evaluate_HC(c_data_df, 7, span)
    print("std 7yr ={:.4f}, H={:.4f}".format(stddev7, H7))
    H10, c10, cls_data10, stddev10, slp10 = evaluate_HC(c_data_df, 10, span)
    print("std 10yr ={:.4f}, H={:.4f}".format(stddev10, H10))

    r_data_df = project_forward(c_data_df, span)

    simHmap = {}
    simSmap = {}

    for i in range(1000):
        col_name = 'sim' + str(i + 1)
        sim_list = r_data_df[col_name].tolist()

        cls_data3 = cls_data3[:len(cls_data3) - len(sim_list)]
        cls_data3.extend(sim_list)
        simH3, simC3, data3 = compute_Hc(cls_data3, kind='price', simplified=True)
        cls_stddev3 = statistics.stdev(cls_data3)
        cls_slp3, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data3)), cls_data3)

        cls_data5 = cls_data5[:len(cls_data5) - len(sim_list)]
        cls_data5.extend(sim_list)
        simH5, simC5, data5 = compute_Hc(cls_data5, kind='price', simplified=True)
        cls_stddev5 = statistics.stdev(cls_data5)
        cls_slp5, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data5)), cls_data5)

        cls_data7 = cls_data7[:len(cls_data7) - len(sim_list)]
        cls_data7.extend(sim_list)
        simH7, simC7, data7 = compute_Hc(cls_data7, kind='price', simplified=True)
        cls_stddev7 = statistics.stdev(cls_data7)
        cls_slp7, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data7)), cls_data7)

        cls_data10 = cls_data10[:len(cls_data10) - len(sim_list)]
        cls_data10.extend(sim_list)
        simH10, simC10, data10 = compute_Hc(cls_data10, kind='price', simplified=True)
        cls_stddev10 = statistics.stdev(cls_data10)
        cls_slp10, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data10)), cls_data10)

        simHmap[col_name] = np.mean([abs(H3 - simH3), abs(H5 - simH5), abs(H7 - simH7), abs(H10 - simH10),
                                     abs(slp3 - cls_slp3), abs(slp5 - cls_slp5), abs(slp7 - cls_slp7),
                                     abs(slp10 - cls_slp10)])
        simSmap[col_name] = np.mean([abs(stddev3 - cls_stddev3), abs(stddev5 - cls_stddev5), abs(stddev7 - cls_stddev7),
                                     abs(stddev10 - cls_stddev10)])

    simHmap = dict(sorted(simHmap.items(), key=lambda item: item[1]))
    topN = 3
    r1, r2 = 20, 50
    for key in simHmap:
        if topN < 0:
            r_data_df.drop([key], axis=1, inplace=True)
            del simSmap[key]
        else:
            print(key + ' stddev:' + str(simSmap[key]))

        topN = topN - 1

    r_data_df.set_index('Date', inplace=True)
    prev_close_df = c_data_df.iloc[-(span * 2):].head(span)
    r_data_df_copy = r_data_df.copy()
    r_data_df_copy['Close'] = np.array(prev_close_df['Close']).tolist()
    results = plot_corr(r_data_df_copy, 20, 50)

    for idx in results.index.values:
        if idx.split('_')[0] in simSmap:
            results.at[idx, 'stddev_diff'] = simSmap[idx.split('_')[0]]
            results.at[idx, 'close_diff'] = abs(r_data_df.iloc[-1][idx] - r_data_df.iloc[-1]['Close'])

    print(results)

    '''
    for col in r_data_df.columns:
        print(col + ':' + str(r_data_df.iloc[-1][col]))
    

    # r_data_df.plot(kind='line')
    # plt.legend(loc='best')
    # plt.show()

    r_data_df.to_csv('data/r_data_df.csv', index=True)
    '''


if __name__ == '__main__':
    print('main...')
    c_data_df = load_data('sp500.csv')
    c_data_df.rename(columns=lambda x: x.strip(), inplace=True)
    c_data_df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)

    c_data_df['Date'] = pd.to_datetime(c_data_df.Date)
    c_data_df = c_data_df.sort_values('Date')
    c_data_df = c_data_df[c_data_df['Date'] < '01/01/2020']

    project_close_values(c_data_df)

    ''' 
            r_data_df.set_index('Date', inplace=True)
            c_data_df.drop(['diff', 'trend'], axis=1, inplace=True)
        
            plt.plot(r_data_df)
            plt.plot(c_data_df.tail(span*2))
            plt.legend(loc='best')
            plt.show()
        
            r_data_df.to_csv('data/r_data_df.csv', index=True)        
                    r_data_df.set_index('Date', inplace=True)
            plt.plot(r_data_df['Close'], label='Close')
            plt.plot(r_data_df['SMA20'], label='SMA20')
            plt.plot(r_data_df['SMA50'], label='SMA50')
            plt.legend(loc=2)
            plt.show()
            
            r_data_df.set_index('Date')
            plt.plot(r_data_df['sim_values'], linestyle='--', color='blue')
            plt.plot(r_data_df['Close'], linestyle='--', color='black')
        
            r_data_df.to_csv('data/r_data_df.csv', index=False)
            # print(r_data_df.iloc[:10].to_string())
    '''
