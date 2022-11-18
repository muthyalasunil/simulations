import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import random
from hurst import compute_Hc

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
        print("simH={:.4f}, simC={:.4f} col_name={:s}".format(simH, simC, col_name))
        i += 1
    return r_data_df


def evaluate_HC(c_data_df, years, span):
    data_df = c_data_df.iloc[-(span * years):]
    cls_data = data_df['Close'].tolist()
    H, c, data = compute_Hc(cls_data, kind='price', simplified=True)
    return H, c, cls_data

def project_forward(c_data_df, span):
    data_df = c_data_df.iloc[-(span * 2):].head(span)
    r_data_df = c_data_df.iloc[-span:]
    print("begin={:s}, end={:s}".format(str(data_df.iloc[0]['Date']), str(data_df.iloc[-1]['Date'])))
    c_data_df['diff'] = c_data_df['Close'].diff()
    c_data_df['trend'] = np.where(c_data_df['diff'] > 0, 1, 0)

    close_val = data_df.iloc[-1]['Close']
    print("Close={:.4f}".format(close_val))

    value_list = c_data_df['diff'].tolist()
    value_list = [0 if math.isnan(x) else x for x in value_list]
    value_list = set(value_list)
    # convert the set to the list
    value_list = (list(value_list))

    threshold = int(c_data_df['trend'].sum() / c_data_df['trend'].count() * 100)
    iterations = r_data_df['Close'].count()
    ret_data_df = run_simuations(1000, iterations, threshold, value_list, r_data_df, close_val)
    return ret_data_df


if __name__ == '__main__':
    print('Hello...')
    c_data_df = load_data('sp500.csv')
    c_data_df.rename(columns=lambda x: x.strip(), inplace=True)
    c_data_df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)

    c_data_df['Date'] = pd.to_datetime(c_data_df.Date)
    c_data_df = c_data_df.sort_values('Date')
    print("data_df beg={:s}, data_df end={:s}".format(str(c_data_df.iloc[0]['Date']), str(c_data_df.iloc[-1]['Date'])))
    span = 250

    H3, c3, cls_data3 = evaluate_HC(c_data_df, 3, span)
    print("H={:.4f}, c={:.4f}".format(H3, c3))
    H5, c5, cls_data5 = evaluate_HC(c_data_df, 5, span)
    print("H={:.4f}, c={:.4f}".format(H5, c5))
    H7, c7, cls_data7 = evaluate_HC(c_data_df, 7, span)
    print("H={:.4f}, c={:.4f}".format(H7, c7))
    H10, c10, cls_data10 = evaluate_HC(c_data_df, 10, span)
    print("H={:.4f}, c={:.4f}".format(H10, c10))

    r_data_df = project_forward(c_data_df, span)

    for i in range(1000):
        col_name = 'sim' + str(i+1)
        sim_list = r_data_df[col_name].tolist()

        cls_data3 = cls_data3[:len(cls_data3) - len(sim_list)]
        cls_data3.extend(sim_list)
        simH3, simC3, data3 = compute_Hc(cls_data3, kind='price', simplified=True)

        cls_data5 = cls_data5[:len(cls_data5) - len(sim_list)]
        cls_data5.extend(sim_list)
        simH5, simC5, data5 = compute_Hc(cls_data5, kind='price', simplified=True)

        cls_data7 = cls_data7[:len(cls_data7) - len(sim_list)]
        cls_data7.extend(sim_list)
        simH7, simC7, data7 = compute_Hc(cls_data7, kind='price', simplified=True)

        cls_data10 = cls_data10[:len(cls_data10) - len(sim_list)]
        cls_data10.extend(sim_list)
        simH10, simC10, data10 = compute_Hc(cls_data10, kind='price', simplified=True)

        if abs(H3-simH3) < 0.001 and abs(H5-simH5) < 0.01 and abs(H7-simH7) < 0.01 and abs(H10-simH10) < 0.01:
            print("diff H3={:.4f}, H5={:.4f} H7={:.4f}, H10={:.4f} ".format(H3 - simH3, H5 - simH5, H7 - simH7,
                                                                            H10 - simH10))
        else:
            r_data_df.drop([col_name], axis=1, inplace=True)

    r_data_df.to_csv('data/r_data_df.csv', index=False)

    '''
    r_data_df.set_index('Date')
    plt.plot(r_data_df['sim_values'], linestyle='--', color='blue')
    plt.plot(r_data_df['Close'], linestyle='--', color='black')

    plt.show();
    r_data_df.to_csv('data/r_data_df.csv', index=False)
    # print(r_data_df.iloc[:10].to_string())
    '''