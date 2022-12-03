import csv
import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dateutil.parser import parser
from numpy import random
import datetime as dt
from scipy import stats
import statistics
from hurst import compute_Hc

import mlmodel
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
    data_df = c_data_df.iloc[-span:]
    r_data_df = c_data_df.iloc[-span:]
    r_data_df['Date'] = data_df['Date'] + datetime.timedelta(days=365)
    r_data_df['Close'] = 0

    print("base data begin={:s}, end={:s}".format(str(data_df.iloc[0]['Date']), str(data_df.iloc[-1]['Date'])))
    close_val = data_df.iloc[-1]['Close']
    print("last Close={:.4f}".format(close_val))

    c_data_df['diff'] = c_data_df['Close'].diff()
    c_data_df['trend'] = np.where(c_data_df['diff'] > 0, 1, 0)

    value_list = c_data_df['diff'].tolist()#data_df['Close'].diff().tolist()
    value_list = [0 if math.isnan(x) else x for x in value_list]
    value_list = set(value_list)
    # convert the set to the list
    value_list = (list(value_list))

    threshold = int(c_data_df['trend'].sum() / c_data_df['trend'].count() * 100)
    iterations = r_data_df['Close'].count()
    ret_data_df = run_simuations(1000, iterations, threshold, value_list, r_data_df, close_val)
    return threshold, ret_data_df


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


def project_close_values(c_data_df, fut_cls_df, span=250, topN = 25):
    print("data_df beg={:s}, data_df end={:s}".format(str(c_data_df.iloc[0]['Date']), str(c_data_df.iloc[-1]['Date'])))
    base_vals = []
    H3, c3, cls_data3, stddev3, slp3 = evaluate_HC(c_data_df, 3, span)
    H5, c5, cls_data5, stddev5, slp5 = evaluate_HC(c_data_df, 5, span)
    H7, c7, cls_data7, stddev7, slp7 = evaluate_HC(c_data_df, 7, span)
    H10, c10, cls_data10, stddev10, slp10 = evaluate_HC(c_data_df, 10, span)

    base_vals.extend([H3, H5, H7, H10])
    base_vals.extend([stddev3, stddev5, stddev7, stddev10])
    base_vals.extend([slp3, slp5, slp7, slp10])

    threshold, r_data_df = project_forward(c_data_df, span)
    base_vals.extend([threshold / 100])

    sim_vals_dict = {}
    simHmap = {}

    for i in range(1000):
        sim_vals_list = []
        col_name = 'sim' + str(i + 1)
        sim_list = r_data_df[col_name].tolist()

        cls_data3 = cls_data3[:len(cls_data3) - len(sim_list)]
        cls_data3.extend(sim_list)
        simH3, simC3, data3 = compute_Hc(cls_data3, kind='price', simplified=True)
        cls_stddev3 = statistics.stdev(cls_data3)
        cls_slp3, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data3)), cls_data3)
        sim_vals_list.extend([simH3, cls_stddev3, cls_slp3])

        cls_data5 = cls_data5[:len(cls_data5) - len(sim_list)]
        cls_data5.extend(sim_list)
        simH5, simC5, data5 = compute_Hc(cls_data5, kind='price', simplified=True)
        cls_stddev5 = statistics.stdev(cls_data5)
        cls_slp5, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data5)), cls_data5)
        sim_vals_list.extend([simH5, cls_stddev5, cls_slp5])

        cls_data7 = cls_data7[:len(cls_data7) - len(sim_list)]
        cls_data7.extend(sim_list)
        simH7, simC7, data7 = compute_Hc(cls_data7, kind='price', simplified=True)
        cls_stddev7 = statistics.stdev(cls_data7)
        cls_slp7, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data7)), cls_data7)
        sim_vals_list.extend([simH7, cls_stddev7, cls_slp7])

        cls_data10 = cls_data10[:len(cls_data10) - len(sim_list)]
        cls_data10.extend(sim_list)
        simH10, simC10, data10 = compute_Hc(cls_data10, kind='price', simplified=True)
        cls_stddev10 = statistics.stdev(cls_data10)
        cls_slp10, intercept, r_value, p_value, std_err = stats.linregress(range(len(cls_data10)), cls_data10)
        sim_vals_list.extend([simH10, cls_stddev10, cls_slp10])

        sim_vals_dict[col_name] = sim_vals_list

        simHmap[col_name] = np.mean([abs(H3 - simH3), abs(H5 - simH5), abs(H7 - simH7), abs(H10 - simH10),
                                     abs(slp3 - cls_slp3), abs(slp5 - cls_slp5), abs(slp7 - cls_slp7),
                                     abs(slp10 - cls_slp10)])

    simHmap = dict(sorted(simHmap.items(), key=lambda item: item[1]))
    for key in simHmap:
        if topN < 0:
            r_data_df.drop([key], axis=1, inplace=True)
            del sim_vals_dict[key]
        topN = topN - 1

    r_data_df.set_index('Date', inplace=True)
    prev_close_df = c_data_df.iloc[-span:]
    r_data_df_copy = r_data_df.copy()
    r_data_df_copy['Close'] = np.array(prev_close_df['Close']).tolist()
    cls_data = r_data_df_copy['Close'].tolist()
    cls_slope, inter, r_val, p_val, std_err = stats.linregress(range(len(cls_data)), cls_data)
    cls_std = statistics.stdev(r_data_df_copy['Close'].tolist())

    cols = r_data_df_copy.columns.values
    for col in cols:
        if col not in 'Close':
            sim_data = r_data_df_copy[col].tolist()
            sim_std = statistics.stdev(sim_data)
            cls_sim_slp, inter, r_val, p_val, std_err = stats.linregress(range(len(sim_data)), sim_data)
            sim_vals_dict[col].extend([cls_slope, cls_sim_slp, cls_std, sim_std])

    corr_results = plot_corr(r_data_df_copy, 20, 50)

    fut_cls_arr = fut_cls_df['Close'].to_numpy()
    cls_list = r_data_df['Close'].tolist()
    fut_cls_arr = np.append(fut_cls_arr, cls_list[len(fut_cls_arr):len(cls_list)], 0)
    r_data_df['Close'] = fut_cls_arr.tolist()

    for idx in corr_results.index.values:
        if idx not in 'Close':
            corr_20 = corr_results.at[idx, 'corr_20']
            corr_50 = corr_results.at[idx, 'corr_50']
            sim_vals_dict[idx].extend([corr_20, corr_50])
            flag = 0
            if r_data_df.iloc[-1]['Close'] > 0:
                perc_diff = abs((r_data_df.iloc[-1]['Close'] - r_data_df.iloc[-1][idx]) / r_data_df.iloc[-1]['Close'])
                if perc_diff < 0.03:
                    flag = 1
            sim_vals_dict[idx].extend([flag])

    r_data_df.to_csv('data/r_data_df.csv', index=True)
    return base_vals, sim_vals_dict


INDEX_FILES = ['sp500', 'sp400', 'sp100', 'sp1000', 'spbse100']
BASE_COLS = ['h3', 'h5', 'h7', 'h10', 'std3', 'std5', 'std7', 'std10', 'slp3', 'slp5', 'slp7', 'slp10',
             'thresh']

SIM_COLS = ['simH3', 'sim_std3', 'sim_slp3', 'simH5', 'sim_std5', 'sim_slp5', 'simH7', 'sim_std7', 'sim_slp7',
            'simH10', 'sim_std10', 'sim_slp10', 'cls_slope', 'cls_sim_slp', 'cls_std', 'sim_std',
            'corr_20', 'corr_50', 'flag']
BASE_COLS.extend(SIM_COLS)
DATE_VAL = ['03', '07', '11', '16', '20', '25', '29']
MONTH_VAL = ['01', '03', '05', '07', '09', '12', '06']


def prepare_simulation_data(idxname):
    print(idxname+' prepare_simulation_data...')

    with open('data/'+idxname+'_feature.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(BASE_COLS)

        c_data_df = load_data(idxname + '.csv')
        c_data_df.rename(columns=lambda x: x.strip(), inplace=True)
        c_data_df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)

        c_data_df['Date'] = pd.to_datetime(c_data_df.Date)
        c_data_df = c_data_df.sort_values('Date')

        y = 1
        for m in range(12):
            str_date = "{:02d}".format(m + 1)
            str_date = str_date + '/' + "{:02d}".format(m * 2 + 1)
            str_date = str_date + '/' + str(2016 + y % 6)
            y += 1

            print(idxname + ':' + str_date)
            _date_end = datetime.datetime.strptime(str_date, "%m/%d/%Y").date()
            _date_end = _date_end - datetime.timedelta(days=365)
            print(_date_end.strftime("%m/%d/%Y"))
            _data_df = c_data_df[c_data_df['Date'] < _date_end.strftime("%m/%d/%Y")]
            fut_cls_df = c_data_df[c_data_df['Date'] >= _date_end.strftime("%m/%d/%Y")].head(250)
            base_vals, sim_vals_dict = project_close_values(_data_df, fut_cls_df, topN=500)

            for key in sim_vals_dict:
                line_arr = []
                line_arr.extend(base_vals)
                line_arr.extend(sim_vals_dict[key])
                wr.writerow(line_arr)
        print(idxname + ' features completed')

        myfile.close()


def test_simulations(idxname, str_date):
    print('test_simulations...'+idxname)
    test_file = idxname+'_test.csv'
    with open('data/'+test_file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(BASE_COLS)
        c_data_df = load_data(idxname + '.csv')
        c_data_df.rename(columns=lambda x: x.strip(), inplace=True)
        c_data_df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)

        c_data_df['Date'] = pd.to_datetime(c_data_df.Date)
        c_data_df = c_data_df.sort_values('Date')

        _date_end = datetime.datetime.strptime(str_date, "%m/%d/%Y").date()
        _date_end = _date_end - datetime.timedelta(days=365)
        print(_date_end.strftime("%m/%d/%Y"))
        _data_df = c_data_df[c_data_df['Date'] < _date_end.strftime("%m/%d/%Y")]
        fut_cls_df = c_data_df[c_data_df['Date'] >= _date_end.strftime("%m/%d/%Y")].head(250)
        base_vals, sim_vals_dict = project_close_values(_data_df, fut_cls_df, topN=200)

        # rs_map[str_date] = results
        for key in sim_vals_dict:
            line_arr = []
            line_arr.extend(base_vals)
            line_arr.extend(sim_vals_dict[key])
            wr.writerow(line_arr)
        print('file completed')

        myfile.close()

    _data_df = load_data(test_file)
    print(_data_df.shape)
    dataset = _data_df.values
    Y = dataset[:, 31]

    confidences = mlmodel.predict_nn(_data_df, idxname)
    _data_df['confidence'] = confidences
    r_data_df = load_data('r_data_df.csv')
    close_val = r_data_df.iloc[-1]['Close']

    cols = r_data_df.columns.values
    str_format = "sim={:s}, actual={:s} pred={:s} perc_diff={:f}"
    i = 0
    max_conf = max(confidences)
    for col in cols:
        if 'sim' in col:
            sim_val = r_data_df.iloc[-1][col]
            if confidences[i] < 0.55 and confidences[i] < max_conf:
                r_data_df.drop([col], axis=1, inplace=True)
            else:
                print(str_format.format(col, str(Y[i]), str(confidences[i]), (close_val - sim_val)/close_val))
            i = i + 1

    r_data_df.to_csv('data/nn_'+test_file, index=False)
    _data_df.to_csv('data/'+test_file, index=False)


if __name__ == '__main__':
    str_date = '03/15/2018'  # 22'

    for idxname in INDEX_FILES:
        #prepare_simulation_data(idxname)
        #mlmodel.build_save_nn(idxname+'_feature')
        test_simulations(idxname, str_date)

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
