import csv
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numpy import random

# Game 1
simulations = 10000  # number of Monte Carlo Simulations
games = 100  # number of times the game is played
threshold = 40  # threshold where if greater than or equal to you win
bet = 1  # dollar bet for the game

DATA_FOLDER = 'data'

def load_data(filename):
    _data_df = pd.read_csv(DATA_FOLDER + '\\' + filename)
    return _data_df

def run_simuations(simulations, iterations, threshold, value_list):
    # outer loop is Monte Carlo sims and inner loop is games played
    print(simulations, iterations, threshold)
    index_value = []
    for g in range(iterations):
        sim_results = []

        for sim in range(simulations):

            random_num = abs(random.choice(value_list))

            if math.isnan(random_num):
                random_num = 0
            sim_results.append(random_num)

        sim_rand_val = abs(np.mean(sim_results))
        number = int(np.random.uniform() * 100)  # get a random number to see who wins
        if number > threshold:
            index_value.append(-sim_rand_val)
        else:
            index_value.append(sim_rand_val)

    return np.array(index_value).tolist()


if __name__ == '__main__':

    print('Hello...')
    c_data_df = load_data('sp1000.csv')
    c_data_df.rename(columns=lambda x: x.strip(), inplace=True)

    c_data_df['Date'] = pd.to_datetime(c_data_df.Date)
    data_df = c_data_df[c_data_df['Date'] < '01/01/2021']

    data_df = data_df.sort_values('Date')
    data_df['diff'] = data_df['Close'].diff()
    data_df['trend'] = np.where(data_df['diff'] > 0, 1, 0)
    print(data_df.shape)

    close_val = data_df.iloc[-1]['Close']

    value_list = data_df['diff'].tolist()
    value_list = [0 if math.isnan(x) else x for x in value_list]
    threshold = int(data_df['trend'].sum()/data_df['trend'].count() * 100)

    r_data_df = c_data_df[c_data_df['Date'] > '12/31/2020']
    r_data_df = r_data_df.sort_values('Date')
    iterations = r_data_df['Date'].count()

    sim_list = run_simuations(1000, iterations, threshold, value_list)
    sim_list_df = pd.DataFrame(sim_list)
    r_data_df['sim1'] = sim_list_df
    print(r_data_df.iloc[:10].to_string())

    sim_list = run_simuations(1000, iterations, threshold, value_list)
    sim_list_df = pd.DataFrame(sim_list)
    r_data_df['sim2'] = sim_list_df
    print(r_data_df.iloc[:10].to_string())

    sim_list = run_simuations(1000, iterations, threshold, value_list)
    sim_list_df = pd.DataFrame(sim_list)
    r_data_df['sim3'] = sim_list_df
    print(r_data_df.iloc[:10].to_string())

    #print(r_data_df.iloc[:10].to_string())

    r_data_df.to_csv('data/r_data_df.csv', index=False)
