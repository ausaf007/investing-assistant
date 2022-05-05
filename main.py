# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import yfinance as yf
import streamlit as st

st.write("""
# Investing Assistant 

""")

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbol = 'GOOGL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

# st.line_chart(tickerDf.Close)
# st.line_chart(tickerDf.Volume)


import csv
import json
import time
from datetime import datetime
from enum import Enum
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.pyplot import figure
import pandas as pd
# Gym stuff
import gym
import gym_anytrading
# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gym_anytrading.envs import StocksEnv


data = pd.read_csv("btc_ind_2016-2021.csv")    # Change path here to data csv

data = data.rename(columns= {'low':'Low'})

data = data.drop(['Unnamed: 0', 'time'], axis = 1)
data.set_index('Date', inplace=True)

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'volume', 'BMSB', 'SMA', 'RSI', 'Sentiment']].to_numpy()[start:end]
    return prices, signal_features


class MyCustomEnv(StocksEnv):
    _process_data = add_signals


model1 = A2C.load("A2C_mlp_policy_sb3.zip")   # change path here to zipped model


env = MyCustomEnv(df=data[157:], window_size=30, frame_bound=(30,500))
obs = env.reset()
while True:
    # obs = obs[np.newaxis, ...]
    action, _states = model1.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

        
st.line_chart(data.reset_index()['volume'][157:])
st.line_chart(data.reset_index()['SMA'][157:])

# plt.figure(figsize=(15,6))
# plt.cla()
# env.render_all()
# plt.show()

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)
#
# st.pyplot(fig)



window_ticks = np.arange(len(env._position_history))
st.line_chart(env.prices)


short_ticks = []
long_ticks = []
for i, tick in enumerate(window_ticks):
    if str(env._position_history[i]) == "Positions.Short":
        # print('short')
        short_ticks.append(tick)
    elif str(env._position_history[i]) == "Positions.Long":
        # print('long')
        long_ticks.append(tick)

fig=plt.figure(figsize = (10, 5))

plt.plot(short_ticks, env.prices[short_ticks], 'ro')
plt.plot(long_ticks, env.prices[long_ticks], 'go')
plt.plot(env.prices)


st.pyplot(fig)

plt.suptitle(
    "Total Reward: %.6f" % env._total_reward + ' ~ ' +
    "Total Profit: %.6f" % env._total_profit
)
trade=False
if ((action == 1 and env._position.value == 0) or
            (action == 0 and env._position.value == 1)):
            trade = True

# position=0, Short
# position=1, Long

# Action=1 buy
# Action=0 sell

if(trade):
    if(action==0):
        st.write("# Recommended Action: Sell")
    if(action==1):
        st.write("# Recommended Action: Buy")
else:
    st.write("# Recommended Action: Hodl")


