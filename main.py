import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

st.title("Investing Assistant")



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


data = pd.read_csv("btc_ind_updated.csv")    # Change path here to data csv

data = data.rename(columns= {'low':'Low'})
    
def calculate_ema(prices, window_size, smoothing=2):
    ema = [sum(prices[:window_size]) / window_size]
    for price in prices[window_size:]:
        ema.append((price * (smoothing / (1 + window_size))) + ema[-1] * (1 - (smoothing / (1 + window_size))))
    # emaList = ['nan' for i in range(window_size-1)]
    # emaList = emaList + ema
    return ema

ema = calculate_ema(data['close'], 21*7)
data = data.drop(['Unnamed: 0', 'time'], axis = 1)
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
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

trade=False
if ((action == 1 and env._position.value == 0) or
            (action == 0 and env._position.value == 1)):
            trade = True

if(trade):
    if(action==0):
        st.write("# Recommended Action: SELL")
    if(action==1):
        st.write("# Recommended Action: BUY")
else:
    st.write("# Recommended Action: HOLD") 

window_ticks = np.arange(len(env._position_history)) 
short_ticks = []
long_ticks = []
for i, tick in enumerate(window_ticks):
    if str(env._position_history[i]) == "Positions.Short":
        # print('short')
        short_ticks.append(tick)
    elif str(env._position_history[i]) == "Positions.Long":
        # print('long')
        long_ticks.append(tick)



fig=plt.figure()
#figsize = (10, 5)
plt.plot(env.prices)
plt.plot(short_ticks, env.prices[short_ticks], 'ro')
plt.plot(long_ticks, env.prices[long_ticks], 'go')
plt.grid()
plt.xlabel('Number of Days')
plt.ylabel('Price of Bitcoin')
plt.suptitle(
    "Total Reward: %.6f" % env._total_reward + ' ~ ' +
    "Total Profit: %.6f" % env._total_profit
)
# plt.suptitle(
#     "Total Reward: 30124 ~ " +
#     "Total Profit: 1.6912"
# )
st.pyplot(fig)

st.write("# Technical Analysis: ")

st.subheader("1. Closing Prices")
fig2 = plt.figure(dpi=600) 
f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4) #axisbg='#07000d')
plt.plot(data['close'][157:])
# f1.plot(btc_window['Date'], btc_window['close'])
f1.xaxis_date()
f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('Prices')
plt.xlabel('Date')
plt.title("Closing Prices")
plt.grid()
plt.show() 
st.pyplot(fig2)

st.subheader("2. Volume")
fig3 = plt.figure(dpi=600) 
f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4) #axisbg='#07000d')
plt.plot(data['volume'][157:])
# f1.plot(btc_window['Date'], btc_window['close'])
f1.xaxis_date()
f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('Volume')
plt.xlabel('Date')
# plt.title("Sentiment Analysis")
plt.grid()
plt.show() 
st.pyplot(fig3)

st.subheader("3. Simple Moving Average")
fig4 = plt.figure(dpi=600) 
f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4) #axisbg='#07000d')
plt.plot(data['SMA'][157:])
# f1.plot(btc_window['Date'], btc_window['close'])
f1.xaxis_date()
f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('Prices')
plt.xlabel('Date')
plt.title("Simple Moving Average")
plt.grid()
plt.show() 
st.pyplot(fig4)

st.subheader("4. Exponential Moving Average")
fig6 = plt.figure(dpi=600) 
f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4) #axisbg='#07000d')
plt.plot(data.index.to_list()[146:], ema)
# f1.plot(btc_window['Date'], btc_window['close'])
f1.xaxis_date()
f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('Prices')
plt.xlabel('Date')
plt.title("Exponential Moving Average")
plt.grid()
plt.show() 
st.pyplot(fig6)

st.subheader("5. Relative Strength Index")
fig5 = plt.figure(dpi=600) 
f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4) #axisbg='#07000d')
plt.plot(data['RSI'][157:])
# f1.plot(btc_window['Date'], btc_window['close'])
f1.xaxis_date()
f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('RSI')
plt.xlabel('Date')
# plt.title("Sentiment Analysis")
plt.grid()
plt.show() 
st.pyplot(fig5)

st.subheader("6. Sentiment Scores")
fig1 = plt.figure(dpi=600) 
f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4) #axisbg='#07000d')
plt.plot(data['Sentiment'][157:])
# f1.plot(btc_window['Date'], btc_window['close'])
f1.xaxis_date()
f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.xticks(rotation=45)
plt.ylabel('Sentiment Scores')
plt.xlabel('Date')
plt.title("Sentiment Analysis")
plt.grid()
plt.show() 
st.pyplot(fig1)



## ACtual plots 
# st.line_chart(data['close'])
# st.line_chart(data.reset_index()['volume'][157:])
# st.line_chart(data.reset_index()['SMA'][157:])
# st.line_chart(data.reset_index()['RSI'][157:])
# st.line_chart(data.reset_index()['Sentiment'][157:])




# plt.figure(figsize=(15,6))
# plt.cla()
# env.render_all()
# plt.show()

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)
#
# st.pyplot(fig)
st.subheader("7. Bull Market Support Band")
x = data['BMSB'].iloc[-1]
# st.write(x)

if -90< x <-50:
  label = "Moderately Undervalued"
elif  -50< x <0:
  label = "Undervalued"
elif  0< x <50:
  label = "Overvalued"
elif  50< x <90:
  label = "Moderately Overvalued"
elif  90< x <100:
  label = "Strongly Overvalued"
elif  -90< x <-100:
  label = "Strongly Undervalued"


fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = x,
    name = label,
    mode = "gauge+number+delta",
    title = {'text': label},
    # delta = {'reference': 380},
    
    gauge = {'axis': {'range': [-100, 100]},
             'steps' : [
                 {'range': [-90, -50], 'color': "lightgray"},
                 {'range': [-50, 0], 'color': "lightgray", 'name': "Undervalued"},
                {'range': [0, 50], 'color': "lightgray", 'name': "Overvalued"},
                {'range': [50, 90], 'color': "lightgray", 'name': "Moderately Overvalued"},
                 {'range': [-100, -90], 'color': "darkred", 'name': "Strongly Undervalued" },
                 {'range': [90, 100], 'color': "darkblue", 'name': "Strongly Overvalued"}],
             }))
st.plotly_chart(fig, use_container_width=True)





