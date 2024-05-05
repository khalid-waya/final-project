import yfinance as yf
import gymnasium as gym
import gym_anytrading
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, DQN
import matplotlib
import pandas as pd
import numpy as np


# %%

def fetch_stock_data(ticker_symbol, start_date, end_date, interval):
    """
    Fetches historical stock data for a given ticker symbol and date range.

    Args:
        ticker_symbol (str): The ticker symbol of the stock.
        start_date (str): The start date of the date range (format: "YYYY-MM-DD").
        end_date (str): The end date of the date range (format: "YYYY-MM-DD").

    Returns:
        pandas.DataFrame: The fetched stock data as a DataFrame.
    """
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
    return stock_data


# Specify the ticker symbol and the date range
ticker_symbol = "AAPL"
start_date = "2024-04-18"
end_date = "2024-04-24"

# Fetch historical data
stock_data = fetch_stock_data(ticker_symbol, start_date, end_date, interval="5m")

# %%
stock_data.dtypes


# %%

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# %%
env = gym.make('stocks-v0', df=stock_data, frame_bound=(10, 100), window_size=10)
# %%
state = env.reset()
while True:

    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print("info", info)
        break
    # %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
plt.cla()
env.render_all()
plt.show()
# %%
print("Data columns:", stock_data.columns)
env_maker = lambda: gym.make('stocks-v0', df=stock_data, frame_bound=(10, 100), window_size=5)
env = DummyVecEnv([env_maker])

# %%
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
# %%
env = gym.make('stocks-v0', df=stock_data, frame_bound=(90, 160), window_size=5)
obs, _ = env.reset()
# %%
obs
# %%
while True:
    try:

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print("info", info)
            break
    except TypeError as e:
        print(f"TypeError occurred: {str(e)}")
        print("Please ensure that 'obs' is a numpy array or can be indexed with 'np.newaxis'.")
        break
    except ValueError as e:
        print(f"ValueError occurred: {str(e)}")
        print("Please ensure that 'obs' has the correct shape and dimensions.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        break
    # %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
plt.cla()
env.render_all()
plt.show()
# %%
model.save("dqn_model")
# %%
dq = model.load("dqn_model")