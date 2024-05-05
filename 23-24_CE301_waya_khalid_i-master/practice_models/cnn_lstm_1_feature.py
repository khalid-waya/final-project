import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from scikeras.wrappers import KerasRegressor
# from sklearn.model_selection import GridSearchCV
from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Reshape
from keras.layers import MaxPooling1D, Flatten
from keras.models import Sequential
from keras.metrics import MeanSquaredError, MeanAbsoluteError
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss



# Specify the ticker symbol and the date range
ticker_symbol = "AAPL"
start_date = "2020-01-01"
end_date = "2024-02-20"

# Fetch historical data
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Ensure the index is a DateTimeIndex
if not isinstance(stock_data.index, pd.DatetimeIndex):
    stock_data.index = pd.to_datetime(stock_data.index)


# Fill in missing dates
all_dates = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='B')
stock_data = stock_data.reindex(all_dates)

# Fill missing values. You can choose different methods like 'ffill' or 'bfill'
stock_data.fillna(method='bfill', inplace=True)
def check_stationarity_adf(series, alpha=0.05):
    """
    Perform ADF test to check stationarity of the time series.
    Returns True if stationary, False otherwise.
    """
    result = adfuller(series)
    p_value = result[1]
    return p_value < alpha

def check_stationarity_kpss(series, alpha=0.05):
    """
    Perform KPSS test to check stationarity of the time series.
    Returns True if stationary, False otherwise.
    """
    statistic, p_value, _, _ = kpss(series, regression='c')
    return p_value > alpha


def decompose_series(series, frequency, model='additive'):
    """
    Decompose a time series into trend, seasonal, and residual components.
    Assumes 'dates' is a pandas Series with the same index as 'series', containing datetime objects.
    'frequency' is the frequency of the time series (e.g., 'B' for business days).
    """
    # Create a new time series with the dates as the index
    series_with_date_index = pd.Series(data=series.values, index=pd.DatetimeIndex(dates))

    # Set the frequency of the DatetimeIndex
    series_with_date_index = series_with_date_index.asfreq(frequency)

    # Perform decomposition
    decomposition = sm.tsa.seasonal_decompose(series_with_date_index.dropna(), model=model)
    return decomposition


def add_rolling_std_feature(stock_data, window_size=20):
    """
    Adds rolling standard deviation as a feature to the stock data.

    Args:
    stock_data (pd.DataFrame): The DataFrame containing stock data.
    window_size (int): The window size for calculating rolling standard deviation.

    Returns:
    pd.DataFrame: Updated DataFrame with the rolling standard deviation feature.
    """
    # Calculate rolling standard deviation (volatility)
    stock_data['Rolling_Std'] = stock_data['Close'].rolling(window=window_size).std()

    # Shift the rolling standard deviation to align with the prediction target
    stock_data['Rolling_Std'] = stock_data['Rolling_Std'].shift(-1)

    # Drop NaN values created by rolling function
    return stock_data.dropna()


def create_model(optimizer='adam', lstm_units=100, dropout_rate=0.5):
    model = Sequential()

    # CNN layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(100, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())

    # LSTM layers
    model.add(Dense(100))
    model.add(Reshape((-1, 100)))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))
    model.add(Dropout(dropout_rate))

    # Final layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

    return model


# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate MACD
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


stock_data["Date"] = stock_data.index

# Convert the 'Date' column to a DateTimeIndex
dates = pd.to_datetime(stock_data["Date"]).dropna()

# Set the converted 'Date' as the index of the DataFrame
stock_data.index = pd.DatetimeIndex(dates)

# Prepare the 'Close' prices for the ADF and KPSS tests
closing_prices = stock_data['Close']

# Check if the series is stationary using both ADF and KPSS tests
is_stationary_adf = check_stationarity_adf(closing_prices)
is_stationary_kpss = check_stationarity_kpss(closing_prices)

if is_stationary_adf and is_stationary_kpss:
    print("The series is likely stationary. Skipping decomposition.")
else:
    print("The series is not stationary, therefore may have a trend. Applying decomposition.")
    # Decompose the time series
    decomposition = decompose_series(closing_prices, model='additive', frequency="B")
    # Plot the decomposition components

    # Add rolling standard deviation feature to the stock data
    stock_data = add_rolling_std_feature(stock_data, window_size=21)

    # decomposition.plot()
    # plt.show()


# Calculate daily returns
stock_data['Daily Return'] = stock_data['Close'].pct_change()

# Calculate moving averages
ma_days = [10, 50, 100]
for ma in ma_days:
    col_name = "MA for {} days".format(str(ma))
    stock_data[col_name] = pd.DataFrame.rolling(stock_data['Close'], ma).mean()

# Add RSI and MACD features
stock_data['RSI'] = calculate_rsi(stock_data)
macd_line, signal_line, macd_hist = calculate_macd(stock_data)
stock_data['MACD'] = macd_line
stock_data['MACD_Signal'] = signal_line
stock_data['MACD_Hist'] = macd_hist
print(stock_data.info())


def normalize_features(stock_data, window_size=100):
    """
    Normalizes selected features using a rolling window approach.

    Args:
    stock_data (pd.DataFrame): DataFrame containing stock data with 'Close', 'Rolling_Std', and moving average columns.
    window_size (int): Size of the rolling window for normalization.

    Returns:
    pd.DataFrame: DataFrame with normalized features.
    """
    # Ensure 'Close', 'Rolling_Std', and the moving average columns exist
    required_columns = ['Close', 'MA for 50 days', "RSI","Daily Return", "MACD"]  # Add other MA columns if necessary
    for col in required_columns:
        if col not in stock_data.columns:
            raise ValueError(f"Missing required column: {col}")

    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(index=stock_data.index, columns=required_columns)

    # Apply normalization within each rolling window
    for start in range(len(stock_data) - window_size):
        end = start + window_size
        window_data = stock_data.iloc[start:end][required_columns]
        normalized_data.iloc[start:end] = scaler.fit_transform(window_data)

    # Drop rows with NaN values that might have been created due to rolling
    normalized_data.dropna(inplace=True)

    return normalized_data


normalised_stock_data = stock_data


# Normalize the 'Daily Return' using MinMaxScaler
daily_return_scaler = MinMaxScaler()
normalized_daily_return = daily_return_scaler.fit_transform(stock_data[['Daily Return']])

X, Y = [], []
window_size = 100

# Create the input sequences and target values using the daily return
for i in range(len(normalized_daily_return) - window_size - 1):
    X.append(normalized_daily_return[i:i+window_size])
    Y.append(normalized_daily_return[i+window_size])

X = np.array(X)
Y = np.array(Y)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# Assuming your CNN-LSTM model expects inputs shaped as (samples, time steps, features)
# No need to reshape X as it should already be in the correct shape
train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_Y = train_Y.astype('float32')
test_Y = test_Y.astype('float32')

# Check and replace NaNs or infinite values
train_X = np.nan_to_num(train_X)
test_X = np.nan_to_num(test_X)
train_Y = np.nan_to_num(train_Y)
test_Y = np.nan_to_num(test_Y)

train_X = train_X.reshape(train_X.shape[0],100,1)
test_X = test_X.reshape(test_X.shape[0],100,1)


# Define the grid search parameters
# Create the model with pre-defined hyperparameters
model = create_model(optimizer='adam', lstm_units=100, dropout_rate=0.5)
# Train the model with the best hyperparameters
# Train the model
history = model.fit(
    train_X, train_Y,
    validation_data=(test_X, test_Y),
    epochs=40,
    batch_size=64,
    verbose=1,
    shuffle=True
)

# ...



# Make predictions on the test data
predicted_returns = model.predict(test_X)

# Reshape the predicted returns
predicted_returns = predicted_returns.flatten()

# Inverse transform the predicted returns using the daily return scaler
denormalized_predicted_returns = daily_return_scaler.inverse_transform(predicted_returns.reshape(-1, 1))[:, 0]

# Inverse transform the test returns using the daily return scaler
denormalized_test_returns = daily_return_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]

# mse = MeanAbsoluteError()
# mae = MeanAbsoluteError()
#
# mse.update_state(y_test, predicted_prices)
# mae.update_state(y_test, predicted_prices)
#
#
#
# # Get the result of the MSE and MAE
# mse_result = mse.result().numpy()
# mae_result = mae.result().numpy()
#
#
# print(f'MSE: {mse_result}')
# print(f'MAE: {mae_result}')
#
# # Reshape the predicted prices
# predicted_prices = predicted_prices.flatten()
#
# # Reshape the predicted prices
# # Inverse transform the predicted prices using the close scaler
# denormalized_predicted_prices = close_scaler.inverse_transform(predicted_prices.reshape(-1, 1))[:, 0]
#
# # Inverse transform the test prices using the close scaler
# denormalized_test_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]


# Create a new DataFrame to store the actual prices, predicted prices, and test prices
# Create a new DataFrame to store the actual returns, predicted returns, and test returns
result_df = pd.DataFrame({
    'Actual Return': stock_data.iloc[-len(y_test):]['Daily Return'],
    'Predicted Return': denormalized_predicted_returns.flatten(),
    'Test Return': denormalized_test_returns.flatten()
}, index=stock_data.index[-len(y_test):])


# Plot the actual prices, predicted prices, and test prices
plt.figure(figsize=(12, 6))

# Plot the actual returns, predicted returns, and test returns
plt.figure(figsize=(12, 6))

plt.plot(result_df['Predicted Return'], label='Predicted Return')
plt.plot(result_df['Test Return'], label='Test Return')
plt.title('Apple Stock Daily Return Prediction with 1 feature')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()
# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss  With 1 Features')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation mean squared error (MSE)
plt.figure(figsize=(12, 6))
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.title('Training and Validation Mean Squared Error with 1 feature')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot the training and validation mean absolute error (MAE)
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation Mean Absolute Error with 1 feature')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
# # Building the model
# model = Sequential()
#
# # CNN layers
# model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(100, 5)))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(128, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(64, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(2))
# model.add(Flatten())
#
# # LSTM layers
# model.add(Dense(100, activation='relu'))
# model.add(Reshape((-1, 100)))
# model.add(Bidirectional(LSTM(100, return_sequences=True)))
# model.add(Dropout(0.5))
# model.add(Bidirectional(LSTM(100, return_sequences=False)))
# model.add(Dropout(0.5))
#
# # Final layer
# model.add(Dense(1, activation='linear'))
#
# # Compile the model
# model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
#
# # Train the model
# history = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=40, batch_size=40, verbose=1, shuffle=True)
#
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# plt.plot(history.history['mse'], label='train mse')
# plt.plot(history.history['val_mse'], label='val mse')
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
# plt.plot(history.history['mae'], label='train mae')
# plt.plot(history.history['val_mae'], label='val mae')
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()
#
#
# # Make predictions on the test data
# predicted_prices = model.predict(test_X)
#
# # Reshape the predicted prices and test prices
# predicted_prices = predicted_prices.reshape(-1)
# test_Y = test_Y.reshape(-1)
# # Unnormalize the predicted prices and test prices
# scaler = MinMaxScaler()
# scaler.fit(stock_data[['Close']])
# unnormalized_predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
# unnormalized_test_prices = scaler.inverse_transform(test_Y.reshape(-1, 1))
#
# # Create a new DataFrame to store the actual prices, predicted prices, and test prices
# result_df = pd.DataFrame({
#     'Actual Price': stock_data.iloc[-len(test_Y):]['Close'],
#     'Predicted Price': unnormalized_predicted_prices.flatten(),
#     'Test Price': unnormalized_test_prices.flatten()
# }, index=stock_data.index[-len(test_Y):])
#
# plt.figure(figsize=(12, 6))
# plt.plot(result_df['Actual Price'], label='Actual Price')
# plt.plot(result_df['Predicted Price'], label='Predicted Price')
# plt.plot(result_df['Test Price'], label='Test Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# model.summary()
#
# # plt.figure(figsize=(10, 6))
# # plt.plot(actual_prices, label='Actual Price')
# # plt.plot(predicted_prices, label='Predicted Price')
# # plt.title('Stock Price Prediction')
# # plt.xlabel('Time')
# # plt.ylabel('Price')
# # plt.legend()
# plt.show()
#%%
