import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf



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

def fetch_stock_data(ticker_symbol, start_date, end_date, interval):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
    return stock_data

def fetch_stock_data_for_trading(ticker_symbol, start_date, end_date):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return stock_data
def add_date_column(input_stock_data):
    input_stock_data["Date"] = input_stock_data.index
    return input_stock_data

def prepare_dataframe_for_lstm(df, n_steps):
    df = df.copy()
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

def prepare_data(input_scaled_data, input_lookback, split_ratio=0.95):
    X = input_scaled_data[:, 1:]
    y = input_scaled_data[:, 0]
    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    X_train = X_train.reshape((-1, input_lookback, 1))
    X_test = X_test.reshape((-1, input_lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    return X_train, X_test, y_train, y_test

def preprocess_unseen_data_for_trading(ticker_symbol, start_date, end_date, lookback, interval):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval= interval)
    stock_data['Date'] = stock_data.index
    close_price_data = stock_data[['Date', 'Close']]
    close_price_data['Date'] = pd.to_datetime(close_price_data['Date'])
    shifted_df = prepare_dataframe_for_lstm(close_price_data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = np.flip(X, axis=1).reshape((-1, lookback, 1))
    y = y.reshape((-1, 1))
    X_unseen = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_unseen = tf.convert_to_tensor(y, dtype=tf.float32)
    return X_unseen, Y_unseen

def preprocess_unseen_data(ticker_symbol, start_date, end_date, lookback):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    close_price_data = stock_data[['Date', 'Close']]
    close_price_data['Date'] = pd.to_datetime(close_price_data['Date'])
    shifted_df = prepare_dataframe_for_lstm(close_price_data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = np.flip(X, axis=1).reshape((-1, lookback, 1))
    y = y.reshape((-1, 1))
    X_unseen = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_unseen = tf.convert_to_tensor(y, dtype=tf.float32)
    return X_unseen, Y_unseen

def preprocess_unseen_data_for_trading(ticker_symbol, start_date, end_date, lookback, interval):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval= interval)
    stock_data['Date'] = stock_data.index
    close_price_data = stock_data[['Date', 'Close']]
    close_price_data['Date'] = pd.to_datetime(close_price_data['Date'])
    shifted_df = prepare_dataframe_for_lstm(close_price_data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    X = np.flip(X, axis=1).reshape((-1, lookback, 1))
    y = y.reshape((-1, 1))
    X_unseen = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_unseen = tf.convert_to_tensor(y, dtype=tf.float32)
    return X_unseen, Y_unseen

def preprocess_unseen_data_for_automated_trading(ticker_symbol, interval):
    # Calculate the start and end dates based on the interval
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    if interval == "1m":
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    elif interval == "5m":
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    elif interval == "15m":
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    elif interval == "30m":
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    else:  # Assuming "60m" interval
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=360)).strftime("%Y-%m-%d")

    # Fetch historical data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)

    # Process the Close price data
    stock_data['Date'] = stock_data.index
    close_price_data = stock_data[['Date', 'Close']]
    close_price_data['Date'] = pd.to_datetime(close_price_data['Date'])

    # Prepare data for LSTM
    lookback = 7
    shifted_df = prepare_dataframe_for_lstm(close_price_data, lookback)
    shifted_df_as_np = shifted_df.to_numpy()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    # Split features and target
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    # Reshape data for LSTM
    X = np.flip(X, axis=1).reshape((-1, lookback, 1))
    y = y.reshape((-1, 1))

    # Convert to tensors
    X_unseen = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_unseen = tf.convert_to_tensor(y, dtype=tf.float32)

    return X_unseen, Y_unseen


def prepare_data_for_cnn_lstm(stock_data, lookback, scaler):
    # Normalize the 'Close' price using the provided scaler
    normalized_close = scaler.transform(stock_data[['Close']])

    # Prepare the data for CNN-LSTM
    X, Y = [], []
    for i in range(len(normalized_close) - lookback - 1):
        X.append(normalized_close[i:i + lookback])
        Y.append(normalized_close[i + lookback])

    X = np.array(X)
    Y = np.array(Y)

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # Convert data types
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Check and replace NaNs or infinite values
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    y_train = np.nan_to_num(y_train)
    y_test = np.nan_to_num(y_test)

    # Reshape the data for CNN-LSTM input
    x_train = x_train.reshape(x_train.shape[0], lookback, 1)
    x_test = x_test.reshape(x_test.shape[0], lookback, 1)

    return x_train, x_test, y_train, y_test


def preprocess_unseen_data_for_cnn_lstm(ticker_symbol, start_date, end_date, lookback, scaler):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Normalize the 'Close' price using the provided scaler
    normalized_close = scaler.transform(stock_data[['Close']])

    # Prepare the data for CNN-LSTM
    X, Y = [], []
    for i in range(len(normalized_close) - lookback):
        X.append(normalized_close[i:i + lookback])
        Y.append(normalized_close[i + lookback])

    X = np.array(X)
    Y = np.array(Y)

    # Convert data types
    X = X.astype('float32')
    Y = Y.astype('float32')

    # Check and replace NaNs or infinite values
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)

    # Reshape the data for CNN-LSTM input
    X = X.reshape(X.shape[0], lookback, 1)

    return X, Y


def preprocess_unseen_data_for_cnn_lstm_trading(ticker_symbol, start_date, end_date, lookback, interval, scaler):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)

    # Normalize the 'Close' price using the provided scaler
    normalized_close = scaler.transform(stock_data[['Close']])

    # Prepare the data for CNN-LSTM
    X = []
    for i in range(len(normalized_close) - lookback):
        X.append(normalized_close[i:i + lookback])

    X = np.array(X)

    # Convert data types
    X = X.astype('float32')

    # Check and replace NaNs or infinite values
    X = np.nan_to_num(X)

    # Reshape the data for CNN-LSTM input
    X = X.reshape(X.shape[0], lookback, 1)

    return X

#
# def prepare_data_for_cnn_lstm(stock_data, lookback, scaler):
#     # Normalize the data using the provided scaler
#     stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']] = scaler.transform(
#         stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']])
#
#     # Prepare the data for CNN-LSTM
#     X, Y = [], []
#     for i in range(len(stock_data) - lookback - 1):
#         X.append(stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']].iloc[i:i + lookback].values)
#         Y.append(stock_data['Close'].iloc[i + lookback])
#
#     X = np.array(X)
#     Y = np.array(Y).reshape(-1, 1)
#
#     # Split the dataset
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
#
#     # Convert data types
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     y_train = y_train.astype('float32')
#     y_test = y_test.astype('float32')
#
#     # Check and replace NaNs or infinite values
#     x_train = np.nan_to_num(x_train)
#     x_test = np.nan_to_num(x_test)
#     y_train = np.nan_to_num(y_train)
#     y_test = np.nan_to_num(y_test)
#
#     # Reshape the data for CNN-LSTM input
#     x_train = x_train.reshape(x_train.shape[0], lookback, 5)
#     x_test = x_test.reshape(x_test.shape[0], lookback, 5)
#
#     return x_train, x_test, y_train, y_test
#
# def preprocess_unseen_data_for_cnn_lstm(ticker_symbol, start_date, end_date, lookback):
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#     stock_data = add_technical_indicators(stock_data)
#
#     # Normalize the data
#     scaler = MinMaxScaler()
#     stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']] = scaler.fit_transform(
#         stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']])
#
#     # Prepare the data for CNN-LSTM
#     X, Y = [], []
#     for i in range(len(stock_data) - lookback):
#         X.append(stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']].iloc[i:i + lookback].values)
#         Y.append(stock_data['Close'].iloc[i + lookback])
#
#     X = np.array(X)
#     Y = np.array(Y).reshape(-1, 1)
#
#     # Convert data types
#     X = X.astype('float32')
#     Y = Y.astype('float32')
#
#     # Check and replace NaNs or infinite values
#     X = np.nan_to_num(X)
#     Y = np.nan_to_num(Y)
#
#     # Reshape the data for CNN-LSTM input
#     X = X.reshape(X.shape[0], lookback, 5)
#
#     return X, Y
#
#
# def preprocess_unseen_data_for_cnn_lstm_trading(ticker_symbol, start_date, end_date, lookback, interval):
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
#     stock_data = add_technical_indicators(stock_data)
#
#     # Normalize the data
#     scaler = MinMaxScaler()
#     stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']] = scaler.fit_transform(
#         stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']])
#
#     # Prepare the data for CNN-LSTM
#     X = []
#     for i in range(len(stock_data) - lookback):
#         X.append(stock_data[['Close', 'MA for 50 days', 'RSI', 'Daily Return', 'MACD']].iloc[i:i + lookback].values)
#
#     X = np.array(X)
#
#     # Convert data types
#     X = X.astype('float32')
#
#     # Check and replace NaNs or infinite values
#     X = np.nan_to_num(X)
#
#     # Reshape the data for CNN-LSTM input
#     X = X.reshape(X.shape[0], lookback, 5)
#
#     return X
#
#
# def add_technical_indicators(stock_data):
#     # Calculate daily returns
#     stock_data['Daily Return'] = stock_data['Close'].pct_change()
#
#     # Calculate moving averages
#     ma_days = [10, 50, 100]
#     for ma in ma_days:
#         col_name = f"MA for {ma} days"
#         stock_data[col_name] = stock_data['Close'].rolling(window=ma).mean()
#
#     # Add RSI and MACD features
#     stock_data['RSI'] = calculate_rsi(stock_data)
#     macd_line, signal_line, macd_hist = calculate_macd(stock_data)
#     stock_data['MACD'] = macd_line
#     stock_data['MACD_Signal'] = signal_line
#     stock_data['MACD_Hist'] = macd_hist
#
#     return stock_data