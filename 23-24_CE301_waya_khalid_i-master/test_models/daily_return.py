import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU devices found.")





def fetch_stock_data(ticker_symbol_input, start_date_input, end_date_input):
    """
    Fetches historical stock data for a given ticker symbol and date range.

    Args:
        ticker_symbol_input (str): The ticker symbol of the stock.
        start_date_input (str): The start date of the date range (format: "YYYY-MM-DD").
        end_date_input (str): The end date of the date range (format: "YYYY-MM-DD").

    Returns:
        pandas.DataFrame: The fetched stock data as a DataFrame.
    """
    stock_data = yf.download(ticker_symbol_input, start=start_date_input, end=end_date_input)
    return stock_data


# Specify the ticker symbol and the date range
ticker_symbol = "AAPL"
start_date = "2000-01-01"
end_date = "2023-12-30"

# Fetch historical data
stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)





def add_date_column(input_stock_data):
    """
    Adds a "Date" column to the stock data DataFrame using the index.

    Args:
        input_stock_data (pandas.DataFrame): The stock data DataFrame.

    Returns:
        pandas.DataFrame: The stock data DataFrame with the added "Date" column.
    """
    input_stock_data["Date"] = input_stock_data.index
    return input_stock_data


stock_data = add_date_column(stock_data)

close_price_data = stock_data[["Date", "Close"]]
close_price_data

close_price_data["Date"] = pd.to_datetime(close_price_data["Date"])

plt.plot(close_price_data["Date"], close_price_data["Close"])

from copy import deepcopy as dc


def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    df['Daily_Return'] = df['Close'].pct_change()  # Calculate daily returns
    df.drop(columns=['Close'], inplace=True)  # Remove the 'Close' column
    df.dropna(inplace=True)  # Remove any NaN values

    for i in range(1, n_steps + 1):
        df[f'Return(t-{i})'] = df['Daily_Return'].shift(i)

    df.dropna(inplace=True)
    return df
lookback = 6
shifted_df = prepare_dataframe_for_lstm(close_price_data, lookback)
shifted_df


from sklearn.preprocessing import MinMaxScaler
import joblib


def create_scaler(shifted_df):
    """
    Creates a MinMaxScaler object with a feature range of (-1, 1) and fits it to the input DataFrame.

    Args:
        shifted_df (pandas.DataFrame): The input DataFrame used to fit the scaler.

    Returns:
        MinMaxScaler: The fitted scaler object.
    """
    # Convert the DataFrame to a NumPy array
    shifted_df_as_numpy = shifted_df.to_numpy()
    print(shifted_df_as_numpy.shape)

    # Create a MinMaxScaler object with a feature range of (-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit the scaler to the data
    scaler.fit(shifted_df_as_numpy)

    return scaler


def scale_data(shifted_df, scaler):
    """
    Scales the input DataFrame using the provided scaler.

    Args:
        shifted_df (pandas.DataFrame): The input DataFrame to be scaled.
        scaler (MinMaxScaler): The scaler object used to transform the data.

    Returns:
        numpy.ndarray: The scaled data as a NumPy array.
    """
    # Convert the DataFrame to a NumPy array
    shifted_df_as_numpy = shifted_df.to_numpy()
    print(shifted_df_as_numpy.shape)

    # Scale the data using the scaler
    shifted_df_as_np = scaler.transform(shifted_df_as_numpy)
    print(shifted_df_as_np)

    return shifted_df_as_np


def save_scaler(scaler, filename):
    """
    Saves the scaler object to a file using joblib.

    Args:
        scaler (MinMaxScaler): The scaler object to be saved.
        filename (str): The name of the file to save the scaler object.
    """
    joblib.dump(scaler, filename)
    print(f"Scaler object saved to {filename}")

    # Create the scaler


scaler = create_scaler(shifted_df)

# Scale the data using the scaler
scaled_data = scale_data(shifted_df, scaler)

# Save the scaler object
save_scaler(scaler, 'scaler.pkl')



def prepare_data(input_scaled_data, input_lookback, split_ratio=0.95):
    """
    Prepares the data for training and testing.

    Args:
        input_scaled_data (numpy.ndarray): The scaled data as a NumPy array.
        input_lookback (int): The number of previous time steps to consider as input features.
        split_ratio (float): The ratio used to split the data into training and testing sets.
                             Default is 0.95, representing a 95% training split.

    Returns:
        tuple: A tuple containing the following elements:
            - X_train (tensorflow.Tensor): The input features for the training set.
            - X_test (tensorflow.Tensor): The input features for the testing set.
            - y_train (tensorflow.Tensor): The target values for the training set.
            - y_test (tensorflow.Tensor): The target values for the testing set.
    """
    X = input_scaled_data[:, 1:]
    y = input_scaled_data[:, 0]

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X = dc(np.flip(X, axis=1))

    split_index = int(len(X) * split_ratio)
    print(f"Split index: {split_index}")

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    X_train = X_train.reshape((-1, input_lookback, 1))
    X_test = X_test.reshape((-1, input_lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    print(f"Reshaped X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Reshaped y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = prepare_data(scaled_data, input_lookback=lookback, split_ratio=0.95)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))


batch_size = 16

train_data = train_data.batch(batch_size).shuffle(len(X_train)).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

for x_batch, y_batch in train_data:
    print(x_batch.shape, y_batch.shape)
    break  # To only process the first batch


batch_size = 16

train_data = train_data.batch(batch_size).shuffle(len(X_train)).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

for x_batch, y_batch in train_data:
    print(x_batch.shape, y_batch.shape)
    break  # To only process the first batch



class LSTMModel(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super(LSTMModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=False,
                                         recurrent_initializer='glorot_uniform',
                                         recurrent_activation='tanh',
                                         stateful=False,
                                         batch_input_shape=(None, input_size))
        self.fc = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.lstm(x)
        return self.fc(x)


model = LSTMModel(input_size=1, hidden_size=4, num_stacked_layers=1)
model



def train_one_epoch(model, optimizer, loss_function, train_dataset, epoch):
    running_loss = 0.0
    for batch_index, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_function(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        running_loss += loss.numpy()

        if (batch_index + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f'Epoch {epoch + 1}, Batch {batch_index + 1}, Loss: {avg_loss:.4f}')
            running_loss = 0.0


def validate_one_epoch(model, loss_function, test_dataset):
    model.trainable = True  # Ensure the model is in evaluation mode
    running_loss = 0.0
    total_batches = 0

    for x_batch, y_batch in test_dataset:
        predictions = model(x_batch, training=False)
        loss = loss_function(y_batch, predictions)
        running_loss += loss.numpy()
        total_batches += 1

    avg_loss = running_loss / total_batches
    print(f'Val Loss: {avg_loss:.3f}')
    print('***************************************************\n')



# Example usage within a training loop
learning_rate = 0.001
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()
num_epochs = 20

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, loss_function, train_data, epoch)
    validate_one_epoch(model, loss_function, test_data)

predicted = model.predict(X_train)

plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback + 1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])
train_predictions


dummies = np.zeros((X_train.shape[0], lookback + 1))
dummies[:, 0] = y_train.numpy().flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
new_y_test

plt.plot(new_y_test, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()


# Make predictions
test_predictions = model.predict(X_test)

# Flatten predictions
test_predictions = test_predictions.flatten()

# Initialize a zeros array
dummies = np.zeros((X_test.shape[0], lookback + 1))

# Assign predictions to the first column
dummies[:, 0] = test_predictions

# Apply inverse transformation
dummies = scaler.inverse_transform(dummies)

# Extract denormalized predictions
test_predictions = dummies[:, 0]


test_predictions

dummies = np.zeros((X_test.shape[0], lookback + 1))
dummies[:, 0] = y_test.numpy().flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
new_y_test

plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
# plt.figure(12.47,11.35)
plt.legend()
plt.show()

from keras.metrics import MeanSquaredError, MeanAbsoluteError

# Assume 'model' is your trained LSTM model, 'x_test' is your test features, and 'y_test' is the actual values.
mse = MeanSquaredError()
mae = MeanAbsoluteError()

# Generate predictions
y_pred = model.predict(X_test)

# Update state of the MSE and MAE with the actual and predicted values
mse.update_state(y_test, y_pred)
mae.update_state(y_test, y_pred)

# Get the result of the MSE and MAE
mse_result = mse.result().numpy()
mae_result = mae.result().numpy()

print(f'MSE: {mse_result}')
print(f'MAE: {mae_result}')

model.compile()

# Save the entire model as a SavedModel.
model.save(
    '/Users/khalidwaya/PycharmProjects/23-24_CE301_waya_khalid_i/saved_models')  # Replace with the path where you want to save the model


# To load the model back:
loaded_model = tf.keras.models.load_model("/Users/khalidwaya/PycharmProjects/23-24_CE301_waya_khalid_i/saved_models")



def finding_stock(ticker_symbol, start_date, end_date):
    # # Specify the ticker symbol and the date range
    # ticker_symbol = "AAPL"
    # start_date = "2000-01-01"
    # end_date = "2023-12-30"

    # Fetch historical data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    stock_data["Date"] = stock_data.index
    stock_data

    close_price_data = stock_data[["Date", "Close"]]
    close_price_data

    close_price_data["Date"] = pd.to_datetime(close_price_data["Date"])

    plt.plot(close_price_data["Date"], close_price_data["Close"])
    return close_price_data


data = finding_stock("AMZN", "2024-01-01", "2024-02-20")





# def prepare_dataframe_for_lstm(df, n_steps):
#     df_copy = df.copy()
#     df_copy.set_index('Date', inplace=True)
#     for i in range(1, n_steps + 1):
#         df_copy[f'Close(t-{i})'] = df_copy['Close'].shift(i)
#     df_copy.dropna(inplace=True)
#     return df_copy
#
# def preprocess_data_for_prediction(ticker_symbol, start_date, end_date, lookback, scaler_path):
#     # Fetch historical data
#     stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
#
#     # Process the Close price data
#     stock_data['Date'] = stock_data.index
#     close_price_data = stock_data[['Date', 'Close']]
#     close_price_data['Date'] = pd.to_datetime(close_price_data['Date'])
#
#     # Prepare data for LSTM
#     shifted_df = prepare_dataframe_for_lstm(close_price_data, lookback)
#
#     shifted_df
#     # Load the fitted scaler
#     scaler = joblib.load(scaler_path) if isinstance(scaler_path, str) else scaler_path
#
#     # Check if the scaler is indeed MinMaxScaler instance
#     if not isinstance(scaler, MinMaxScaler):
#         raise ValueError("The scaler_path must be a path to a joblib file containing a fitted MinMaxScaler instance.")
#     #
#     # # Normalize features
#     features = shifted_df.iloc[:, 1:].values  # Exclude the target (first column)
#     features_scaled = scaler.transform(features )
#     print(features_scaled)
#     # Extract target and features
#     X_scaled = np.flip(features_scaled, axis=1)  # Flip columns if needed to match training data
#     y = shifted_df.iloc[:, 0].values
# #
#     # Reshape data for LSTM
#     X_scaled = X_scaled.reshape((-1, lookback, 1))
#     y = y.reshape((-1, 1))
#
#     # Convert to tensors
#     X_unseen = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
#     y_unseen = tf.convert_to_tensor(y, dtype=tf.float32)
#
#     return X_unseen, y_unseen
# #
# # # Usage example
# scaler_path = 'scaler.save'  # Replace with your actual scaler path
# X_unseen, y_unseen = preprocess_data_for_prediction(
#     ticker_symbol="AAPL",
#     start_date="2024-01-01",
#     end_date="2024-02-20",
#     lookback=7,
#     scaler_path=scaler_path
# )
#
# X_unseen.shape

def preprocess_unseen_data(ticker_symbol, start_date, end_date, lookback):
    # Fetch historical data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Process the Close price data
    stock_data['Date'] = stock_data.index
    close_price_data = stock_data[['Date', 'Close']]
    close_price_data['Date'] = pd.to_datetime(close_price_data['Date'])

    # Prepare data for LSTM
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

    joblib.dump(scaler, "scaler1.save")

    return X_unseen, Y_unseen



ticker_symbol = "AMZN"
start_date = "2024-01-01"
end_date = "2024-03-01"
lookback = 7  # Assuming lookback of 7
scaler = MinMaxScaler(feature_range=(-1, 1))  # Initialize the scaler with the range used during training

# You need to fit the scaler on your training data before using it here
# scaler.fit(your_training_data)

X_unseen, Y_unseen = preprocess_unseen_data(ticker_symbol, start_date, end_date, lookback)

Y_unseen.shape

loaded_model.evaluate(X_unseen, Y_unseen)

predictions = loaded_model.predict(X_unseen)


plt.plot(Y_unseen, label='Actual Close')
plt.plot(predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# Assume 'model' is your trained LSTM model, 'x_test' is your test features, and 'y_test' is the actual values.
mse = MeanSquaredError()
mae = MeanAbsoluteError()

# Generate predictions
y_pred = model.predict(X_unseen)

# Update state of the MSE and MAE with the actual and predicted values
mse.update_state(Y_unseen, y_pred)
mae.update_state(Y_unseen, y_pred)

# Get the result of the MSE and MAE
mse_result = mse.result().numpy()
mae_result = mae.result().numpy()

print(f'MSE: {mse_result}')
print(f'MAE: {mae_result}')

# Load the saved scaler
scaler = joblib.load('scaler1.save')

# Flatten predictions
test_predictions = predictions.flatten()

# Initialize a zeros array
dummies = np.zeros((X_unseen.shape[0], lookback + 1))

# Assign predictions to the first column
dummies[:, 0] = test_predictions

# Apply inverse transformation
denormalized_predictions = scaler.inverse_transform(dummies)[:, 0]

# Extract the actual (denormalized) close values
dummies = np.zeros((Y_unseen.shape[0], lookback + 1))
dummies[:, 0] = Y_unseen.numpy().flatten()
denormalized_actual_close = scaler.inverse_transform(dummies)[:, 0]

# You can now use denormalized_predictions to plot the true results
plt.plot(denormalized_actual_close, label='Actual Close')
plt.plot(denormalized_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
