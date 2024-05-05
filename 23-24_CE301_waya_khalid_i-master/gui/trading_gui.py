import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import joblib
from alpaca_trade_api.rest import TimeFrame
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import alpaca_trade_api as tradeapi
import pandas as pd
import threading
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lstm_functions import (fetch_stock_data,
                            fetch_stock_data_for_trading, add_date_column, prepare_dataframe_for_lstm,
                            prepare_data, preprocess_unseen_data, preprocess_unseen_data_for_trading,
                            prepare_data_for_cnn_lstm, preprocess_unseen_data_for_cnn_lstm_trading
                            )

# Set up Alpaca API credentials
api_key = "PK2BOYAMBWYQXGOCJNGX"
api_secret = "jFQb9fkgTrPZSFh9gR3NR8Q9JEodoNcTweiqjn01"
base_url = "https://paper-api.alpaca.markets"

# Create an instance of the Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")


def train_model():
    """
    Train the LSTM model using the selected ticker symbol and date range.
    """
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")

        # Fetch historical data
        stock_data = fetch_stock_data_for_trading(ticker_symbol, start_date, end_date)
        stock_data = add_date_column(stock_data)

        # Prepare the data for LSTM
        lookback = 7
        shifted_df = prepare_dataframe_for_lstm(stock_data[["Date", "Close"]], lookback)
        if shifted_df.empty:
            messagebox.showerror("Error", "Insufficient data for the selected date range.")
            return

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(shifted_df.to_numpy())
        joblib.dump(scaler, "scaler.pkl")

        X_train, X_test, y_train, y_test = prepare_data(scaled_data, lookback)

        # Load the trained model
        model = tf.keras.models.load_model("../trained_models")

        # Plot the training results
        plot_results(model, X_train, y_train, scaler)
    except Exception as e:
        print(e)
        messagebox.showerror("Error", f"An error occurred during model training: invalid time frame input to train the models")


def train_model_for_trading():
    """
    Train the LSTM model for trading using the selected ticker symbol, date range, and interval.
    """
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")
        interval = interval_var.get()

        # Fetch historical data
        stock_data = fetch_stock_data(ticker_symbol, start_date, end_date, interval=interval)
        stock_data = add_date_column(stock_data)

        # Prepare the data for LSTM
        lookback = 7
        shifted_df = prepare_dataframe_for_lstm(stock_data[["Date", "Close"]], lookback)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(shifted_df.to_numpy())
        joblib.dump(scaler, "scaler.pkl")

        X_train, X_test, y_train, y_test = prepare_data(scaled_data, lookback)

        # Load the trained model
        model = tf.keras.models.load_model("../trained_models")

        # Plot the training results
        plot_results(model, X_train, y_train, scaler)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during model training for trading: invalid time frame input to train the models")


def test_model():
    """
    Test the LSTM model using the selected ticker symbol and date range.
    """
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")

        # Preprocess the unseen data
        lookback = 7
        X_unseen, Y_unseen = preprocess_unseen_data(ticker_symbol, start_date, end_date, lookback)

        # Load the trained model and scaler
        model = tf.keras.models.load_model("../trained_models")
        scaler = joblib.load("scaler.pkl")

        # Make predictions
        predictions = model.predict(X_unseen)

        # Plot the testing results
        plot_results(model, X_unseen, Y_unseen, scaler, predictions)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during model testing: {str(e)}")


def calculate_metrics():
    """
    Calculate and display the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)
    for the model's predictions.
    """
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")

        # Preprocess the unseen data
        lookback = 7
        X_unseen, Y_unseen = preprocess_unseen_data(ticker_symbol, start_date, end_date, lookback)

        # Load the trained model and scaler
        model = tf.keras.models.load_model("../trained_models")
        scaler = joblib.load("scaler.pkl")

        # Make predictions
        predictions = model.predict(X_unseen)

        # Denormalize the predictions and actual values
        predictions = denormalize_data(predictions, scaler)
        Y_unseen = denormalize_data(Y_unseen, scaler)

        # Calculate the metrics
        mse = mean_squared_error(Y_unseen, predictions)
        mae = mean_absolute_error(Y_unseen, predictions)
        rmse = np.sqrt(mse)

        # Display the metrics
        metrics_label.config(text=f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during metrics calculation: {str(e)}")


def make_trade():
    """
    Make a trade based on the model's predictions using the selected ticker symbol, date range, and interval.
    """
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")
        interval = interval_var.get()

        # Preprocess the unseen data
        lookback = 7
        X_unseen, _ = preprocess_unseen_data_for_trading(ticker_symbol, start_date, end_date, lookback, interval)

        # Load the trained model and scaler
        model = tf.keras.models.load_model("../trained_models")
        scaler = joblib.load("scaler.pkl")

        # Make predictions
        predictions = model.predict(X_unseen)

        # Denormalize the predictions
        predictions = denormalize_data(predictions, scaler)

        # Get the latest close price
        latest_data = api.get_bars(ticker_symbol, TimeFrame.Minute, limit=1).df
        latest_close = latest_data['close'][0]

        # Compare the predicted price with the latest close price
        if predictions[-1] > latest_close:
            # Place a buy order
            api.submit_order(
                symbol=ticker_symbol,
                qty=1,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            trade_label.config(text=f"Bought {ticker_symbol} at {latest_close:.2f}")
        else:
            # Place a sell order
            api.submit_order(
                symbol=ticker_symbol,
                qty=1,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            trade_label.config(text=f"Sold {ticker_symbol} at {latest_close:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during trading: {str(e)}")
        print(e)


def plot_pnl_chart():
    try:
        # Fetch the account information
        account = api.get_account()

        # Get the current portfolio value
        portfolio_value = float(account.portfolio_value)

        # Get the list of all positions
        positions = api.list_positions()

        # Calculate the total cost basis and market value of positions
        total_cost_basis = 0
        total_market_value = 0
        for position in positions:
            total_cost_basis += float(position.cost_basis)
            total_market_value += float(position.market_value)

        # Calculate the unrealized PnL
        unrealized_pnl = total_market_value - total_cost_basis

        # Determine if the portfolio is in profit or loss
        if unrealized_pnl >= 0:
            profit_loss_text = f"Portfolio is in profit by ${unrealized_pnl:.2f}"
        else:
            profit_loss_text = f"Portfolio is in loss by ${-unrealized_pnl:.2f}"

        # Clear the existing plot
        figure.clear()

        # Create a new subplot for the portfolio value graph
        ax1 = figure.add_subplot(211)

        # Fetch historical portfolio data
        portfolio_history = api.get_portfolio_history(period='1D', timeframe='5Min', extended_hours=True).df

        # Plot the portfolio value graph
        ax1.plot(portfolio_history.index, portfolio_history['equity'])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Graph')

        # Create a new subplot for the profit/loss text
        ax2 = figure.add_subplot(212)

        # Display the portfolio value and profit/loss text
        ax2.text(0.5, 0.5, f"Portfolio Value: ${portfolio_value:.2f}", fontsize=16, ha='center')
        ax2.text(0.5, 0.3, profit_loss_text, fontsize=14, ha='center')
        ax2.set_axis_off()

        # Adjust the spacing between subplots
        figure.tight_layout()

        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while plotting the portfolio data: {str(e)}")

def train_and_test_cnn_lstm_model():
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")

        # Fetch historical data
        stock_data = fetch_stock_data_for_trading(ticker_symbol, start_date, end_date)

        # Load the saved scaler
        scaler = joblib.load("close_scaler.pkl")

        # Prepare the data for CNN-LSTM
        lookback = 100
        train_X, test_X, train_Y, test_Y = prepare_data_for_cnn_lstm(stock_data, lookback, scaler)

        # Load the pre-trained CNN-LSTM model
        model = tf.keras.models.load_model("../cnn_lstm1")

        # Evaluate the model on test data
        model.evaluate(test_X, test_Y)

        # Plot the training and testing results
        plot_results(model, train_X, train_Y, scaler)
        plot_results(model, test_X, test_Y, scaler)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during model training and testing: {str(e)}")

def make_trade_with_cnn_lstm():
    try:
        ticker_symbol = entry_ticker.get()
        start_date = start_cal.get_date().strftime("%Y-%m-%d")
        end_date = end_cal.get_date().strftime("%Y-%m-%d")
        interval = interval_var.get()

        # Load the saved scaler
        scaler = joblib.load("close_scaler.pkl")

        # Preprocess the unseen data for trading
        lookback = 100
        X_unseen = preprocess_unseen_data_for_cnn_lstm_trading(ticker_symbol, start_date, end_date, lookback, interval, scaler)

        # Load the pre-trained CNN-LSTM model
        model = tf.keras.models.load_model("../cnn_lstm1")

        # Make predictions
        predictions = model.predict(X_unseen)

        # Denormalize the predictions
        predictions = scaler.inverse_transform(predictions)

        # Get the latest close price
        latest_data = api.get_bars(ticker_symbol, TimeFrame.Minute, limit=1).df
        latest_close = latest_data['close'][0]
        print("It does work")

        # Compare the predicted price with the latest close price
        if predictions[-1] > latest_close:
            # Place a buy order using a bracket order
            take_profit_price = round(latest_close * 1.05, 2)  # Round to two decimal places
            stop_loss_price = round(latest_close * 0.95, 2)  # Round to two decimal places
            api.submit_order(
                symbol=ticker_symbol,
                qty=1,
                side='buy',
                type='market',  # Set the entry order type to 'market'
                time_in_force='gtc',
                order_class='bracket',
                take_profit=dict(
                    limit_price=take_profit_price
                ),
                stop_loss=dict(
                    stop_price=stop_loss_price
                )
            )
            trade_label.config(text=f"Bought {ticker_symbol} at {latest_close:.2f}")
        else:
            # Place a sell order using a bracket order
            take_profit_price = round(latest_close * 0.95, 2)  # Round to two decimal places
            stop_loss_price = round(latest_close * 1.05, 2)  # Round to two decimal places
            api.submit_order(
                symbol=ticker_symbol,
                qty=1,
                side='sell',
                type='market',  # Set the entry order type to 'market'
                time_in_force='gtc',
                order_class='bracket',
                take_profit=dict(
                    limit_price=take_profit_price
                ),
                stop_loss=dict(
                    stop_price=stop_loss_price
                )
            )
            trade_label.config(text=f"Sold {ticker_symbol} at {latest_close:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during trading: {str(e)}")
        print(e)


# Create the main window
my_window = tk.Tk()
my_window.title("Stock Price Prediction")
my_window.geometry("1200x800")

# Create a frame to hold all the widgets
frame_main = ttk.Frame(my_window)
frame_main.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Create input fields for the user to enter the ticker symbol, start date, and end date
frame_input = ttk.Frame(frame_main)
frame_input.pack(pady=10)

label_ticker = ttk.Label(frame_input, text="Ticker Symbol:")
label_ticker.pack(side=tk.LEFT)
entry_ticker = ttk.Entry(frame_input)
entry_ticker.pack(side=tk.LEFT)

label_start = ttk.Label(frame_input, text="Start Date:")
label_start.pack(side=tk.LEFT)
start_cal = DateEntry(frame_input, width=12, background="darkblue", foreground="white", borderwidth=2)
start_cal.pack(side=tk.LEFT, padx=10)

label_end = ttk.Label(frame_input, text="End Date:")
label_end.pack(side=tk.LEFT)
end_cal = DateEntry(frame_input, width=12, background="darkblue", foreground="white", borderwidth=2)
end_cal.pack(side=tk.LEFT, padx=10)

# Create buttons for training and testing the LSTM model
frame_buttons = ttk.Frame(frame_main)
frame_buttons.pack(pady=10)

button_train = ttk.Button(frame_buttons, text="Train Model", command=train_model)
button_train.pack(side=tk.LEFT)

button_test = ttk.Button(frame_buttons, text="Test Model", command=test_model)
button_test.pack(side=tk.LEFT)

button_metrics = ttk.Button(frame_buttons, text="Calculate Metrics", command=calculate_metrics)
button_metrics.pack(side=tk.LEFT)

metrics_label = ttk.Label(frame_main, text="")
metrics_label.pack(pady=10)

button_train_for_trading = ttk.Button(frame_buttons, text="Train Model for Trading", command=train_model_for_trading)
button_train_for_trading.pack(side=tk.LEFT)


# Create a button for making trades using the CNN-LSTM model
button_trade_cnn_lstm = ttk.Button(frame_buttons, text="Make Trade with CNN-LSTM", command=make_trade_with_cnn_lstm)
button_trade_cnn_lstm.pack(side=tk.LEFT)

button_trade = ttk.Button(frame_buttons, text="Make Trade", command=make_trade)
button_trade.pack(side=tk.LEFT)

trade_label = ttk.Label(frame_main, text="")
trade_label.pack(pady=10)

# Create buttons for training and testing the CNN-LSTM model
button_train_test_cnn_lstm = ttk.Button(frame_buttons, text="Train and Test CNN-LSTM", command= train_and_test_cnn_lstm_model)
button_train_test_cnn_lstm.pack(side=tk.LEFT)


interval_var = tk.StringVar()
interval_var.set("1m")  # Default interval value

interval_label = ttk.Label(frame_input, text="Interval:")
interval_label.pack(side=tk.LEFT)
interval_dropdown = ttk.Combobox(frame_input, textvariable=interval_var, values=["1m", "5m", "15m", "30m", "60m"])
interval_dropdown.pack(side=tk.LEFT)

button_pnl_chart = ttk.Button(frame_buttons, text="Plot PnL Chart", command=plot_pnl_chart)
button_pnl_chart.pack(side=tk.LEFT)

# Create a frame to display the graphical representation of the results
frame_plot = ttk.Frame(frame_main)
frame_plot.pack(expand=True, fill=tk.BOTH)

figure = plt.Figure(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(figure, master=frame_plot)
canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

# Create a frame for instructions
frame_instructions = ttk.Frame(frame_main)
frame_instructions.pack(pady=10)

instructions_label = ttk.Label(frame_instructions, text="Instructions:")
instructions_label.pack(side=tk.LEFT)

instructions_text = ttk.Label(frame_instructions, text="To use 'Train Model for Trading':\n"
                                                       "- Select the past 30 days for 1-minute interval\n"
                                                       "- Select the past 60 days for 5-minute interval\n"
                                                       "- Select the past 90 days for 15-minute interval\n"
                                                       "- Select the past 180 days for 30-minute interval\n"
                                                       "- Select the past 360 days for 60-minute interval")
instructions_text.pack(side=tk.LEFT)


def plot_results(model, X, y, scaler, predictions=None):
    """
    Plot the actual and predicted stock prices.
    """
    figure.clear()
    ax = figure.add_subplot(111)

    if predictions is None:
        # Plot training results
        train_predictions = model.predict(X)
        train_predictions = denormalize_data(train_predictions, scaler)
        y_train = denormalize_data(y, scaler)
        ax.plot(y_train, label='Actual Close')
        ax.plot(train_predictions, label='Predicted Close')
    else:
        # Plot testing results
        test_predictions = denormalize_data(predictions, scaler)
        Y_unseen_denormalized = denormalize_data(y, scaler)
        ax.plot(Y_unseen_denormalized, label='Actual Close')
        ax.plot(test_predictions, label='Predicted Close')

    ax.set_xlabel('Day')
    ax.set_ylabel('Close')
    ax.legend()

    canvas.draw()


def denormalize_data(data, scaler):
    """
    Denormalize the scaled data using the scaler.
    """
    # Convert TensorFlow tensor to NumPy array if necessary
    if isinstance(data, tf.Tensor):
        data = data.numpy()
    # Reshape the data to have the same shape as the scaler input
    lookback = scaler.n_features_in_ - 1
    dummy_data = np.zeros((data.shape[0], lookback + 1))
    dummy_data[:, 0] = data.flatten()

    # Inverse transform the data
    denormalized_data = scaler.inverse_transform(dummy_data)[:, 0]

    return denormalized_data


# Run the main event loop
my_window.mainloop()