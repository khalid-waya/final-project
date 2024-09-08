# CE301 Final Project - Stock Price Prediction Using LSTM and CNN Models

## Project Overview
This project focuses on predicting stock prices using advanced machine learning techniques such as Long Short-Term Memory (LSTM) networks and Convolutional Neural Networks (CNN). The primary goal is to accurately forecast stock prices by leveraging time-series data and applying various feature engineering and scaling techniques.

The project is implemented in **Python** and makes extensive use of deep learning libraries such as **Keras** and **TensorFlow**, alongside data manipulation libraries like **NumPy** and **Pandas**. The models trained include both standalone LSTM networks and a hybrid CNN-LSTM architecture to capture both temporal dependencies and feature correlations in stock data.

## Features
- **LSTM Model**: A sequential neural network tailored to handle time-series data.
- **CNN-LSTM Hybrid Model**: Combines convolutional layers for feature extraction and LSTM layers for sequence prediction.
- **Reinforcement Learning (RL) Strategy**: An RL agent is implemented to make trading decisions based on the predictions.
- **Data Scaling**: Various scaling techniques, including **MinMax Scaler**, are employed to normalize input data.
- **Prediction and Visualization**: Model outputs are visualized, showing both training and testing performance on stock data.

## Repository Structure
- `cnn_lstm_model.py`: Script defining the hybrid CNN-LSTM model architecture.
- `lstm_model.py`: Script for the standalone LSTM model.
- `rl_strategy.ipynb`: Jupyter notebook implementing a reinforcement learning strategy for stock trading.
- `daily_return.py`: A script to calculate and visualize daily stock returns.
- `scaler.pkl`: Pickled file for the data scaler used in pre-processing.
- `result_on_test_data.png`: Image output of the model’s predictions on the test data.
- `test_predictions.png`: Visualization comparing predicted vs. actual stock prices.
  
## Installation
To run this project locally:
1. Clone the repository:
    ```bash
    git clone https://github.com/khalid-waya/final-project.git
    cd final-project
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the training script for the LSTM model:
    ```bash
    python lstm_model.py
    ```

## Usage
Once the models are trained, you can visualize the results using the provided plotting scripts or load the saved models to make further predictions on new data.

## Technologies Used
- **Python** (main language)
- **TensorFlow / Keras** (for deep learning)
- **Jupyter Notebook** (for exploratory analysis)
- **Matplotlib** (for visualizations)

## Results
The models are evaluated using mean squared error (MSE) and accuracy metrics. The project demonstrates promising results in stock price prediction, with graphical visualizations highlighting model performance.

## Contributors
- **Khalid Waya** (Project Author)
