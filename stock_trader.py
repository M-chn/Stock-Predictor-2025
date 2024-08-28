ALPACA_API_KEY = "PKG03QR2IM5ULS271SDH"
ALPACA_SECRET_KEY = "tEnUGIoq1F6R9D3ihRegunJFUrKa4mFW521HzdUt"
BASE_URL = "https://paper-api.alpaca.markets" 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from alpaca_trade_api.rest import REST, TimeFrame
import tensorflow as tf


api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

# # Test the connection
# account = api.get_account()
# print(account)

# # Get historical data
# symbol = "SPY" # Stock symbol for S&P 500 ETF
# barset = api.get_bars(symbol, TimeFrame.Day, "2023-01-01", "2024-08-01").df

# # View the data
# # print(barset.head())

# # Calculate moving averages
# short_window = 40
# long_window = 100

# barset['short_mavg'] = barset['close'].rolling(window=short_window, min_periods=1).mean()
# barset['long_mavg'] = barset['close'].rolling(window=long_window, min_periods=1).mean()

# # Plot the stock's closing price and moving averages
# plt.figure(figsize=(12, 6))
# plt.plot(barset.index, barset['close'], label='Closing Price', color='blue', alpha=0.6)
# plt.plot(barset.index, barset['short_mavg'], label=f'{short_window}-Day Moving Avg', color='green', alpha=0.7)
# plt.plot(barset.index, barset['long_mavg'], label=f'{long_window}-Day Moving Avg', color='red', alpha=0.7)

# # Add labels and title
# plt.title(f'{symbol} Price with {short_window}-Day and {long_window}-Day Moving Averages', fontsize=14)
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()




# Fetch SPY historical data
symbol = "AAPL"
barset = api.get_bars(symbol, TimeFrame.Day, "2021-01-01", "2024-08-20").df

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale data between 0 and 1
scaled_data = scaler.fit_transform(barset['close'].values.reshape(-1, 1))

# Create the dataset for training
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # Use the last 60 days to predict the next day
X, Y = create_dataset(scaled_data, time_step)

# Reshape X for LSTM input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=25),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
# Train the model
model.fit(X, Y, batch_size=1, epochs=2)

# # Make predictions
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Inverse transform to get actual prices
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)
# Y_train = scaler.inverse_transform(Y_train.reshape(-1, 1))
# Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# # Plot the results
# # plt.figure(figsize=(12, 6))
# # plt.plot(barset.index[train_size + time_step + 1:], Y_test, label='Actual Price')
# # plt.plot(barset.index[train_size + time_step + 1:], test_predict, label='Predicted Price', color='red')
# # plt.title('S&P 500 (SPY) Price Prediction')
# # plt.xlabel('Date')
# # plt.ylabel('Price (USD)')
# # plt.legend()
# # plt.show()

#print current day price
current_price = barset['close'][-1]
print(f"Current price: {current_price:.2f}")

# Predict the next  7 trading days
last_60 = scaled_data[-600:]
x_input = last_60.reshape(1, -1, 1)
next_7_days_pred = []
for i in range(7):
    next_day_pred = model.predict(x_input)
    next_7_days_pred.append(next_day_pred[0][0])
    x_input = np.append(x_input[:, 1:, :], [next_day_pred], axis=1)
    
next_7_days_pred = scaler.inverse_transform(np.array(next_7_days_pred).reshape(-1, 1))
for i, price in enumerate(next_7_days_pred, 1):
    print(f"Predicted price for day {i}: {price[0]:.2f}")

# last_60 = scaled_data[-365:]
# x_input = last_60.reshape(1, -1, 1)
# next_day_pred = model.predict(x_input)
# next_day_pred = scaler.inverse_transform(next_day_pred)
# print(f"Predicted price for the next trading day: {next_day_pred[0][0]:.2f}")


