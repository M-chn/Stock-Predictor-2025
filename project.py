import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from alpaca_trade_api.rest import REST, TimeFrame
import tensorflow as tf
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL
import tkinter as tk
from tkinter import ttk
import sys

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

def get_prediction(symbol, date_start, date_end):
    barset = api.get_bars(symbol, TimeFrame.Day, str(date_start), str(date_end)).df
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(barset['close'].values.reshape(-1, 1))

    def create_dataset(data, time_step):
        X, Y = [], []
        for i in range(len(data)-time_step-1):
            X.append(data[i:(i+time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 365
    X, Y = create_dataset(scaled_data, time_step)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    input_layer = tf.keras.Input(shape=(X.shape[1], 1))
    model = tf.keras.Sequential([
        input_layer,
        tf.keras.layers.LSTM(units=50, return_sequences=True),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dense(units=25),
        tf.keras.layers.Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X, Y, batch_size=64, epochs=5)

    last_time_step = scaled_data[-time_step:]
    x_input = last_time_step.reshape(1, -1, 1)
    next_7_days_pred = []
    for i in range(7):
        next_day_pred = model.predict(x_input)
        next_7_days_pred.append(next_day_pred[0][0])
        x_input = np.append(x_input[:, 1:, :], [next_day_pred], axis=1)
        
    next_7_days_pred = scaler.inverse_transform(np.array(next_7_days_pred).reshape(-1, 1))
    
    last_month = barset[-30:]
    future_dates = pd.date_range(last_month.index[-1] + pd.Timedelta(days=1), periods=7, freq='B')
    predicted_df = pd.DataFrame(next_7_days_pred, index=future_dates, columns=['Predicted'])
    combined_df = pd.concat([last_month, predicted_df])
    
    plt.figure(figsize=(10, 6))
    plt.plot(combined_df.index[:30], combined_df['close'][:30], label='Actual Price', color='blue')
    plt.plot(combined_df.index[29:], combined_df['Predicted'][29:], linestyle='--', color='red', label='Predicted Price')
    plt.plot(
        [combined_df.index[29], combined_df.index[30]],
        [combined_df['close'].iloc[29], combined_df['Predicted'].iloc[30]],
        color='red',
        linestyle='--'
    )
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{symbol} Stock Price and 7-Day Prediction')
    plt.legend()
    plt.savefig('static/stock_prediction.png')
    
    return plt

def display_popup(symbol, news, plt):
    root = tk.Tk()
    root.title(f"{symbol} Stock Prediction and News")
    root.resizable(True, True)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Create a frame for the plot with padding
    frame1 = ttk.Frame(root, padding=10)
    frame1.grid(row=0, column=0, sticky='nsew')
    title1 = ttk.Label(frame1, text=f"{symbol} Stock Price Prediction", font=("Helvetica", 16, "bold"))
    title1.pack(side=tk.TOP, pady=5)

    # Create a canvas for the plot
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame1)
    canvas.draw()

    # Set the size of the canvas by adjusting the width and height of the Tkinter widget
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.NONE, expand=False)
    canvas_widget.config(width=1000, height=400)

    frame2 = ttk.Frame(root, padding=10)
    frame2.grid(row=1, column=0, sticky='nsew')
    title2 = ttk.Label(frame2, text="Latest News", font=("Helvetica", 16, "bold"))
    title2.pack(side=tk.TOP, pady=5)
    news_text = tk.Text(frame2, wrap=tk.WORD, font=("Helvetica", 12), bg="#f7f7f7", state=tk.DISABLED)
    news_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(frame2, command=news_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    news_text.config(yscrollcommand=scrollbar.set)

    # Enable insertion temporarily to add news, then disable
    news_text.config(state=tk.NORMAL)
    for i in news:
        news_text.insert(tk.END, f"{i}\n\n")
    news_text.config(state=tk.DISABLED)

    # Bind the window close event to exit the program
    def on_closing():
        root.destroy()
        sys.exit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    while True:
        symbol = input("Enter the stock symbol: ").upper()
        try:
            api.get_asset(symbol)
            print(f"{symbol} is a valid stock symbol.")
            news = api.get_news(symbol)
            date_start = datetime.date.today() - datetime.timedelta(days=7*365)
            date_end = datetime.date.today() - datetime.timedelta(days=1)
            plt = get_prediction(symbol, date_start, date_end)
            news1 = []
            for i in news:
                news1.append(str(i.summary).replace("&#39;", "'"))
        
            display_popup(symbol, news1, plt)
            break
        except Exception as e:
            print(f"Invalid stock symbol: {symbol}. Please try again.")
