import csv 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from datetime import datetime

# Load blink data
file_path = "Soniya_blink_rate_1.csv"
timestamps, blink_counts = [], []
try:
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        print(f"Header: {header}")  # Debugging line
        
        for row in reader:
            print(f"Row: {row}")  # Debugging line
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                print(f"Skipping invalid row: {row}")  # Debugging line
                continue

            try:
                # Add '00:' to the beginning of the timestamp to convert to HH:MM:SS format
                time_obj = datetime.strptime("00:" + row[0], "%H:%M:%S")
            except ValueError:
                print(f"Skipping invalid timestamp: {row[0]}")  # Debugging line
                continue
            
            if not timestamps:
                start_time = time_obj
            seconds_since_start = (time_obj - start_time).total_seconds()
            
            try:
                blink_count = int(row[1])
            except ValueError:
                print(f"Skipping invalid blink count: {row[1]}")  # Debugging line
                blink_count = 0
            
            timestamps.append(seconds_since_start)
            blink_counts.append(blink_count)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

if not blink_counts:
    print("Error: No valid blink data found.")
    exit()

# Use the timestamps and blink counts directly for prediction
time_axis = np.array(timestamps)  # X-axis: time in seconds
blink_rate_values = np.array(blink_counts)  # Y-axis: Blink Rate (0 or 1)

# --- ARIMA Prediction ---
def predict_arima(data):
    try:
        model = ARIMA(data, order=(2, 1, 1))
        model_fit = model.fit()
        return np.clip(model_fit.fittedvalues[-len(data):], 0, 1)  # Clip between 0 and 1 for blink rates
    except Exception as e:
        print(f"ARIMA error: {e}")
        return np.full_like(data, data[-1])

# --- Ridge Polynomial Regression ---
def predict_ridge_polynomial(data, degree=4):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data
    
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_poly, y)
    
    return np.clip(model.predict(X_poly), 0, 1)  # Clip predictions to 0-1 range for blink rates

# --- LSTM Prediction ---
def predict_lstm(data, look_back=10, epochs=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X_train, y_train = [], []
    for i in range(len(data_scaled) - look_back):
        X_train.append(data_scaled[i:i+look_back])
        y_train.append(data_scaled[i+look_back])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([Input(shape=(look_back, 1)), LSTM(50, return_sequences=True), LSTM(50), Dense(1)])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=1)
    
    predictions = model.predict(X_train, verbose=0).flatten()
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

def smooth_blink_rate(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
smoothed_blink_rate = smooth_blink_rate(blink_rate_values)

# Calculate blink rate per minute (blinks/minute)
blink_rate_per_minute = blink_rate_values * 60  # If blink data is per second, multiply by 60 for rate per minute

# Generate predictions based on smoothed data
arima_predictions = predict_arima(smoothed_blink_rate)
ridge_poly_predictions = predict_ridge_polynomial(smoothed_blink_rate, degree=4)
lstm_predictions = predict_lstm(smoothed_blink_rate)

# Align LSTM prediction length with actual data
if len(lstm_predictions) > len(smoothed_blink_rate):
    lstm_predictions = lstm_predictions[:len(smoothed_blink_rate)]
elif len(lstm_predictions) < len(smoothed_blink_rate):
    lstm_predictions = np.pad(lstm_predictions, (0, len(smoothed_blink_rate) - len(lstm_predictions)), mode='edge')

# --- Plot Blink Rate and Error Graph ---
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- First subplot: Blink Rate ---
axs[0].plot(time_axis[:len(smoothed_blink_rate)], smoothed_blink_rate, label='Smoothed Blink Rate', color='blue', linestyle='-')
axs[0].plot(time_axis[:len(arima_predictions)], arima_predictions, label='ARIMA Predictions', color='green', linestyle='--')
axs[0].plot(time_axis[:len(ridge_poly_predictions)], ridge_poly_predictions, label='Ridge Polynomial Predictions', color='purple', linestyle='-.')
axs[0].plot(time_axis[:len(lstm_predictions)], lstm_predictions, label='LSTM Predictions', color='orange', linestyle='-')
axs[0].set_ylabel('Blink Rate (blinks/min)')
axs[0].set_title('Blink Rate vs Predictions')
axs[0].legend()
axs[0].grid(True)

# --- Second subplot: MAE Error Over Time ---
mae_arima = np.abs(arima_predictions - smoothed_blink_rate)
mae_poly = np.abs(ridge_poly_predictions - smoothed_blink_rate)
mae_lstm = np.abs(lstm_predictions - smoothed_blink_rate)

axs[1].plot(time_axis[:len(mae_arima)], mae_arima, label='ARIMA MAE', color='green', linestyle='-', linewidth=2)
axs[1].plot(time_axis[:len(mae_poly)], mae_poly, label='Ridge Polynomial MAE', color='purple', linestyle='-', linewidth=2)
axs[1].plot(time_axis[:len(mae_lstm)], mae_lstm, label='LSTM MAE', color='orange', linestyle='-', linewidth=2)
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Mean Absolute Error (MAE)')
axs[1].set_title('Prediction Error Over Time')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
