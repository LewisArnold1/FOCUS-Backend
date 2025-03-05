import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge

# Load blink data
file_path = "Soniya_blinkrate_2_30_data.csv"
timestamps, blink_counts = [], []
try:
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        print(f"Header: {header}")  # Debugging line
        
        for row in reader:
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                continue

            try:
                time_obj = datetime.strptime(row[0], "%H:%M:%S")
            except ValueError:
                continue
            
            if not timestamps:
                start_time = time_obj
            seconds_since_start = (time_obj - start_time).total_seconds()
            
            try:
                blink_count = int(row[1])
            except ValueError:
                blink_count = 0
            
            timestamps.append(seconds_since_start)
            blink_counts.append(blink_count)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit()

if not blink_counts:
    print("Error: No valid blink data found.")
    exit()

# Calculate blink rate per minute (blinks/minute)
def calculate_blink_rate(timestamps, blink_counts):
    blink_rate_per_minute = []
    blink_in_progress = False
    current_blink_count = 0
    current_time = 0
    
    for i in range(1, len(timestamps)):
        if blink_counts[i] == 1:
            if not blink_in_progress:
                current_blink_count += 1
                blink_in_progress = True
        else:
            blink_in_progress = False

        # Update blink rate every minute
        if timestamps[i] >= current_time + 60:
            blink_rate_per_minute.append(current_blink_count)
            current_time += 60
            current_blink_count = 0

    # Append remaining blink count for last minute
    if current_blink_count > 0:
        blink_rate_per_minute.append(current_blink_count)

    return np.array(blink_rate_per_minute)

# Blink rate per minute calculation
blink_rate_per_minute = calculate_blink_rate(timestamps, blink_counts)

# --- ARIMA Prediction ---
def predict_arima(train_data, steps=10):
    try:
        model = ARIMA(train_data, order=(2, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except Exception as e:
        print(f"ARIMA error: {e}")
        return np.full((steps,), train_data[-1])

# --- Ridge Polynomial Regression ---
def predict_ridge_polynomial(train_data, degree=1, steps=10):
    X = np.arange(len(train_data)).reshape(-1, 1)
    y = train_data
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = Ridge(alpha=1.0)
    model.fit(X_poly, y)
    
    # Create the prediction range for the next 10 minutes
    future_X = np.arange(len(train_data), len(train_data) + steps).reshape(-1, 1)
    future_X_poly = poly.transform(future_X)
    return model.predict(future_X_poly)

# --- LSTM Prediction ---
def predict_lstm(train_data, look_back=10, steps=10, epochs=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(len(train_data_scaled) - look_back):
        X_train.append(train_data_scaled[i:i + look_back])
        y_train.append(train_data_scaled[i + look_back])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build a more complex LSTM model with additional layers and more units
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.3))  # Increased dropout
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))  # Increased dropout
    model.add(LSTM(50))
    model.add(Dropout(0.2))  # Increased dropout
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    # Early stopping with more patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0, validation_split=0.1, callbacks=[early_stopping])

    # Predict for the next 10 steps
    last_data = train_data_scaled[-look_back:].reshape(1, look_back, 1)
    predictions_scaled = []
    for _ in range(steps):
        prediction = model.predict(last_data, verbose=0)
        predictions_scaled.append(prediction[0][0])
        last_data = np.roll(last_data, -1, axis=1)
        last_data[0, -1, 0] = prediction

    return scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

# --- Prepare Training and Test Data ---
train_data = blink_rate_per_minute[5:20]  # Data from 5 to 20 minutes for training
test_steps = 10  # Predict the next 10 minutes

# Generate predictions for the next 10 minutes
arima_predictions = predict_arima(train_data, steps=test_steps)
ridge_poly_predictions = predict_ridge_polynomial(train_data, steps=test_steps)
lstm_predictions = predict_lstm(train_data, steps=test_steps)

# --- Actual Data for 30 to 40 minutes ---
actual_data_30_40 = blink_rate_per_minute[15:25]  # Assuming this is available

# Ensure the lengths match
if len(actual_data_30_40) != len(arima_predictions):
    min_length = min(len(actual_data_30_40), len(arima_predictions))
    actual_data_30_40 = actual_data_30_40[:min_length]
    arima_predictions = arima_predictions[:min_length]
    ridge_poly_predictions = ridge_poly_predictions[:min_length]
    lstm_predictions = lstm_predictions[:min_length]

# --- Clip extreme values to avoid large MAPE ---
threshold = 100  # Set an appropriate threshold based on your data
actual_data_30_40 = np.clip(actual_data_30_40, None, threshold)
arima_predictions = np.clip(arima_predictions, None, threshold)
ridge_poly_predictions = np.clip(ridge_poly_predictions, None, threshold)
lstm_predictions = np.clip(lstm_predictions, None, threshold)

# --- Calculate MAE (Mean Absolute Error) ---
mae_arima = np.abs(arima_predictions - actual_data_30_40)
mae_poly = np.abs(ridge_poly_predictions - actual_data_30_40)
mae_lstm = np.abs(lstm_predictions - actual_data_30_40)

# --- RMSE (Root Mean Squared Error) ---
rmse_arima = np.sqrt(np.mean((arima_predictions - actual_data_30_40) ** 2))
rmse_poly = np.sqrt(np.mean((ridge_poly_predictions - actual_data_30_40) ** 2))
rmse_lstm = np.sqrt(np.mean((lstm_predictions - actual_data_30_40) ** 2))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)  # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

mape_arima = mean_absolute_percentage_error(actual_data_30_40, arima_predictions)
mape_poly = mean_absolute_percentage_error(actual_data_30_40, ridge_poly_predictions)
mape_lstm = mean_absolute_percentage_error(actual_data_30_40, lstm_predictions)

# --- Average Blink Rate --- 
average_blink_rate = np.mean(blink_rate_per_minute)
print(f"Average Blink Rate per Minute: {average_blink_rate:.2f}")

# --- Plot Predictions and MAE for Last 10 Minutes ---
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot the actual vs predicted blink rates
axs[0].plot(np.arange(test_steps), actual_data_30_40, label='Actual Blink Rate', color='blue', linestyle='-')
axs[0].plot(np.arange(test_steps), arima_predictions, label='ARIMA Predictions', color='green', linestyle='--')
axs[0].plot(np.arange(test_steps), ridge_poly_predictions, label='Ridge Polynomial Predictions', color='purple', linestyle='-.')
axs[0].plot(np.arange(test_steps), lstm_predictions, label='LSTM Predictions', color='orange', linestyle='-')

axs[0].set_ylabel('Blink Rate (blinks/min)')
axs[0].set_title('Actual vs Predicted Blink Rates (30-40 minutes)')
axs[0].legend()
axs[0].grid(True)

# Plot the MAE for each model
axs[1].plot(np.arange(test_steps), mae_arima, label='ARIMA MAE', color='green', linestyle='-', linewidth=2)
axs[1].plot(np.arange(test_steps), mae_poly, label='Ridge Polynomial MAE', color='purple', linestyle='-', linewidth=2)
axs[1].plot(np.arange(test_steps), mae_lstm, label='LSTM MAE', color='orange', linestyle='-', linewidth=2)

axs[1].set_xlabel('Time (minutes)')
axs[1].set_ylabel('Mean Absolute Error (MAE)')
axs[1].set_title('Mean Absolute Error (MAE) for Each Model')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# --- Output Error Metrics ---
print(f"ARIMA MAPE: {mape_arima:.2f}%")
print(f"Ridge Polynomial MAPE: {mape_poly:.2f}%")
print(f"LSTM MAPE: {mape_lstm:.2f}%")
print(f"ARIMA RMSE: {rmse_arima:.2f}")
print(f"Ridge Polynomial RMSE: {rmse_poly:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")
print(f"ARIMA MAE: {np.mean(mae_arima):.2f}")
print(f"Ridge Polynomial MAE: {np.mean(mae_poly):.2f}")
print(f"LSTM MAE: {np.mean(mae_lstm):.2f}")
