import numpy as np
import matplotlib.pyplot as plt
import random
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# Generate noisy decreasing blink rate
def generate_random_decreasing_curve(duration=10, sample_rate=10, initial_blink_rate=20):
    time = [i / sample_rate for i in range(duration * sample_rate)]
    
    # Generate noisy blink rate
    blink_rate = [0] * len(time)
    blink_rate[0] = initial_blink_rate
    
    for i in range(1, len(time)):
        step = random.uniform(0.1, 1.5)
        blink_rate[i] = max(blink_rate[i - 1] - step, 0)
        blink_rate[i] += random.gauss(0, 1)  
        blink_rate[i] = max(min(blink_rate[i], 20), 0)
    
    return time, blink_rate

# ARIMA Prediction
def predict_arima(blink_data, steps=1):
    if len(blink_data) < 5:
        return [max(blink_data[-1], 0)]  # Ensure non-negative output
    
    try:
        model = ARIMA(blink_data, order=(5, 1, 0))
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=steps)
        return np.clip(prediction, 0, 20)  # Ensure predictions are between 0 and 20
    except Exception as e:
        print(f"ARIMA error: {e}")
        return [max(blink_data[-1], 0)]

# Ridge Regression for better Polynomial Fit
def predict_ridge_polynomial(blink_data, degree=4, prediction_steps=40, alpha=1.0):
    X = [[i] for i in range(len(blink_data))]
    y = blink_data
    
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    # Apply Ridge Regression to avoid overfitting
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    
    # Make predictions for the next steps
    X_pred = [[i] for i in range(len(blink_data), len(blink_data) + prediction_steps)]
    X_pred_poly = poly.transform(X_pred)
    
    return np.clip(model.predict(X_pred_poly), 0, 20)  # Clip output to valid range

# LSTM Prediction
def predict_lstm(blink_data, prediction_steps=40, look_back=10):
    X_train, y_train = [], []
    
    for i in range(len(blink_data) - look_back):
        X_train.append(blink_data[i:i+look_back])
        y_train.append(blink_data[i+look_back])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Model definition
    model = Sequential([
        Input(shape=(look_back, 1)),  # Fix the Keras warning
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Training the model
    print("Starting model training...")
    model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)  # Reduced epochs for faster debugging
    print("Training complete!")

    # Predict Next Steps
    last_sequence = np.array(blink_data[-look_back:]).reshape(1, look_back, 1)
    lstm_predictions = []

    for _ in range(prediction_steps):
        next_pred = model.predict(last_sequence, verbose=0)[0][0]
        next_pred = max(0, min(next_pred, 20))  # Ensure 0 ≤ blink_rate ≤ 20
        lstm_predictions.append(next_pred)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)

    return lstm_predictions


# Generate Data
time, blink_rate = generate_random_decreasing_curve(duration=10, sample_rate=10)

# Define the split time (first 6 minutes for training, last 4 minutes for prediction)
train_duration = 6
train_samples = train_duration * 10  # 10 samples per minute

# Train on the first 6 minutes of data
train_blink_rate = blink_rate[:train_samples]
train_time = time[:train_samples]

# Predict the next 4 minutes using ARIMA, Ridge Polynomial Regression, and LSTM
arima_predictions = predict_arima(train_blink_rate, steps=40)
ridge_poly_predictions = predict_ridge_polynomial(train_blink_rate, degree=4, prediction_steps=40, alpha=1.0)  # Ridge Regression
lstm_predictions = predict_lstm(train_blink_rate, prediction_steps=40)

# Create time for the prediction period (next 4 minutes)
prediction_time = time[train_samples:train_samples + 40]

# Polynomial Regression for Line of Best Fit (for the entire 10 minutes)
def polynomial_best_fit(time, blink_rate, degree=4):
    model = PolynomialFeatures(degree)
    X_poly = model.fit_transform(np.array(time).reshape(-1, 1))
    
    # Apply linear regression to the polynomial features
    lin_reg = Ridge()
    lin_reg.fit(X_poly, blink_rate)
    
    blink_rate_pred = lin_reg.predict(X_poly)
    
    return blink_rate_pred

# Generate the best fit line for the noisy data (for the entire 10 minutes)
polynomial_fit_full = polynomial_best_fit(time, blink_rate, degree=4)

# Visualization
plt.figure(figsize=(12, 8))

# Plot Blink Rate vs Predictions
plt.subplot(2, 1, 1)
plt.plot(time, blink_rate, label='Noisy Blink Rate', color='blue', alpha=0.7)
plt.plot(prediction_time, arima_predictions, label='ARIMA Predictions', color='green', linestyle='dotted')
plt.plot(prediction_time, ridge_poly_predictions, label='Ridge Polynomial Regression Predictions', color='purple', linestyle='dashdot')
plt.plot(prediction_time, lstm_predictions, label='LSTM Predictions', color='orange', linestyle='solid')
plt.plot(time, polynomial_fit_full, label='Polynomial Best Fit (Entire Data)', color='red', linestyle='--')  # Polynomial fit for entire data
plt.xlabel('Time (minutes)')
plt.ylabel('Blink Rate (blinks per minute)')
plt.title('Blink Rate vs Predictions (First 6 Minutes for Training)')
plt.legend()

# Compute MAE (Mean Absolute Error) between the noisy blink rate and the predictions
mae_arima = [abs(pred - blink_rate[i+train_samples]) for i, pred in enumerate(arima_predictions)]
mae_poly = [abs(pred - blink_rate[i+train_samples]) for i, pred in enumerate(ridge_poly_predictions)]
mae_lstm = [abs(pred - blink_rate[i+train_samples]) for i, pred in enumerate(lstm_predictions)]

# Plot Error Over Time (MAE)
plt.subplot(2, 1, 2)
plt.plot(prediction_time, mae_arima, label='ARIMA Error (MAE)', color='green')
plt.plot(prediction_time, mae_poly, label='Ridge Polynomial Error (MAE)', color='purple')
plt.plot(prediction_time, mae_lstm, label='LSTM Error (MAE)', color='orange')
plt.xlabel('Time (minutes)')
plt.ylabel('Error (MAE)')
plt.title('Prediction Error Over Time')
plt.legend()

plt.tight_layout()
plt.show()
