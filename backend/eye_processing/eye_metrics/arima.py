from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def predict_blink_count(user):
    from eye_processing.models import SimpleEyeMetrics
    # Retrieve all blink counts and sort by timestamp
    metrics = SimpleEyeMetrics.objects.filter(user=user).order_by('timestamp')

    if len(metrics) < 10:  # At least number of datapoints required
        print("Not enough data to predict the next blink count.")
        return None

    # Prepare data
    blink_counts = [metric.blink_count for metric in metrics]

    # Ensure the blink counts are in the form of a time series (e.g., a numpy array)
    blink_counts = np.array(blink_counts)

    # Fit an ARIMA model
    try:
        model = ARIMA(blink_counts, order=(1, 1, 1))  # ARIMA(p, d, q) with p=1, d=1, q=1 (adjust as needed)
        model_fit = model.fit()

        # Predict the next blink count
        predicted_blink_count = model_fit.forecast(steps=1)[0]
        
        return predicted_blink_count
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None
