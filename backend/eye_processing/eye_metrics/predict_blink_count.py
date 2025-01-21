from sklearn.linear_model import LinearRegression

def predict_blink_count(user):
    from eye_processing.models import SimpleEyeMetrics
    # Retrieve all blink counts and sort by timestamp
    metrics = SimpleEyeMetrics.objects.filter(user=user).order_by('timestamp')

    if len(metrics) < 1:  #atleast number of datapoints
        print("Not enough data to predict the next blink count.")
        return None

    # Prepare data
    blink_counts = [metric.blink_count for metric in metrics]
    blink_lags = [blink_counts[i-1] if i > 0 else 0 for i in range(len(blink_counts))]

    # simple linear regression with lag feature
    X = [[lag] for lag in blink_lags[1:]]  # Lag 
    y = blink_counts[1:]  # Target is the current blink count

    
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next blink count using the last blink count as input
    predicted_blink_count = model.predict([[blink_counts[-1]]])[0]

    return predicted_blink_count
