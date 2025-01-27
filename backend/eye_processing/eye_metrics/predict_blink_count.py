from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def predict_blink_count(user):
    from eye_processing.models import SimpleEyeMetrics
    # Retrieve all blink counts and sort by timestamp
    metrics = SimpleEyeMetrics.objects.filter(user=user).order_by('timestamp')

    if len(metrics) < 10:  #atleast number of datapoints
        print("Not enough data to predict the next blink count.")
        return None

    # Prepare data
    blink_counts = [metric.blink_count for metric in metrics]
    blink_lags = [blink_counts[i-1] if i > 0 else 0 for i in range(len(blink_counts))]

    # simple linear regression with lag feature
    X = [[lag] for lag in blink_lags[1:]]  # Lag 
    y = blink_counts[1:]  # Target is the current blink count

     # Transform features to polynomial features
    poly = PolynomialFeatures(degree=2)  # Adjust degree as needed
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the next blink count using the last blink count as input
    last_blink_count_poly = poly.transform([[blink_counts[-1]]])
    predicted_blink_count = model.predict(last_blink_count_poly)[0]

    return predicted_blink_count
