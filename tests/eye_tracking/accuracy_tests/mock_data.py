import numpy as np
import pandas as pd

# Define save path (same as image location)
save_path = "/Users/lewisarnold1/Desktop/eye_tracking_data.csv"

# Number of frames needed
num_frames = 100

# Define x and y value ranges
x_range = (75, 625)
y_range = (90, 250)

# Define mean and standard deviation for ground truth values
truth_mean = np.array([391, 135])
truth_std = 100

# Function to generate valid ground truth points
def generate_valid_points(num_frames, mean, std, x_range, y_range):
    valid_x = []
    valid_y = []

    while len(valid_x) < num_frames:
        # Generate a candidate point
        tx = int(np.random.normal(mean[0], std))
        ty = int(np.random.normal(mean[1], std))

        # Check if the point is strictly within the bounding box (not on edges)
        if x_range[0] < tx < x_range[1] and y_range[0] < ty < y_range[1]:
            valid_x.append(tx)
            valid_y.append(ty)

    return np.array(valid_x), np.array(valid_y)

# Generate exactly 100 valid ground truth points
truth_x, truth_y = generate_valid_points(num_frames, truth_mean, truth_std, x_range, y_range)

# Define mean and standard deviation for predicted values' distance from ground truth
pred_distance_mean = 30
pred_distance_std = 15

# Function to generate valid predicted points
def generate_valid_predicted_points(truth_x, truth_y, x_range, y_range, pred_distance_mean, pred_distance_std):
    pred_x = []
    pred_y = []
    euclidean_distances = []

    for tx, ty in zip(truth_x, truth_y):
        while True:  # Keep generating until a valid point is found
            # Generate a random distance with mean and std deviation
            distance = max(1, np.random.normal(pred_distance_mean, pred_distance_std))  # Ensure distance is positive
            
            # Generate a random angle in radians
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Compute predicted coordinates
            px = int(tx + distance * np.cos(angle))
            py = int(ty + distance * np.sin(angle))

            # Check if the predicted point lies within the bounding box (not on edges)
            if x_range[0] < px < x_range[1] and y_range[0] < py < y_range[1]:
                pred_x.append(px)
                pred_y.append(py)
                euclidean_distances.append(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))
                break  # Valid point found, break out of loop

    return np.array(pred_x), np.array(pred_y), np.array(euclidean_distances)

# Generate exactly 100 valid predicted points
pred_x, pred_y, euclidean_distances = generate_valid_predicted_points(
    truth_x, truth_y, x_range, y_range, pred_distance_mean, pred_distance_std
)

# Compute overall accuracy (1 - normalized error)
accuracy = 1 - (np.mean(euclidean_distances) / np.sqrt(x_range[1]**2 + y_range[1]**2))

# Create a DataFrame
df = pd.DataFrame({
    "Frame": np.arange(1, num_frames + 1),
    "Truth_X": truth_x,
    "Truth_Y": truth_y,
    "Predicted_X": pred_x,
    "Predicted_Y": pred_y,
    "Euclidean_Distance": euclidean_distances
})

# Save to CSV
df.to_csv(save_path, index=False)

# Print confirmation
print(f"CSV file saved at: {save_path}")
print(f"Overall Accuracy: {accuracy:.2%}")
