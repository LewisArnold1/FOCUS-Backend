import cv2
import pandas as pd

# File paths
csv_path = "/Users/lewisarnold1/Desktop/eye_tracking_data.csv"
image_path = "/Users/lewisarnold1/Desktop/Eye.png"
output_image_path = "/Users/lewisarnold1/Desktop/Eye_Truth_Annotated.png"

# Load the CSV file
df = pd.read_csv(csv_path)

# Load the original image
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Loop through each ground truth position and draw blue circles
for _, row in df.iterrows():
    center = (int(row["Truth_X"]), int(row["Truth_Y"]))
    cv2.circle(image, center, 5, (255, 0, 0), -1)  # Blue circles

# Save the annotated image
cv2.imwrite(output_image_path, image)

# Display the result
cv2.imshow("Ground Truth Positions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Annotated image saved at: {output_image_path}")
