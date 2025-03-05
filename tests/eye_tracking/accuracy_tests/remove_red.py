import cv2
import numpy as np

# File paths
image_path = "/Users/lewisarnold1/Desktop/Screenshot 2025-03-04 at 14.07.01.png"
output_image_path = "/Users/lewisarnold1/Desktop/Screenshot_Cleaned.png"

# Load the image
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Convert to HSV for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red color range in HSV
lower_red1 = np.array([0, 100, 100])    # Lower bound for red
upper_red1 = np.array([10, 255, 255])   # Upper bound for red
lower_red2 = np.array([170, 100, 100])  # Second red range (due to HSV wrap-around)
upper_red2 = np.array([180, 255, 255])

# Create masks for red
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2  # Combine both masks

# Replace red with dark gray (RGB = (50, 50, 50), OpenCV uses BGR)
dark_gray = (50, 50, 50)
image[np.where(mask_red > 0)] = dark_gray

# Apply Gaussian blur to blend edges
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Save the cleaned image
cv2.imwrite(output_image_path, blurred_image)

# Display the result
cv2.imshow("Red Replaced with Dark Gray", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Cleaned image saved at: {output_image_path}")
