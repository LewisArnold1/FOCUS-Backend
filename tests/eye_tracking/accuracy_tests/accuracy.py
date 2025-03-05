import cv2
import numpy as np

# Load the image
image_path = "/Users/lewisarnold1/Desktop/Eye.png"  # Update with correct path
output_path = "/Users/lewisarnold1/Desktop/Eye_Tracked.png"  # Path to save the new image
image = cv2.imread(image_path)

# Convert image to HSV for better colour detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert HEX colors to BGR (OpenCV format)
red_bgr = (1, 33, 250)   # FA2101 in BGR
green_bgr = (80, 176, 0) # 00B050 in BGR

# Convert BGR to HSV
red_hsv = cv2.cvtColor(np.uint8([[red_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
green_hsv = cv2.cvtColor(np.uint8([[green_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

# Define colour tolerance
tolerance = 20  

# Create HSV color range masks
lower_red = np.array([max(0, red_hsv[0] - tolerance), 50, 50])
upper_red = np.array([min(179, red_hsv[0] + tolerance), 255, 255])

lower_green = np.array([max(0, green_hsv[0] - tolerance), 50, 50])
upper_green = np.array([min(179, green_hsv[0] + tolerance), 255, 255])

# Create masks
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Function to find centroid of a detected blob
def find_centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        return (x, y)
    return None

# Get centroids of red and green dots
red_dot = find_centroid(mask_red)
green_dot = find_centroid(mask_green)

# Compute Euclidean distance and save edited image
if red_dot and green_dot:
    distance = np.sqrt((red_dot[0] - green_dot[0]) ** 2 + (red_dot[1] - green_dot[1]) ** 2)
    
    # Print (x, y) coordinates and distance
    print(f"Green Dot (Ground Truth): {green_dot}")
    print(f"Red Dot (Tracker Output): {red_dot}")
    print(f"Pixel Distance Between Dots: {distance:.2f}")

    # Draw markers on image
    cv2.circle(image, red_dot, 5, (0, 0, 255), -1)  
    cv2.circle(image, green_dot, 5, (255, 0, 0), -1) 
    cv2.line(image, red_dot, green_dot, (255, 255, 255), 2) 

    # Save the edited image
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to {output_path}")

    # Show the result
    cv2.imshow("Tracking Accuracy", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not detect both dots.")
