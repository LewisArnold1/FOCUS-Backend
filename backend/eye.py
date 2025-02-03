import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Load the image
image_path = '/Users/lewisarnold1/Desktop/ey2.jpeg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the image
# Step 1: Apply Histogram Equalization
gray_eq = cv2.equalizeHist(gray)

# Step 2: Reduce reflections by thresholding
_, mask = cv2.threshold(gray_eq, 190, 255, cv2.THRESH_BINARY)
gray_eq[mask == 255] = gray_eq.min()

# Step 3: Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray_eq, (7, 7), 2)

# Hough Circle Transform parameters
hough_params = [100, 15, 200, 1000000]
hough_circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=hough_params[0],
    param2=hough_params[1],
    minRadius=hough_params[2],
    maxRadius=hough_params[3]
)

# Validate detected circles
if hough_circles is None:
    print("No circles detected! Try adjusting parameters.")
else:
    hough_circles = np.uint16(np.around(hough_circles))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    
    for circle in hough_circles[0, :]:
        x, y, r = circle

        # Filter small or irrelevant circles
        if r < 40 or r > 100:
            continue

        # Crop the iris region
        if 0 <= y - r and y + r < image.shape[0] and 0 <= x - r and x + r < image.shape[1]:
            iris = image[y - r:y + r, x - r:x + r]
            cv2.imwrite('/Users/lewisarnold1/Desktop/iris_detected.jpeg', iris)
        else:
            print(f"Cropping region out of bounds for circle: x={x}, y={y}, r={r}")
        
        # Draw the detected circle
        circ = Circle((x, y), r, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(circ)
    
    plt.title("Detected Iris (After Preprocessing)")
    plt.axis("off")
    plt.show()
