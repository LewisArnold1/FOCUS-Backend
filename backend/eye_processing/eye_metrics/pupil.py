import cv2
import numpy as np

class PupilProcessor:
    def __init__(self):
        pass

    def process_pupil(self, frame, left_eye, right_eye):
        self.frame = frame

        # Show the full frame
        cv2.imshow("Frame", self.frame)

        # Convert left eye image to binary
        left_eye = np.array(left_eye)
        left = self.crop_eyes(left_eye)
        left_grey = self.convert_to_greyscale(left)
        left_binary = self.convert_to_binary(left_grey)

        # Display intermediate and final outputs
        cv2.imshow("Left (Cropped)", left)
        cv2.imshow("Left (Grayscale)", left_grey)
        cv2.imshow("Left (Binary)", left_binary)

        # Wait for user input to close windows
        print("Press 'q' to close windows...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Left (Binary)", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        return None
    
    def crop_eyes(self, left_eye):
        x_min, y_min, x_max, y_max = self.find_bounding_box(left_eye)
        
        # Draw a rectangle around the eye
        cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.imshow("Box (Full Frame)", self.frame)

        # Crop the region of interest
        left = self.frame[y_min:y_max, x_min:x_max]
        return left 

    def find_bounding_box(self, eye_points, padding=20):
        # Get min and max x and y values
        x_min = np.min(eye_points[:, 0])
        x_max = np.max(eye_points[:, 0])
        y_min = np.min(eye_points[:, 1])
        y_max = np.max(eye_points[:, 1])
        
        # Add padding
        x_min = max(x_min - padding, 0)  # Ensure x_min doesn't go below 0
        x_max = min(x_max + padding, self.frame.shape[1])  # Ensure x_max doesn't exceed image width
        y_min = max(y_min - padding, 0)  # Ensure y_min doesn't go below 0
        y_max = min(y_max + padding, self.frame.shape[0]) # Ensure y_max doesn't exceed image height
        
        # Return bounding box
        return (x_min, y_min, x_max, y_max)

    def convert_to_greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def convert_to_binary(self, image, threshold=20):
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image
