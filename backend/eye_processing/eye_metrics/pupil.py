import cv2
import numpy as np

class PupilProcessor:
    def __init__(self):
        pass

    def process_pupil(self, frame, left_eye, right_eye):
        self.frame = frame

        # Convert left eye image to binary
        left_eye = np.array(left_eye)
        left = self.crop_eyes(left_eye)
        left_grey = self.convert_to_greyscale(left)
        left_binary = self.convert_to_binary(left_grey)

        # Draw a rectangle around the eye
        x_min, y_min, x_max, y_max = self.find_bounding_box(left_eye)
        cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        # Call the new display function
        self.display_images_in_grid(self.frame, left, left_grey, left_binary)
        
        return None
    
    def crop_eyes(self, left_eye):
        x_min, y_min, x_max, y_max = self.find_bounding_box(left_eye)

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
        
        return (x_min, y_min, x_max, y_max)

    def convert_to_greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def convert_to_binary(self, image, threshold=40):
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def resize_with_aspect_ratio(self, image, target_size):
        # Convert grayscale/binary images to BGR for consistency
        if len(image.shape) == 2:  # Check if single-channel image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Extract original and target dimensions
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor to fit the image in the target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image while maintaining aspect ratio
        resized = cv2.resize(image, (new_w, new_h))

        # Create a blank canvas (black background) with the target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # Calculate padding to center the image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Place the resized image in the center of the canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def display_images_in_grid(self, full_frame, left_cropped, left_greyscale, left_binary):
        # Define target size based on the full frame dimensions
        target_size = (self.frame.shape[1], self.frame.shape[0])

        # Resize images to fill their grid cells while maintaining aspect ratio
        full_frame_resized = self.resize_with_aspect_ratio(full_frame, target_size)
        left_cropped_resized = self.resize_with_aspect_ratio(left_cropped, target_size)
        left_greyscale_resized = self.resize_with_aspect_ratio(left_greyscale, target_size)
        left_binary_resized = self.resize_with_aspect_ratio(left_binary, target_size)
        
        # Combine images into a 2x2 grid
        top_row = np.hstack((full_frame_resized, left_cropped_resized))
        bottom_row = np.hstack((left_greyscale_resized, left_binary_resized))
        grid = np.vstack((top_row, bottom_row))

        # Display the combined grid
        cv2.imshow("Pupil Processor Output", grid)

        # # Wait for 'q' to close the window
        # print("Press 'q' to exit...")
        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # cv2.destroyAllWindows()
