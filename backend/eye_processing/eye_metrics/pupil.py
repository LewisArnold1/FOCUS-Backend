import cv2
import numpy as np
import time
from scipy.interpolate import splprep, splev

class PupilProcessor:
    def __init__(self):
        self.pupil_centre = None
        self.pupil_radius = None

    def _resize_with_aspect_ratio(self, image, target_size):
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

    def _display_images_in_grid(self, full_frame, left_cropped, left_greyscale, left_binary):
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

    def process_pupil(self, frame, eye_points):
        self.frame = frame
        self.eye_points = np.array(eye_points)
        self.pupil_centre, self.pupil_radius = self.detect_pupil()

        return self.pupil_centre, self.pupil_radius

    def detect_pupil(self):
        cropped = self.crop_eyes_spline(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        contrast = self.enhance_contrast(grey, clip_limit=8.0, tile_grid_size=(1, 1))
        no_reflection = self.remove_reflections(contrast, grey)
        binary = self.convert_to_binary(no_reflection)
        contours, center, radius = self.process_convex_arc(binary, grey)
        # self._display_images_in_grid(contrast, no_reflection, binary, contours)
        return center, radius

    def crop_eyes_spline(self, eye_points, smoothing_factor=5.0, shift=3):
        # Extract x and y coordinates of the points
        points = np.squeeze(eye_points)
        x = points[:, 0]
        y = points[:, 1] - shift

        # Fit a closed B-spline through the points with a smoothing factor
        tck, _ = splprep([x, y], s=smoothing_factor, per=True)  # `smoothing_factor` controls tightness
        u_fine = np.linspace(0, 1, 100)  # Generate finer points for smoothness
        x_smooth, y_smooth = splev(u_fine, tck)

        # Convert the smooth curve back to integer coordinates
        smooth_curve = np.array([np.round(x_smooth).astype(int), np.round(y_smooth).astype(int)]).T

        # Create a blank mask the same size as the frame
        mask = np.zeros_like(self.frame[:, :, 0])  # Single-channel mask (grayscale)

        # Fill the smooth curve on the mask
        cv2.fillPoly(mask, [smooth_curve], 255)

        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        # Extract the bounding rectangle of the smooth curve
        x_min, y_min, w, h = cv2.boundingRect(smooth_curve)

        # Crop the bounding rectangle and include only the masked region
        cropped_eye = masked_frame[y_min:y_min + h, x_min:x_min + w]

        return cropped_eye, smooth_curve

    def convert_to_greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def convert_to_binary(self, image, threshold=30):
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image
    
    def remove_reflections(self, contrast, image):

        # Threshold to find bright areas
        _, bright_regions = cv2.threshold(contrast, 180, 255, cv2.THRESH_BINARY)

        # Filter bright regions based on contour size
        contours, _ = cv2.findContours(bright_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter reflections: small bright spots only
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Inpaint the small reflections using the mask
        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted_image
    
    def enhance_contrast(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def draw_largest_contour(self, binary_image, grey_image):
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
        # Sort contours by area in descending order
        largest_contour = self.find_largest_contour(contours)

        if largest_contour is None:
            print("No contours found")
            return grey_image  # Return the original image if no contours are detected

        # Create a mask for the largest contour (black on white background)
        largest_contour_mask = np.ones_like(binary_image) * 255  # Start with a white mask

        cv2.drawContours(grey_image, [largest_contour], -1, (0, 0, 255), 1) # Draw the second largest contour on the grey image
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 0, thickness=cv2.FILLED)  # Draw largest contour in black

        return largest_contour_mask
    
    def find_largest_contour(self, contours):
        # Sort contours by area in descending order
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        
        if len(contours) <= 1:
            return None
        
        return contours[1]
    
    def calculate_curvature(self, contour_points):
        """
        Calculate the curvature at each point of the contour using neighboring points.
        """
        curvatures = []
        for i in range(1, len(contour_points) - 1):
            # Previous, current, and next points
            prev_point = contour_points[i - 1]
            current_point = contour_points[i]
            next_point = contour_points[i + 1]

            # Calculate vectors
            vec1 = current_point - prev_point
            vec2 = next_point - current_point

            # Calculate angle between vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:  # Avoid division by zero
                curvatures.append(0)
                continue

            cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability

            angle = np.arccos(cos_theta)  # Angle in radians
            curvature = angle / norm1  # Curvature is proportional to angle divided by arc length
            curvatures.append(curvature)

        # Add 0 curvature for first and last points (can't calculate curvature there)
        curvatures = [0] + curvatures + [0]
        return np.array(curvatures)

    def extract_longest_convex_arc(self, contour_points, curvatures):
        """
        Extract the longest arc with continuous positive curvature (convex region).
        """
        longest_arc = []
        current_arc = []

        for i in range(len(curvatures)):
            if curvatures[i] > 0:  # Convex region
                current_arc.append(contour_points[i])
            else:
                if len(current_arc) > len(longest_arc):  # Check if current arc is longer
                    longest_arc = current_arc
                current_arc = []  # Reset for the next arc

        # Check the last arc
        if len(current_arc) > len(longest_arc):
            longest_arc = current_arc

        return np.array(longest_arc)

    def fit_circle_to_arc(self, arc_points, grey_image):
        """
        Fit a circle to the arc points and draw it on the image.
        """
        if len(arc_points) < 3:
            print("Not enough points to fit a circle.")
            return grey_image, None, None

        # Fit a circle using cv2.minEnclosingCircle
        (x, y), radius = cv2.minEnclosingCircle(arc_points)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the fitted circle
        cv2.circle(grey_image, center, radius, (0, 255, 0), 1)  # Green circle

        return grey_image, center, radius

    def process_convex_arc(self, binary_image, grey_image):
        """
        Detect the contour, calculate curvature, and fit a circle to the longest convex arc.
        """
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = self.find_largest_contour(contours)

        if largest_contour is None:
            print("No contours found")
            return grey_image
        
        # Flatten the contour and calculate curvature
        contour_points = largest_contour[:, 0, :]  # Reshape to (N, 2)
        curvatures = self.calculate_curvature(contour_points)

        # Extract the longest convex arc
        longest_arc = self.extract_longest_convex_arc(contour_points, curvatures)

        # Fit a circle to the longest arc and draw only the final circle
        result_image, center, radius = self.fit_circle_to_arc(longest_arc, grey_image)

        return result_image, center, radius

class PupilTracker:
    def __init__(self):
        self.prev_left_pupil = None
        self.prev_right_pupil = None
        self.prev_left_radius = None
        self.prev_right_radius = None
        self.prev_time = None

        self.kalman_left = self._initialise_kalman_filter()
        self.kalman_right = self._initialise_kalman_filter()

    def _initialise_kalman_filter(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],  
                                            [0, 1, 0, 1],  
                                            [0, 0, 1, 0],  
                                            [0, 0, 0, 1]], dtype=np.float32)
        kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        return kalman
    
    def update_pupil(self, left_pupil, left_radius, right_pupil, right_radius):
        current_time = time.time()
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_left_pupil, self.prev_right_pupil = left_pupil, right_pupil
            self.prev_left_radius, self.prev_right_radius = left_radius, right_radius
            return left_pupil, left_radius, right_pupil, right_radius

        dt = current_time - self.prev_time
        left_velocity, right_velocity = self._compute_velocity(left_pupil, right_pupil, dt)
        measurement_noise = self._adjust_measurement_confidence(left_velocity, right_velocity)

        self.kalman_left.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kalman_right.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        left_pupil_smoothed = self._apply_kalman_filter(self.kalman_left, left_pupil)
        right_pupil_smoothed = self._apply_kalman_filter(self.kalman_right, right_pupil)

        self.prev_left_pupil, self.prev_right_pupil = left_pupil, right_pupil
        self.prev_left_radius, self.prev_right_radius = left_radius, right_radius
        self.prev_time = current_time

        return left_pupil_smoothed, left_radius, right_pupil_smoothed, right_radius
    
    def _compute_velocity(self, left_pupil, right_pupil, dt):
        if dt == 0 or self.prev_left_pupil is None or self.prev_right_pupil is None:
            return 0, 0

        left_velocity = np.linalg.norm(np.array(left_pupil) - np.array(self.prev_left_pupil)) / dt
        right_velocity = np.linalg.norm(np.array(right_pupil) - np.array(self.prev_right_pupil)) / dt

        return left_velocity, right_velocity

    def _adjust_measurement_confidence(self, left_velocity, right_velocity):
        base_noise = 0.1
        high_speed_threshold = 0.2
        velocity_difference_threshold = 0.1

        if abs(left_velocity - right_velocity) > velocity_difference_threshold or max(left_velocity, right_velocity) > high_speed_threshold:
            return base_noise * 5
        return base_noise
    
    def _apply_kalman_filter(self, kalman, pupil_position):
        if pupil_position is None:
            return None

        measured = np.array([[np.float32(pupil_position[0])], [np.float32(pupil_position[1])]])
        kalman.correct(measured)
        predicted = kalman.predict()
        return (int(predicted[0]), int(predicted[1]))


