import cv2
import numpy as np
from scipy.interpolate import splprep, splev

class IrisProcessor:
    def __init__(self):
        pass

    def process_iris(self, frame, eye_points):
        self.frame = frame
        self.eye_points = np.array(eye_points)
        grey, colour, overall_centroid = self.detect_iris()
        return grey, colour, overall_centroid
    
    def detect_iris(self):
        cropped, mask = self.crop_eyes_spline(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        contrast = self.enhance_contrast(grey, clip_limit=8.0, tile_grid_size=(1, 1))
        colour, overall_centroid = self.iris(contrast, mask)
        return grey, colour, overall_centroid

    def crop_eyes_spline(self, eye_points, smoothing_factor=5.0, shift=0):
        # Extract x and y coordinates of the points
        points = np.squeeze(eye_points)
        x = points[:, 0]
        y = points[:, 1] - shift

        # Fit a closed B-spline through the points with a smoothing factor
        tck, u = splprep([x, y], s=smoothing_factor, per=True)  # `smoothing_factor` controls tightness
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
        cropped_mask = mask[y_min:y_min + h, x_min:x_min + w]

        return cropped_eye, cropped_mask

    def convert_to_greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def enhance_contrast(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def iris(self, contrast, convex_hull_mask):
        # Apply the convex hull mask to the contrast image
        contrast_masked = cv2.bitwise_and(contrast, contrast, mask=convex_hull_mask)

        # Mask for regions above a threshold (red regions)
        bright_mask = (contrast_masked >= 70)

        # Convert grayscale to BGR for colouring
        colour_image = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)

        # Create a binary mask for non-red regions
        dark_mask = ~bright_mask
        binary_dark = (dark_mask * 255).astype(np.uint8)
        binary_dark = cv2.bitwise_and(binary_dark, convex_hull_mask)  # Restrict to convex hull

        # Find contours of non-red regions
        contours, _ = cv2.findContours(binary_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialise variables for overall centroid calculation
        centroids = []
        total_weight = 0
        weighted_sum_x = 0
        weighted_sum_y = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Check if the centroid is inside the mask
                    if convex_hull_mask[cY, cX] > 0:
                        centroids.append((cX, cY))
                        # Accumulate weighted sums for overall centroid
                        total_weight += area
                        weighted_sum_x += cX * area
                        weighted_sum_y += cY * area

        # Calculate the overall centroid
        if total_weight != 0:
            overall_cX = int(weighted_sum_x / total_weight)
            overall_cY = int(weighted_sum_y / total_weight)
            overall_centroid = (overall_cX, overall_cY)
        else:
            overall_centroid = None

        return colour_image, overall_centroid
    
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
        full_frame_resized = self._resize_with_aspect_ratio(full_frame, target_size)
        left_cropped_resized = self._resize_with_aspect_ratio(left_cropped, target_size)
        left_greyscale_resized = self._resize_with_aspect_ratio(left_greyscale, target_size)
        left_binary_resized = self._resize_with_aspect_ratio(left_binary, target_size)
        
        # Combine images into a 2x2 grid
        top_row = np.hstack((full_frame_resized, left_cropped_resized))
        bottom_row = np.hstack((left_greyscale_resized, left_binary_resized))
        grid = np.vstack((top_row, bottom_row))

        # Display the combined grid
        cv2.imshow("Pupil Processor Output", grid)