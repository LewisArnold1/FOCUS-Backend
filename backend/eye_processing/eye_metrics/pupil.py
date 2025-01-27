import cv2
import numpy as np
import os

class PupilProcessor:
    def __init__(self):
        pass

    def process_pupil(self, frame, eye_points):
        self.frame = frame
        self.eye_points = np.array(eye_points)
        
        # Apply method
        # self.method_1_basic_thresholding()
        # self.method_2_reflection_removal()
        # self.method_3_hough_transform()
        # self.method_4_contrast_enhancement()
        # self.method_5_contour_largest_area()
        # self.method_6_convex_hull()
        # self.method_7_super_resolution()
        self.method_8_longest_convex_arc()
        
        return None
    
    def method_1_basic_thresholding(self):
        cropped = self.crop_eyes(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        binary = self.convert_to_binary(grey)
        self.display_images_in_grid(self.frame, cropped, grey, binary)

    def method_2_reflection_removal(self):
        cropped = self.crop_eyes(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        no_reflection = self.remove_reflections(grey)
        binary = self.convert_to_binary(no_reflection)
        self.display_images_in_grid(cropped, grey, no_reflection, binary)

    def method_3_hough_transform(self):
        cropped = self.crop_eyes(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        circles = self.create_hough_circles(grey)
        self.display_images_in_grid(self.frame, cropped, grey, circles)

    def method_4_contrast_enhancement(self):
        cropped = self.crop_eyes(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        contrast = self.enhance_contrast(grey, clip_limit=8.0, tile_grid_size=(1, 1))
        binary = self.convert_to_binary(contrast, threshold=10)
        self.display_images_in_grid(self.frame, cropped, grey, binary)

    def method_5_contour_largest_area(self):
        cropped = self.crop_eyes(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        binary = self.convert_to_binary(grey)
        contours = self.draw_largest_contour(binary, grey)
        self.display_images_in_grid(self.frame, cropped, grey, contours)

    def method_6_convex_hull(self):
        cropped = self.crop_eyes_hull(self.eye_points)
        grey = self.convert_to_greyscale(cropped)
        binary = self.convert_to_binary(grey)
        self.display_images_in_grid(self.frame, cropped, grey, binary)

    def method_7_super_resolution(self):
        cropped = self.crop_eyes(self.eye_points)
        super_resolved = self.enhance_image_with_super_resolution(cropped)
        grey = self.convert_to_greyscale(super_resolved)
        self.display_images_in_grid(self.frame, cropped, super_resolved, grey)

    def method_8_longest_convex_arc(self):
        cropped = self.crop_eyes(self.eye_points, padding=5)
        grey = self.convert_to_greyscale(cropped)
        binary = self.convert_to_binary(grey)
        contours = self.process_convex_arc(binary, grey)
        self.display_images_in_grid(self.frame, cropped, grey, contours)
    
    def crop_eyes(self, left_eye, padding=0):
        x_min, y_min, x_max, y_max = self.find_bounding_box(left_eye, padding=padding)

        # Crop the region of interest
        left = self.frame[y_min:y_max, x_min:x_max]
        return left

    def crop_eyes_hull(self, left_eye):
        # Create the convex hull of the eye points
        hull = cv2.convexHull(left_eye)

        # Create a blank mask the same size as the frame
        mask = np.zeros_like(self.frame[:, :, 0])  # Single-channel mask (grayscale)

        # Fill the convex hull on the mask
        cv2.fillConvexPoly(mask, hull, 255)

        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(self.frame, self.frame, mask=mask)

        # Extract the bounding rectangle of the convex hull
        x, y, w, h = cv2.boundingRect(hull)

        # Crop the bounding rectangle and include only the masked region
        cropped_eye = masked_frame[y:y + h, x:x + w]

        return cropped_eye 

    def find_bounding_box(self, eye_points, padding=0):
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
    
    def convert_to_binary(self, image, threshold=30):
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image
    
    def remove_reflections(self, image):

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Threshold to find bright areas
        _, bright_regions = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

        # Filter bright regions based on contour size
        contours, _ = cv2.findContours(bright_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter reflections: small bright spots only
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Inpaint the small reflections using the mask
        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted_image 

    def create_hough_circles(self, image):
        # c = cv2.HoughCircles(contours, cv2.HOUGH_GRADIENT, 2, image.shape[0]/2)

        blurred = cv2.GaussianBlur(image, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,                   # Inverse resolution ratio
            minDist=20,              # Very small minimum distance between circles
            param1=50,              # Upper threshold for edge detection
            param2=20,              # Lower threshold for center detection (increase sensitivity)
            minRadius=0,            # No lower size limit
            maxRadius=50             # No upper size limit
        )

        circle_image = blurred
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle
                cv2.circle(circle_image, (x, y), radius, (0, 0, 255), 2)  # Red circle
                cv2.circle(circle_image, (x, y), 2, (0, 255, 0), 3)  # Green center

        return circle_image
    
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
        
    def enhance_image_with_super_resolution(self, image):
        
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of pupil.py
        model_path = os.path.join(current_dir, "EDSR_x3.pb")

        # Read the pre-trained model
        sr.readModel(model_path)
        sr.setModel("edsr", 3)  # Use EDSR with a scaling factor of 3

        # Get the original size of the image
        original_size = (image.shape[1], image.shape[0])  # (width, height)

        # Enhance the image with super-resolution
        upscaled_image = sr.upsample(image)

        # Resize the enhanced image back to the original size
        enhanced_image = cv2.resize(upscaled_image, original_size, interpolation=cv2.INTER_CUBIC)

        return enhanced_image
    
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

        print(f"Fitted Circle - Center: {center}, Radius: {radius}")
        return result_image

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
