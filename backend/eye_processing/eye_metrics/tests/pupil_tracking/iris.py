import cv2
import numpy as np

class IrisProcessor:
    def __init__(self):
        self.frame = None
        self.eye_points = None

    def process_iris(self, frame, eye_points, hough_params=None, binary_threshold=50, ratio_thresh=0.6):
        self.frame = frame
        self.eye_points = np.array(eye_points)

        hough_params = hough_params if hough_params else [180, 5, 5, 17]

        cropped_eye = self.crop_eyes(self.eye_points, padding=5)
        gray_eye = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2GRAY)
        binary_eye = self.apply_threshold(gray_eye, binary_threshold)

        hough_circles = self.detect_hough_circles(binary_eye, hough_params)
        if hough_circles is None:
            print("No circles detected!")
            return None

        valid_circles = self.filter_circles(hough_circles, binary_eye, ratio_thresh)
        if len(valid_circles) == 0:
            print("No valid circles detected after filtering!")
            return None

        selected_circle = valid_circles[0]

        # Frame with the detected circle
        frame_with_circle = cropped_eye.copy()
        self.draw_circle(frame_with_circle, selected_circle)

        # Display images in a grid
        self.display_images_in_grid(
            full_frame=self.frame,
            cropped_frame=cropped_eye,
            frame_with_circle=frame_with_circle,
            additional_frame=binary_eye
        )

        return selected_circle

    def crop_eyes(self, eye_points, padding=0):
        x_min, y_min, x_max, y_max = self.find_bounding_box(eye_points, padding)
        return self.frame[y_min:y_max, x_min:x_max]

    def find_bounding_box(self, eye_points, padding=0):
        x_min = max(np.min(eye_points[:, 0]) - padding, 0)
        x_max = min(np.max(eye_points[:, 0]) + padding, self.frame.shape[1])
        y_min = max(np.min(eye_points[:, 1]) - padding, 0)
        y_max = min(np.max(eye_points[:, 1]) + padding, self.frame.shape[0])
        return x_min, y_min, x_max, y_max

    def apply_threshold(self, gray_image, threshold):
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image

    def detect_hough_circles(self, binary_image, hough_params):
        hough_circles = cv2.HoughCircles(
            binary_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=hough_params[0],
            param2=hough_params[1],
            minRadius=hough_params[2],
            maxRadius=hough_params[3]
        )
        return np.uint16(np.around(hough_circles)) if hough_circles is not None else None

    def filter_circles(self, circles, binary_image, ratio_thresh):
        valid_circles = []
        for circle in circles[0, :]:
            circle_x, circle_y, circle_r = circle
            mask = np.zeros_like(binary_image, dtype=np.uint8)
            cv2.circle(mask, (circle_x, circle_y), circle_r, 255, thickness=-1)
            masked_region = cv2.bitwise_and(binary_image, binary_image, mask=mask)

            black_pixels = np.sum(masked_region == 0)
            white_pixels = np.sum(masked_region == 255)
            ratio = black_pixels / (black_pixels + white_pixels) if (black_pixels + white_pixels) > 0 else 0

            if ratio > ratio_thresh:
                valid_circles.append(circle)

        return valid_circles

    def draw_circle(self, image, circle):
        circle_x, circle_y, circle_r = circle
        cv2.circle(image, (circle_x, circle_y), circle_r, (0, 255, 0), 2)
        cv2.circle(image, (circle_x, circle_y), 2, (0, 0, 255), 3)

    def resize_with_aspect_ratio(self, image, target_size):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return canvas

    def display_images_in_grid(self, full_frame, cropped_frame, frame_with_circle, additional_frame=None):
        target_size = (self.frame.shape[1], self.frame.shape[0])

        full_frame_resized = self.resize_with_aspect_ratio(full_frame, target_size)
        cropped_frame_resized = self.resize_with_aspect_ratio(cropped_frame, target_size)
        frame_with_circle_resized = self.resize_with_aspect_ratio(frame_with_circle, target_size)
        additional_frame_resized = (
            self.resize_with_aspect_ratio(additional_frame, target_size) if additional_frame is not None else None
        )

        top_row = np.hstack((full_frame_resized, cropped_frame_resized))
        bottom_row = (
            np.hstack((frame_with_circle_resized, additional_frame_resized))
            if additional_frame_resized is not None
            else frame_with_circle_resized
        )
        grid = np.vstack((top_row, bottom_row))

        cv2.imshow("Iris Processor Output", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
