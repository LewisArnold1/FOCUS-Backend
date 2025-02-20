import numpy as np

class FixationSaccadeDetector:
    def __init__(self, fixation_threshold=0.02):
        self.prev_left_iris = None
        self.prev_right_iris = None
        self.prev_time = None
        self.fixation_threshold = fixation_threshold

    def process_eye_movements(self, left_iris, right_iris, frame_width, frame_height, timestamp_dt):
        dt = self._compute_timestep(timestamp_dt)
        if dt == 0:
            return 0, 0, "fixation"  # Default to fixation when no movement
        
        left_velocity, right_velocity = self._compute_velocity(left_iris, right_iris, frame_width, frame_height, dt)
        
        # Compute the overall velocity by averaging left and right eye velocities
        overall_velocity = (left_velocity + right_velocity) / 2

        # Classify as fixation or saccade
        movement_type = "fixation" if overall_velocity < self.fixation_threshold else "saccade"

        return left_velocity, right_velocity, movement_type

    def _compute_timestep(self, timestamp_dt):
        if self.prev_time is None:
            self.prev_time = timestamp_dt
            return 0  # No movement detected on first frame

        dt = (timestamp_dt - self.prev_time).total_seconds()
        self.prev_time = timestamp_dt  # Update for the next frame
        return dt if dt > 0 else 0

    def _compute_velocity(self, left_iris, right_iris, frame_width, frame_height, dt):
        left_velocity = None
        right_velocity = None

        # Compute velocity in x and y directions separately, then normalize
        if left_iris is not None and self.prev_left_iris is not None:
            dx = (left_iris[0] - self.prev_left_iris[0]) / frame_width
            dy = (left_iris[1] - self.prev_left_iris[1]) / frame_height
            left_velocity = np.linalg.norm([dx, dy]) / dt  # Compute overall velocity

        if right_iris is not None and self.prev_right_iris is not None:
            dx = (right_iris[0] - self.prev_right_iris[0]) / frame_width
            dy = (right_iris[1] - self.prev_right_iris[1]) / frame_height
            right_velocity = np.linalg.norm([dx, dy]) / dt

        # Update previous values
        self.prev_left_iris = left_iris
        self.prev_right_iris = right_iris

        # Handle cases where one or both velocities are None
        if left_velocity is None and right_velocity is not None:
            left_velocity = right_velocity
        elif right_velocity is None and left_velocity is not None:
            right_velocity = left_velocity
        elif left_velocity is None and right_velocity is None:
            left_velocity = 0
            right_velocity = 0

        return left_velocity, right_velocity