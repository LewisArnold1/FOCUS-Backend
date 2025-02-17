import numpy as np

class FixationSaccadeDetector:
    def __init__(self):
        self.prev_left_iris = None
        self.prev_right_iris = None
        self.prev_time = None

    def process_eye_movements(self, left_iris, right_iris, dpi, timestamp_dt):
        dt = self._compute_timestep(timestamp_dt)
        left_velocity, right_velocity = self._compute_velocity(left_iris, right_iris, dpi, timestamp_dt)

    def _compute_velocity(self, left_iris, right_iris, dpi, dt):
        if dt == 0:
            return 0, 0

        left_velocity = None
        right_velocity = None

        # Calculate left iris velocity if possible
        if left_iris is not None and self.prev_left_iris is not None:
            left_velocity = (np.linalg.norm(np.array(left_iris) - np.array(self.prev_left_iris))) / (dpi*dt)

        # Calculate right iris velocity if possible
        if right_iris is not None and self.prev_right_iris is not None:
            right_velocity = (np.linalg.norm(np.array(right_iris) - np.array(self.prev_right_iris))) / (dpi*dt)

        # Handle cases where one or both velocities are None
        if left_velocity is None and right_velocity is not None:
            left_velocity = right_velocity  # Approximate left velocity with right velocity
        elif right_velocity is None and left_velocity is not None:
            right_velocity = left_velocity  # Approximate right velocity with left velocity
        elif left_velocity is None and right_velocity is None:
            # Use previous velocities if neither can be calculated
            left_velocity = 0
            right_velocity = 0

        return left_velocity, right_velocity


