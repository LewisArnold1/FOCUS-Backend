import numpy as np
import matplotlib.pyplot as plt

def generate_random_decreasing_curve(duration=10, sample_rate=10):
    """
    Generates a mock blink rate curve that generally decreases over time
    but does not follow a standard function. Instead, it uses random perturbations.

    Parameters:
    - duration: Total time in minutes
    - sample_rate: Number of samples per minute

    Returns:
    - time: Array of time values
    - blink_rate: Array of corresponding blink rates
    """
    time = np.linspace(0, duration, duration * sample_rate)
    blink_rate = np.zeros_like(time)

    # Start at a random initial value between 15 and 20 blinks per minute
    blink_rate[0] = np.random.uniform(15, 20)

    for i in range(1, len(time)):
        # Random decreasing step size
        step = np.random.uniform(0.1, 1.5)
        blink_rate[i] = max(blink_rate[i - 1] - step, 0)

        # Add some random noise
        blink_rate[i] += np.random.normal(0, 1)
        blink_rate[i] = np.clip(blink_rate[i], 0, 20)

    return time, blink_rate

# Example usage
time, blink_rate = generate_random_decreasing_curve(duration=10, sample_rate=10)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time, blink_rate, label='Blink Rate', color='blue')
plt.xlabel('Time (minutes)')
plt.ylabel('Blink Rate (blinks per minute)')
plt.title('Mock Blink Rate per Minute with Random Noise')
plt.legend()
plt.show()