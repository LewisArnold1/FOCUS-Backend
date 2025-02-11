import numpy as np
import matplotlib.pyplot as plt

def generate_random_decreasing_curve(duration=10, sample_rate=10, initial_blink_rate=20):
    # Generate time intervals based on the duration and sample rate
    time = np.linspace(0, duration, duration * sample_rate)

    # Define the parameters for the reference function: decay and oscillation
    k1, k2, k3 = 0.15, 0.3, 0.2  # Different decay rates
    C = initial_blink_rate * 0.4  # Secondary decay component
    A = initial_blink_rate * 0.2  # Amplitude of oscillations
    f = 1.5  # Frequency of oscillations

    # Create the reference curve with a combination of exponential decay and sinusoidal oscillation
    reference_curve = (initial_blink_rate * np.exp(-k1 * time) + 
                       C * np.exp(-k2 * time) + 
                       A * np.sin(f * time) * np.exp(-k3 * time))

    # Generate the noisy blink rate by adding random fluctuations
    blink_rate = np.zeros_like(time)  # Initialize the blink_rate array with zeros
    blink_rate[0] = initial_blink_rate  # Set the initial blink rate

    for i in range(1, len(time)):
        # Simulate a random decrease in the blink rate at each time step
        step = np.random.uniform(0.1, 1.5)  
        blink_rate[i] = max(blink_rate[i - 1] - step, 0)  # Ensure blink rate doesn't go negative
        blink_rate[i] += np.random.normal(0, 1)  # Add random noise
        blink_rate[i] = np.clip(blink_rate[i], 0, 20)  # Limit the blink rate to a maximum of 20

    return time, blink_rate, reference_curve

# Generate the time, blink rate, and reference curve data
time, blink_rate, reference_curve = generate_random_decreasing_curve(duration=10, sample_rate=10)

# Compute the Mean Absolute Error (MAE) at each time step between the blink rate and the reference curve
mae_per_time = np.abs(blink_rate - reference_curve)

# Plot the blink rate and reference curve in the first subplot
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, blink_rate, label='Noisy Blink Rate', color='blue', alpha=0.7)
plt.plot(time, reference_curve, label='Reference Curve', color='red', linestyle='dashed')
plt.fill_between(time, blink_rate, reference_curve, color='gray', alpha=0.3, label='Error')  # Fill the error area
plt.xlabel('Time (minutes)')
plt.ylabel('Blink Rate (blinks per minute)')
plt.title('Blink Rate vs.  Reference Curve')
plt.legend()

# Plot the error (MAE) over time in the second subplot
plt.subplot(2, 1, 2)
plt.plot(time, mae_per_time, label='MAE at Each Time Step', color='green')
plt.xlabel('Time (minutes)')
plt.ylabel('Error (MAE)')
plt.title('Error Over Time')
plt.legend()

# Adjust layout for better visualization
plt.tight_layout()
plt.show()
