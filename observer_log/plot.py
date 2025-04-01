import matplotlib.pyplot as plt
import json

# Load your data (assuming it's stored as JSON in a separate file or a long string)
# For this example, let's assume you've saved the JSON list to a file called `data.json`
with open("episode_0.json", "r") as f:
    data = json.load(f)

# Extract the values for plotting
cart_pos = [step['cart_pos'] for step in data]
pole_angle = [step['pole_angle'] for step in data]
cart_vel = [step['cart_vel'] for step in data]
pole_vel = [step['pole_vel'] for step in data]
timesteps = list(range(len(data)))

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(timesteps, cart_pos)
plt.title('Cart Position')
plt.xlabel('Timestep')
plt.ylabel('Position (m)')

plt.subplot(2, 2, 2)
plt.plot(timesteps, pole_angle)
plt.title('Pole Angle')
plt.xlabel('Timestep')
plt.ylabel('Angle (rad)')

plt.subplot(2, 2, 3)
plt.plot(timesteps, cart_vel)
plt.title('Cart Velocity')
plt.xlabel('Timestep')
plt.ylabel('Velocity (m/s)')

plt.subplot(2, 2, 4)
plt.plot(timesteps, pole_vel)
plt.title('Pole Angular Velocity')
plt.xlabel('Timestep')
plt.ylabel('Angular Velocity (rad/s)')

plt.tight_layout()
plt.show()
