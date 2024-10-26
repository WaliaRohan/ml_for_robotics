import matplotlib.pyplot as plt
import numpy as np
from dubinEHF3d import \
    dubinEHF3d  # Assuming you have converted this function separately

# Keep the initial position zero
x1 = 0
y1 = 0
alt1 = 0

psi1 = 20 * np.pi / 180  # initial heading, between [0, 2*pi]

# Change these variables to get different paths
gamma = -30 * np.pi / 180  # climb angle, keep in between [-30 deg, 30 deg]
x2 = 100
y2 = -250

# Keep these constants
steplength = 10  # trajectory discretization level
r_min = 100  # vehicle turn radius

# Call the dubinEHF3d function
path, psi_end, num_path_points = dubinEHF3d(x1, y1, alt1, psi1, x2, y2, r_min, steplength, gamma)

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(path[:, 0], path[:, 1], path[:, 2], 'b.-')  # Path points
ax.plot([x1], [y1], [alt1], 'r*')  # Start point
ax.plot(path[-1][0], path[-1][1], path[-1][2], 'm*')  # End point

# Set labels and grid
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('alt')
ax.grid(True)
plt.axis('equal')
plt.show()
