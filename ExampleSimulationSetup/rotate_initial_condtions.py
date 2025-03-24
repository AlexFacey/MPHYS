import numpy as np

def rotate_coordinates(initial_conditions, theta_deg):
    """
    Rotates position and velocity vectors by a given angle about the x-axis.
    """
    x, y, z, vx, vy, vz = initial_conditions
    position = np.array([x, y, z])
    velocity = np.array([vx, vy, vz])

    theta = np.radians(theta_deg)
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    rotated_position = np.dot(R_x, position)
    rotated_velocity = np.dot(R_x, velocity)
    rotated_initial_conditions = np.concatenate((rotated_position, rotated_velocity))
    rotated_initial_conditions = rotated_initial_conditions.tolist()
    return rotated_initial_conditions

