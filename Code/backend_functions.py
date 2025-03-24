import numpy as np
import os
import math
import glob

def read_orient(filename,velocity=False,nsmth=1001,norder=2):
    """definition to read EXP '.orient.'-style files
    One dictionary is returned.
    """
    columns = ['time','energy','used','xaxis','yaxis','zaxis','xaxisc','yaxisc','zaxisc','xcena','ycena','zcena','xcenr','ycenr','zcenr','xcenc','ycenc','zcenc','xbaryc','ybarc','zbaryc','xdiff','ydiff','zdiff']
    # open the file once to see the structure
    A = np.genfromtxt(filename,skip_header=2)
    Log = dict()
    for indx in range(0,len(columns)):
        Log[columns[indx]] = A[:,indx]

    # what is the primary centre we want to return? select current centre
    Log['xcen'] = Log['xcenc']
    Log['ycen'] = Log['ycenc']
    Log['zcen'] = Log['zcenc']
    Log['rcen'] = np.sqrt(Log['xcen']**2+Log['ycen']**2+Log['zcen']**2)

    return Log

def rotation_matrix_from_vectors(vec1, vec2):

    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:  # vec1 and vec2 are already aligned
        return np.eye(3)
    k_matrix = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotation_matrix = np.eye(3) + k_matrix + k_matrix @ k_matrix * ((1 - c) / (s ** 2))
    return rotation_matrix

def compute_phi1_phi2(positions, velocities,original=False):

    # Compute angular momentum vector
    angular_momentum = np.cross(positions, velocities)
    L = np.mean(angular_momentum, axis=0)  # Average angular momentum
    # Maybe change this to median?
    L_hat = L / np.linalg.norm(L)  # Normalized angular momentum vector

    # Compute rotation matrix to align L with z-axis
    z_axis = np.array([0, 0, 1])  # Target axis
    rotation_matrix = rotation_matrix_from_vectors(L_hat, z_axis)

    # Transform positions into the new frame
    if original==False:
      rotated_positions = positions @ rotation_matrix.T
    else:
      rotated_positions = positions

    x_prime, y_prime, z_prime = rotated_positions.T

    # Compute phi1 and phi2
    phi1 = np.arctan2(y_prime, x_prime)  # Longitude-like angle
    phi2 = np.arcsin(z_prime / np.sqrt(x_prime**2 + y_prime**2 + z_prime**2))  # Latitude-like angle

    mean_sin = np.median(np.sin(phi1))
    mean_cos = np.median(np.cos(phi1))
    phi1_mean = np.arctan2(mean_sin, mean_cos)

    phi1 = phi1 - phi1_mean
    phi1 = (phi1 + np.pi) % (2 * np.pi) - np.pi

    mean_sin = np.median(np.sin(phi2))
    mean_cos = np.median(np.cos(phi2))
    phi2_mean = np.arctan2(mean_sin, mean_cos)

    phi2 = phi2 - phi2_mean
    phi2 = (phi2 + np.pi) % (2 * np.pi) - np.pi

    # maybe remove x_prime, y_prime, z_prime from return
    return x_prime, y_prime, z_prime, phi1, phi2

def get_phi1_phi2(data):

    positions = np.array([data.xpos, data.ypos, data.zpos]).T
    velocities = np.array([data.xvel, data.yvel, data.zvel]).T

    x_prime, y_prime, z_prime, phi1, phi2 = compute_phi1_phi2(positions, velocities)
    return phi1, phi2


def get_phi1_phi2_straight(data, n_sections=10):
    # Create positions and velocities arrays from your data
    positions = np.column_stack((data.xpos, data.ypos, data.zpos))
    velocities = np.column_stack((data.xvel, data.yvel, data.zvel))
    
    # Split the data into n_sections parts (each part corresponding to a local region)
    pos_segments = np.array_split(positions, n_sections)
    vel_segments = np.array_split(velocities, n_sections)
    
    phi1_all = []
    phi2_all = []

    for pos_seg, vel_seg in zip(pos_segments, vel_segments):
        _, _, _, phi1, phi2 = compute_phi1_phi2(pos_seg, vel_seg)
        phi1_all.append(phi1)
        phi2_all.append(phi2)
    
    phi1_local = np.concatenate(phi1_all)
    phi2_local = np.concatenate(phi2_all)
    
    return phi1_local, phi2_local


# THIS FUNCTION IS TOO SLOW
def get_simulation_files(x, y, z, vx, vy, vz, sr, mass, tol=1e-6):
    # Must be a quicker way to do this
    base_dir = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/NewBigCollectionOfSims/'
    matching_folders = []
    
    # Use glob to directly list folders that have an "=" in their name
    pattern = os.path.join(base_dir, '**', '*=*')
    for folder_path in glob.glob(pattern, recursive=True):
        if not os.path.isdir(folder_path):
            continue
        
        folder_name = os.path.basename(folder_path)
        tokens = folder_name.split('_')
        params = {}
        for token in tokens:
            if '=' in token:
                key, value = token.split('=', 1)
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

        required_keys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'sr', 'mass']
        if not all(key in params and params[key] is not None for key in required_keys):
            continue

        if (math.isclose(params['x'], x, rel_tol=tol) and
            math.isclose(params['y'], y, rel_tol=tol) and
            math.isclose(params['z'], z, rel_tol=tol) and
            math.isclose(params['vx'], vx, rel_tol=tol) and
            math.isclose(params['vy'], vy, rel_tol=tol) and
            math.isclose(params['vz'], vz, rel_tol=tol) and
            math.isclose(params['sr'], sr, rel_tol=tol) and
            math.isclose(params['mass'], mass, rel_tol=tol)):
            matching_folders.append(folder_path)

    if not matching_folders:
        return None, None

    folder = matching_folders[0]
    orient_file = os.path.join(folder, 'cluster.orient.run1')
    out_files = glob.glob(os.path.join(folder, 'OUT.run1.*'))
    out_files.sort()
    print(len(out_files))
    return folder, out_files, orient_file



if __name__ == '__main__':

    file = get_simulation_files(12.0, 0, 8.0, 0, 56.0, -20.0, 0.01, 1e+05)
    print(file)