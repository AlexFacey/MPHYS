from backend_functions import read_orient, compute_phi1_phi2
from tidal_radius import calc_tidal_radius
import numpy as np
import os
import sys
sys.path.append(r'/home/afacey/MPhysProj/MySims/exptool')
from exptool.io import particle


def particle_leaves_sphere(folder_path, orient_data, sr, mass):
    # Get all relevant files.
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith('OUT.run1.')]
    files.sort()
    exit_info = []
    
    if sr == 0.01:
        mult_fac = 2500
    else:
        mult_fac = 250

    for i in range(len(files) - 2):
        file_before = files[i]
        file_after = files[i+1]

        O_before = particle.Input(filename=file_before, comp='cluster')
        O_after  = particle.Input(filename=file_after, comp='cluster')

        # Extract positions and velocities as NumPy arrays.
        x_before, y_before, z_before = O_before.data['x'], O_before.data['y'], O_before.data['z']
        x_after, y_after, z_after    = O_after.data['x'], O_after.data['y'], O_after.data['z']

        vx_after, vy_after, vz_after = O_after.data['vx'], O_after.data['vy'], O_after.data['vz']
        particle_ids = O_before.data['id']  # Assume same ordering for both

        # Get cluster centers from orient_data.
        xcen_before = orient_data['xcen'][i * mult_fac]
        ycen_before = orient_data['ycen'][i * mult_fac]
        zcen_before = orient_data['zcen'][i * mult_fac]
        xcen_after  = orient_data['xcen'][(i+1) * mult_fac]
        ycen_after  = orient_data['ycen'][(i+1) * mult_fac]
        zcen_after  = orient_data['zcen'][(i+1) * mult_fac]

        # Center the coordinates.
        x_before_centered = x_before - xcen_before
        y_before_centered = y_before - ycen_before
        z_before_centered = z_before - zcen_before

        x_after_centered = x_after - xcen_after
        y_after_centered = y_after - ycen_after
        z_after_centered = z_after - zcen_after

        # Compute distances from cluster center.
        distance_before = np.sqrt(x_before_centered**2 + y_before_centered**2 + z_before_centered**2)
        distance_after  = np.sqrt(x_after_centered**2 + y_after_centered**2 + z_after_centered**2)

        # In this example, we use sr as the tidal radius.
        tidal_radius_before = sr
        tidal_radius_after  = sr

        if i == 0:
            initial_tid_radius = tidal_radius_before
            initial_mass_cluster = np.sum(distance_before < initial_tid_radius)
            print(f'{folder_path}')
            print("Initial Tidal Radius:", initial_tid_radius)
            print("Initial Mass:", initial_mass_cluster)


        inside_before = distance_before < tidal_radius_before
        inside_after  = distance_after < tidal_radius_after
        # Identify particles that were inside and then left.
        exit_mask = inside_before & (~inside_after)
        exit_indices = np.where(exit_mask)[0]

        if exit_indices.size > 0:
            # Calculate the fraction for exit interpolation.
            fraction = (tidal_radius_before - distance_before[exit_indices]) / (distance_after[exit_indices] - distance_before[exit_indices])
            
            x_exit = x_before_centered[exit_indices] + fraction * (x_after_centered[exit_indices] - x_before_centered[exit_indices])
            y_exit = y_before_centered[exit_indices] + fraction * (y_after_centered[exit_indices] - y_before_centered[exit_indices])
            z_exit = z_before_centered[exit_indices] + fraction * (z_after_centered[exit_indices] - z_before_centered[exit_indices])
            
            x_exit_nc = x_before[exit_indices] + fraction * (x_after[exit_indices] - x_before[exit_indices])
            y_exit_nc = y_before[exit_indices] + fraction * (y_after[exit_indices] - y_before[exit_indices])
            z_exit_nc = z_before[exit_indices] + fraction * (z_after[exit_indices] - z_before[exit_indices])
            
            for idx, j in enumerate(exit_indices):
                exit_info.append({
                    "time_step": i + 1,
                    "time": O_after.time,
                    "particle_id": particle_ids[j],
                    "x": x_exit[idx], "y": y_exit[idx], "z": z_exit[idx],
                    "x_non_centered": x_exit_nc[idx], "y_non_centered": y_exit_nc[idx], "z_non_centered": z_exit_nc[idx],
                    "vx": vx_after[j], "vy": vy_after[j], "vz": vz_after[j],
                    "clus_cenx": xcen_after, "clus_ceny": ycen_after, "clus_cenz": zcen_after,
                    "initial_mass_cluster": initial_mass_cluster,
                    "mass_of_particles": mass / 10000
                })
    return exit_info

def convert_to_polar_coords(exit_info):
    for exit in exit_info:
        # Centered coordinates
        x, y, z = exit['x'], exit['y'], exit['z']
        exit['r'] = np.sqrt(x**2 + y**2 + z**2)
        exit['theta'] = np.degrees(np.arctan2(y, x))
        exit['phi'] = np.degrees(np.arccos(z / exit['r']))

        # Non-centered coordinates
        x_nc, y_nc, z_nc = exit['x_non_centered'], exit['y_non_centered'], exit['z_non_centered']
        exit['r_nc'] = np.sqrt(x_nc**2 + y_nc**2 + z_nc**2)
        exit['theta_nc'] = np.degrees(np.arctan2(y_nc, x_nc))
        exit['phi_nc'] = np.degrees(np.arccos(z_nc / exit['r_nc']))


    return exit_info

def rotate_with_gal(exit_info):

    for exit in exit_info:

        x_stars, y_stars, z_stars = exit['x_non_centered'], exit['y_non_centered'], exit['z_non_centered']
        x_clus_cen, y_clus_cen, z_clus_cen = exit['clus_cenx'], exit['clus_ceny'], exit['clus_cenz']
        vx, vy, vz = exit['vx'], exit['vy'], exit['vz']

        phi_c = np.arctan2(exit['clus_ceny'], exit['clus_cenx'])

        x_rot = x_stars * np.cos(phi_c) + y_stars * np.sin(phi_c)
        y_rot = -x_stars * np.sin(phi_c) + y_stars * np.cos(phi_c)
        
        x_clus_cen_rot = x_clus_cen * np.cos(phi_c) + y_clus_cen * np.sin(phi_c)
        y_clus_cen_rot = -x_clus_cen * np.sin(phi_c) + y_clus_cen * np.cos(phi_c)

        vx_rot = vx * np.cos(phi_c) + vy * np.sin(phi_c)
        vy_rot = -vx * np.sin(phi_c) + vy * np.cos(phi_c)

        exit['x_rot'] = x_rot
        exit['y_rot'] = y_rot
        exit['z_rot'] = z_stars

        exit['clus_cenx_rot'] = x_clus_cen_rot
        exit['clus_ceny_rot'] = y_clus_cen_rot
        exit['clus_cenz_rot'] = z_clus_cen

        exit['vx_rot'] = vx_rot
        exit['vy_rot'] = vy_rot
        exit['vz_rot'] = vz

    return exit_info

if __name__ == '__main__':

    folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/NewBigCollectionOfSims/batch_12/x=12.0_y=0.0_z=0.0_vx=0.0_vy=164.0_vz=-60.0_sr=0.5_mass=1e+06'
    orient_file = os.path.join(folder, 'cluster.orient.run1')

    orient_data = read_orient(orient_file)

    sr = 0.5
    mass = 1e+06

    exit_info = particle_leaves_sphere(folder, orient_data, sr, mass)