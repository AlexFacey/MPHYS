import numpy as np
from scipy.stats import truncnorm
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
import math
import glob
import os
import sys
sys.path.append(r'/home/afacey/MPhysProj/MySims/exptool')
from exptool.io import particle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed

def convert_from_observ_to_cartesian(ra, dec, distance, pm_ra, pm_dec, radial_velocity):
    """
    Parameters:
      ra              : float
                        Right Ascension in degrees.
      dec             : float
                        Declination in degrees.
      distance        : float
                        Distance in kiloparsecs (kpc).
      pm_ra           : float
                        Proper motion in RA (mas/yr).
      pm_dec          : float
                        Proper motion in Dec (mas/yr).
      radial_velocity : float
                        Radial velocity in km/s.
                        
    Returns:
      vx, vy, vz, phi : tuple of floats
                        vx, vy, vz are the Galactocentric velocity components in km/s,
                        and phi is the azimuthal angle in radians in the Galactocentric frame.
    """
    # Create a SkyCoord object in ICRS coordinates
    c = SkyCoord(ra=ra*u.deg,
                 dec=dec*u.deg,
                 distance=distance*u.kpc,
                 pm_ra_cosdec=pm_ra*u.mas/u.yr,
                 pm_dec=pm_dec*u.mas/u.yr,
                 radial_velocity=radial_velocity*u.km/u.s)
    
    # Transform to the Galactocentric frame
    galcen = c.transform_to(Galactocentric())

    x = galcen.x.to(u.kpc).value
    y = galcen.y.to(u.kpc).value
    z = galcen.z.to(u.kpc).value

    vx = galcen.v_x.to(u.km/u.s).value
    vy = galcen.v_y.to(u.km/u.s).value
    vz = galcen.v_z.to(u.km/u.s).value

    azi_ang = np.arctan2(y, x)

    # These are the coordiantes in galactocentric frame
    return [x,y,z], [vx, vy, vz], azi_ang

def rotate_to_azimuthal_angle(position, velocity, azimuthal_angle, rotation_type='neg'):

    x, y, z = position
    vx, vy, vz = velocity

    # Determine rotation direction
    if rotation_type == 'pos':
        angle = azimuthal_angle  # Rotate positively
    else:
        angle = -azimuthal_angle  # Default: Rotate negatively

    # Apply rotation matrix
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    z_rot = z

    vx_rot = vx * np.cos(angle) - vy * np.sin(angle)
    vy_rot = vx * np.sin(angle) + vy * np.cos(angle)
    vz_rot = vz 

    return [x_rot, y_rot, z_rot], [vx_rot, vy_rot, vz_rot]

def abs_and_sign_coords(position, velocity, azimuthal_angle):

    original_signs_pos = np.sign(position)
    original_signs_vel = np.sign(velocity)

    target_position = np.abs(position)
    target_velocity = np.abs(velocity)

    if original_signs_pos[0] < 0:
        target_velocity[0] = -target_velocity[0]

    return original_signs_pos, original_signs_vel, target_position, target_velocity

def reconstruct_real_sim_string(sim, sign_pos, sign_vel):
    params = sim.split('_')
    x = float(params[0].split('=')[1])
    y = float(params[1].split('=')[1])
    z = float(params[2].split('=')[1])
    vx = float(params[3].split('=')[1])
    vy = float(params[4].split('=')[1])
    vz = float(params[5].split('=')[1])

    x = x * sign_pos[0]
    y = y * sign_pos[1]
    z = z * sign_pos[2]
    
    vx = vx * sign_vel[0]
    vy = vy * sign_vel[1]
    vz = vz * sign_vel[2]

    real_sim_string = f"x={x:.1f}_y={y:.1f}_z={z:.1f}_vx={vx:.1f}_vy={vy:.1f}_vz={vz:.1f}"

    return real_sim_string

def process_sim(sim, sim_data, target_pos, target_vel, sign_pos, sign_vel, obs_azi_angle):
    """
    Process a single simulation group and return the candidate best match for that sim.
    """
    # Get positions and velocities (n_timesteps, 3)

    pos_array = np.array([sim_data['x'], sim_data['y'], sim_data['z']]).T
    vel_array = np.array([sim_data['vx'], sim_data['vy'], sim_data['vz']]).T

    azi_ang_timesteps = np.arctan2(pos_array[:, 1], pos_array[:, 0])
    cos_a = np.cos(-azi_ang_timesteps)
    sin_a = np.sin(-azi_ang_timesteps)

    x_rot = pos_array[:, 0] * cos_a - pos_array[:, 1] * sin_a
    y_rot = pos_array[:, 0] * sin_a + pos_array[:, 1] * cos_a
    z_rot = pos_array[:, 2]  # unchanged
    rotated_pos_array = np.stack((x_rot, y_rot, z_rot), axis=1)

    vx_rot = vel_array[:, 0] * cos_a - vel_array[:, 1] * sin_a
    vy_rot = vel_array[:, 0] * sin_a + vel_array[:, 1] * cos_a
    vz_rot = vel_array[:, 2]  # unchanged
    rotated_vel_array = np.stack((vx_rot, vy_rot, vz_rot), axis=1)

    diff_all = np.hstack((rotated_pos_array - target_pos, rotated_vel_array - target_vel))

    cov = np.cov(diff_all, rowvar=False)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    cost_total = np.sum(diff_all * (diff_all.dot(cov_inv)), axis=1)

    min_orbit_id = int(np.argmin(cost_total))
    min_cost = cost_total[min_orbit_id]

    real_sim = reconstruct_real_sim_string(sim, sign_pos, sign_vel)

    signed_x = np.array(sim_data['x']) * sign_pos[0]
    signed_y = np.array(sim_data['y']) * sign_pos[1]
    signed_z = np.array(sim_data['z']) * sign_pos[2]
    signed_vx = np.array(sim_data['vx']) * sign_vel[0]
    signed_vy = np.array(sim_data['vy']) * sign_vel[1]
    signed_vz = np.array(sim_data['vz']) * sign_vel[2]


    signed_positions = np.stack((signed_x, signed_y, signed_z), axis=1)
    signed_velocities = np.stack((signed_vx, signed_vy, signed_vz), axis=1)

    cos_obs = np.cos(obs_azi_angle)
    sin_obs = np.sin(obs_azi_angle)
    

    x_final = signed_positions[:, 0] * cos_obs - signed_positions[:, 1] * sin_obs
    y_final = signed_positions[:, 0] * sin_obs + signed_positions[:, 1] * cos_obs
    z_final = signed_positions[:, 2] 

    vx_final = signed_velocities[:, 0] * cos_obs - signed_velocities[:, 1] * sin_obs
    vy_final = signed_velocities[:, 0] * sin_obs + signed_velocities[:, 1] * cos_obs
    vz_final = signed_velocities[:, 2]

    # Save the final orbital info with the additional rotation

    # params = sim.split('_')
    # sr = float(params[6].split('=')[1])
    # mass = float(params[7].split('=')[1])


    # # THIS IS SETUP FOR MY FINAL DATA SET SO WONT WORK WITH THE OLD ONE
    # # Find the corresponding mass loss curves and length curves
    # if math.isclose(sr, 0.01, abs_tol=1e-6) and math.isclose(mass, 1e5, abs_tol=1e-6):
    #     ml_curve = sim_data['ml_sr=0.01_m=1e5']
    #     l_length_curve = sim_data['l_sr=0.01_m=1e5']
    #     r_length_curve = sim_data['r_sr=0.01_m=1e5']
    # elif math.isclose(sr, 0.01, abs_tol=1e-6) and math.isclose(mass, 1e6, abs_tol=1e-6):
    #     ml_curve = sim_data['ml_sr=0.01_m=1e6']
    #     l_length_curve = sim_data['l_sr=0.01_m=1e6']
    #     r_length_curve = sim_data['r_sr=0.01_m=1e6']
    # elif math.isclose(sr, 0.1, abs_tol=1e-6) and math.isclose(mass, 1e5, abs_tol=1e-6):
    #     ml_curve = sim_data['ml_sr=0.1_m=1e5']
    #     l_length_curve = sim_data['l_sr=0.1_m=1e5']
    #     r_length_curve = sim_data['r_sr=0.1_m=1e5']
    # elif math.isclose(sr, 0.1, abs_tol=1e-6) and math.isclose(mass, 1e6, abs_tol=1e-6):
    #     ml_curve = sim_data['ml_sr=0.1_m=1e6']
    #     l_length_curve = sim_data['l_sr=0.1_m=1e6']
    #     r_length_curve = sim_data['r_sr=0.1_m=1e6']
    # elif math.isclose(sr, 0.5, abs_tol=1e-6) and math.isclose(mass, 1e5, abs_tol=1e-6):
    #     ml_curve = sim_data['ml_sr=0.5_m=1e5']
    #     l_length_curve = sim_data['l_sr=0.5_m=1e5']
    #     r_length_curve = sim_data['r_sr=0.5_m=1e5']
    # elif math.isclose(sr, 0.5, abs_tol=1e-6) and math.isclose(mass, 1e6, abs_tol=1e-6):
    #     ml_curve = sim_data['ml_sr=0.5_m=1e6']
    #     l_length_curve = sim_data['l_sr=0.5_m=1e6']
    #     r_length_curve = sim_data['r_sr=0.5_m=1e6']

    # For now, just so we can do some tests:
    ml_curve = np.linspace(0, 6001, 200).astype(int)
    l_length_curve = np.linspace(0, np.pi, 200)
    r_length_curve = np.linspace(0, np.pi, 200)

    orbital_info = {
        'x': x_final.tolist(),
        'y': y_final.tolist(),
        'z': z_final.tolist(),
        'vx': vx_final.tolist(),
        'vy': vy_final.tolist(),
        'vz': vz_final.tolist(),
        'mass_loss_curve': ml_curve,
        'left_lengths': l_length_curve,
        'right_lengths': r_length_curve
    }
    # Candidate dictionary for this simulation
    candidate = {
        'init_cond_sim': sim,
        'init_cond_real': real_sim,
        't_idx': min_orbit_id,
        'cost': float(min_cost),
        'state': {
            'pos': rotated_pos_array[min_orbit_id].tolist(),
            'vel': rotated_vel_array[min_orbit_id].tolist()
        },
        'orbital_info': orbital_info
    }
    return candidate

def predict_orbit_of_stream(ra, dec, distance, pm_ra, pm_dec, radial_velocity, n_jobs=1):
    # Convert the observed parameters to Galactocentric Cartesian coordinates
    positions, velocities, azi_ang = convert_from_observ_to_cartesian(
        ra, dec, distance, pm_ra, pm_dec, radial_velocity
    )

    print("Target conversion:", positions, velocities, azi_ang)
    # Load the simulation data
    # total_data_file = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/total_data_orbit_test2.csv'
    total_data_file = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/only_inits_orbits.csv'
    start_time = time.time()
    df = pd.read_csv(total_data_file)
    end_time = time.time()
    print(f"Time taken to read the file: {end_time - start_time} seconds")


    azi_ang = np.arctan2(positions[1], positions[0])

    sign_pos, sign_vel, abs_pos, abs_vel = abs_and_sign_coords(positions, velocities, azi_ang)
    target_pos, target_vel = rotate_to_azimuthal_angle(abs_pos, abs_vel, azi_ang)
    print("Target (rotated) pos and vel:", target_pos, target_vel)

    # Pre-group the simulations once
    sim_groups = df.groupby('init_cond')
    print(f"Processing {len(sim_groups)} simulations...")

    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(process_sim)(sim, sim_data, target_pos, target_vel, sign_pos, sign_vel, obs_azi_angle=azi_ang)
            for sim, sim_data in sim_groups
        )
    else:
        results = []
        for sim, sim_data in sim_groups:
            results.append(process_sim(sim, sim_data, target_pos, target_vel, sign_pos, sign_vel, obs_azi_angle=azi_ang))

    # Collect candidate matches into a dictionary and find the best match
    # Sort candidates by cost and only keep the 9 with the lowest cost
    results_sorted = sorted(results, key=lambda candidate: candidate['cost'])
    lowest9 = results_sorted[:9]

    best_orbits = {}
    for candidate in lowest9:
        best_orbits[candidate['init_cond_sim']] = candidate

    return best_orbits



if __name__ == '__main__':
    pal5_data = {
        'ra': 229.018,
        'dec': -0.1114,
        'distance': 22.1,
        'pm_ra': -2.296,
        'pm_dec': -2.257,
        'radial_velocity': -58.7
    }


    best_orbits = predict_orbit_of_stream(**pal5_data, n_jobs=8)
    print(best_orbits.keys())

    best_candidates = list(best_orbits.values())
    pos_pal5, vel_pal5, azi_ang = convert_from_observ_to_cartesian(**pal5_data)

    # 3D PLOTS
    fig3d = plt.figure(figsize=(15, 12))
    for i, candidate in enumerate(best_candidates):
        ax = fig3d.add_subplot(3, 3, i+1, projection='3d')
        orbital_info = candidate['orbital_info']
        x = orbital_info['x']
        y = orbital_info['y']
        z = orbital_info['z']
        t_idx = candidate['t_idx']
        
        start_idx = max(0, t_idx - 15)
        end_idx = min(len(x), t_idx + 15)
        
        x_before = x[start_idx:t_idx]
        y_before = y[start_idx:t_idx]
        z_before = z[start_idx:t_idx]
        
        x_after = x[t_idx:end_idx]
        y_after = y[t_idx:end_idx]
        z_after = z[t_idx:end_idx]
        
        if len(x_before) > 0:
            ax.plot(x_before, y_before, z_before, lw=1, color='blue', alpha=0.5)
        if len(x_after) > 0:
            ax.plot(x_after, y_after, z_after, lw=1, color='red', alpha=0.5)
        
        ax.scatter(pos_pal5[0], pos_pal5[1], pos_pal5[2], color='black', s=10, label='Pal 5')
        ax.scatter(x[t_idx], y[t_idx], z[t_idx], color='red', s=10, label='Matched Position')
        ax.scatter(-8, 0, 0, color='yellow', s=20, label='Sun')
        
        ax.plot([-20, 20], [0, 0], [0, 0], 'k--')
        ax.plot([0, 0], [-20, 20], [0, 0], 'k--')
        
        ax.view_init(elev=30, azim=-135)
        ax.set_xlabel("X (kpc)")
        ax.set_ylabel("Y (kpc)")
        ax.set_zlabel("Z (kpc)")
        ax.set_title(f'Predicted Orbit {i+1}: Timestep {t_idx}\nAge: {(t_idx/200)*3.615:.2f} Gyr')
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('best_orbits_3d.png')

    # 2D PLOT
    fig2d = plt.figure(figsize=(15, 12))
    for i, candidate in enumerate(best_candidates):
        ax = fig2d.add_subplot(3, 3, i+1)
        orbital_info = candidate['orbital_info']
        x = orbital_info['x']
        y = orbital_info['y']
        t_idx = candidate['t_idx']
        
        start_idx = max(0, t_idx - 15)
        end_idx = min(len(x), t_idx + 15)
        
        x_before = x[start_idx:t_idx]
        y_before = y[start_idx:t_idx]
        x_after = x[t_idx:end_idx]
        y_after = y[t_idx:end_idx]
        
        if len(x_before) > 0:
            ax.plot(x_before, y_before, lw=1, color='red', alpha=0.5, label='Before')
        if len(x_after) > 0:
            ax.plot(x_after, y_after, lw=1, color='blue', alpha=0.5, label='After')
        
        ax.scatter(pos_pal5[0], pos_pal5[1], color='black', s=10, label='Pal 5')
        ax.scatter(x[t_idx], y[t_idx], color='red', s=10, label='Matched Position')
        ax.scatter(-8, 0, color='yellow', s=20, label='Sun')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_xlabel("X (kpc)")
        ax.set_ylabel("Y (kpc)")
        ax.set_title(f'Predicted Orbit {i+1}: Timestep {t_idx}\nAge: {(t_idx/200)*3.615:.2f} Gyr')
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig('best_orbits_2d.png')
