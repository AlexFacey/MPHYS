
import numpy as np
import sys
sys.path.append(r'/home/afacey/MPhysProj/MySims/exptool')
from exptool.io import particle
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from backend_functions import read_orient, get_phi1_phi2, get_phi1_phi2_straight
import os
from tidal_radius import normalise_time_to_gcr, calc_tidal_radius
from when_particle_leaves_radius import particle_leaves_sphere
from scipy.ndimage import gaussian_filter1d
import glob
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plotting_func import set_size

def find_density_along_stream(phi1, phi2, n_bins=100):

    bin_edges = np.linspace(min(phi1), max(phi1), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count the number of points in each bin
    counts, _ = np.histogram(phi1, bins=bin_edges)

    # Compute density (counts per unit length)
    bin_width = bin_edges[1] - bin_edges[0]
    densities = counts / bin_width

    # Need to figure out what i need to keep here
    bin_edges_phi2 = np.linspace(min(phi2), max(phi2), n_bins + 1)
    bin_centers_phi2 = 0.5 * (bin_edges_phi2[:-1] + bin_edges_phi2[1:])
    counts_phi2, _ = np.histogram(phi2, bins=bin_edges_phi2)
    densities_phi2 = counts_phi2 / (bin_edges_phi2[1] - bin_edges_phi2[0])

    return bin_centers, densities, phi1, phi2

def find_length_of_tails(phi1, phi1_max):
    # Split into left and right tails

    # phi1 = np.unwrap(phi1)

    mask_left = phi1 < -phi1_max
    mask_right = phi1 > phi1_max

    phi1_left = phi1[mask_left]
    phi1_right = phi1[mask_right]

    # Check for empty arrays and handle accordingly
    if phi1_left.size > 0:
        left_max = np.max(phi1_left)
        left_min = np.percentile(phi1_left, 5)
    else:
        left_max = np.nan  # or a default value or raise an error
        left_min = np.nan

    if phi1_right.size > 0:
        right_max = np.min(phi1_right)
        right_min = np.percentile(phi1_right, 95)
    else:
        right_max = np.nan
        right_min = np.nan

    return (left_min, left_max, right_min, right_max)

def find_length_of_stream_curves(folder, positions_origin, velocities_origin, sr, mass):

    x_ori, y_ori, z_ori = positions_origin
    vx_ori, vy_ori, vz_ori = velocities_origin

    left_lengths = []
    right_lengths = []
    time_steps = []
    t_chars = []

    # For tracking time steps in this folder
    timestep = 0

    orient_file = os.path.join(folder, 'cluster.orient.run1')
    
    orient_data = read_orient(orient_file)
    out_files = glob.glob(os.path.join(folder, 'OUT.run1.*'))
    out_files.sort()

    for i, out_file in enumerate(out_files[:200]):
        timestep += 1
        # Read the current snapshot
        data = particle.Input(filename=out_file, comp='cluster', legacy=True)

        # Determine the cluster center at this timestep using orient_data.
        # Note: Adjust the multiplier (250) if needed based on sr.

        if sr == 0.01:
            mult_fac = 250
        else:
            mult_fac = 250

        xcen = orient_data['xcen'][i * mult_fac]
        ycen = orient_data['ycen'][i * mult_fac]
        zcen = orient_data['zcen'][i * mult_fac]

        # Use the provided mass parameter (comment out hard-coded value if not needed)
        # mass = 1e7

        x, y, z = data.xpos, data.ypos, data.zpos
        phi1, phi2 = get_phi1_phi2(data)
        tidal_radius = calc_tidal_radius(x, y, z, mass)

        # Calculate the maximum phi1 value based on the tidal radius and cluster center distance
        r_cen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
        phi1_max = tidal_radius / r_cen

        # Get the tail lengths (left and right) using your provided function.
        left_min, left_max, right_min, right_max = find_length_of_tails(phi1, phi1_max)

        left_lengths.append(np.abs(left_max - left_min))
        right_lengths.append(np.abs(right_max - right_min))

        # Normalize time using the origin positions and velocities.
        t_char = normalise_time_to_gcr([x_ori, y_ori, z_ori], [vx_ori, vy_ori, vz_ori])
        t_chars.append(t_char)
        time_steps.append(timestep / t_char)

    return left_lengths, right_lengths, time_steps, t_chars

def fit_bending_to_stream(phi1, phi2, length):

    left_min, left_max, right_min, right_max = length

    mask_left = (phi1 >= left_min) & (phi1 <= left_max)
    phi1_left = phi1[mask_left]
    phi2_left = phi2[mask_left]

    mask_right = (phi1 >= right_max) & (phi1 <= right_min)
    phi1_right = phi1[mask_right]
    phi2_right = phi2[mask_right]

    # Define the cosine function model
    def cosine_func(x, A, B, C):
        return A * np.cos(B * x + C)

    A_left0 = (np.max(phi2_left) - np.min(phi2_left)) / 2 # amplitude
    omega_left0 = 2 * np.pi / (np.max(phi1_left) - np.min(phi1_left)) # frequency
    phase_left0 = 0 # phase
    p0_left = [A_left0, omega_left0, phase_left0]

    popt_left, _ = curve_fit(cosine_func, phi1_left, phi2_left, p0=p0_left, maxfev=10000)
    A_left, omega_left, phase_left = popt_left

    A_right0 = (np.max(phi2_right) - np.min(phi2_right)) / 2 # amplitude
    omega_right0 = 2 * np.pi / (np.max(phi1_right) - np.min(phi1_right)) # frequency
    phase_right0 = 0 # phase
    p0_right = [A_right0, omega_right0, phase_right0]

    popt_right, _ = curve_fit(cosine_func, phi1_right, phi2_right, p0=p0_right, maxfev=10000)
    A_right, omega_right, phase_right = popt_right

    # Generate fitted curves over the defined extents
    # phi1_left_range = np.linspace(left_min, left_max, 100)
    # phi2_left_fit = cosine_func(phi1_left_range, A_left, omega_left, phase_left)
    # phi2_left_fit -= 0.8 * phi2_left_fit[-1]
    # phi1_right_range = np.linspace(right_min, right_max, 100)
    # phi2_right_fit = cosine_func(phi1_right_range, A_right, omega_right, phase_right)
    # phi2_right_fit -= 0.8 * phi2_right_fit[-1]

    return popt_left, popt_right

def find_bending_along_stream_curves(folders, sr, mass):

    total_time_steps = []

    tot_amp_left = []
    tot_amp_right = []

    tot_freq_left = []
    tot_freq_right = []

    tot_phase_left = []
    tot_phase_right = []


    for folder in folders:


        timesteps = []
        amp_left = []
        amp_right = []
        freq_left = []
        freq_right = []
        phase_left = []
        phase_right = []

        timestep = 0

        orient_file = os.path.join(folder, 'cluster.orient.run1')
        orient_data = read_orient(orient_file)
        out_files = glob.glob(os.path.join(folder, 'OUT.run1.*'))
        out_files.sort()

        for i, out_file in enumerate(out_files[:-1]):

            timestep += 1
            data = particle.Input(filename=out_files[i], comp='cluster', legacy=True)
            # this needs to be changed depending on wether it is 0.01 or 0.1
            if sr == 0.01:
                mult_fac = 2500
            else:
                mult_fac = 250
            xcen = orient_data['xcen'][i*mult_fac]
            ycen = orient_data['ycen'][i*mult_fac]
            zcen = orient_data['zcen'][i*mult_fac]
            # change this to be generalised (mass)

            x, y, z = data.xpos, data.ypos, data.zpos
            phi1, phi2 = get_phi1_phi2(data)
            tidal_radius = calc_tidal_radius(x, y, z, mass)

            r_cen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
            phi1_max = tidal_radius / r_cen
            
            length = find_length_of_tails(phi1, phi1_max)

            popt_left, popt_right = fit_bending_to_stream(phi1, phi2, length)[0], fit_bending_to_stream(phi1, phi2, length)[1]
            amp_l, freq_l, phase_l = popt_left
            amp_r, freq_r, phase_r = popt_right

            amp_left.append(amp_l)
            amp_right.append(amp_r)
            freq_left.append(freq_l)
            freq_right.append(freq_r)
            phase_left.append(phase_l)
            phase_right.append(phase_r)
            timesteps.append(timestep)

        tot_amp_left.append(amp_left)
        tot_amp_right.append(amp_right)
        tot_freq_left.append(freq_left)
        tot_freq_right.append(freq_right)
        tot_phase_left.append(phase_left)
        tot_phase_right.append(phase_right)
        total_time_steps.append(timesteps)

    return tot_amp_left, tot_amp_right, tot_freq_left, tot_freq_right, tot_phase_left, tot_phase_right, total_time_steps

def find_gaps_in_stream(bin_centers, densities, left_min, right_max):

    # Find apocenters ie when the stream is at its most extended


    inverted_densities = -densities  # Flip densities to find minima
    # Smooth the inverted densities
    inverted_densities = gaussian_filter1d(inverted_densities, sigma=2)
    minima_indices, _ = find_peaks(inverted_densities, prominence=10)

    dips_phi1 = bin_centers[minima_indices]
    dips_densities = densities[minima_indices]

    mask = (dips_phi1 >= left_min) & (dips_phi1 <= right_max)
    dips_phi1 = dips_phi1[mask]
    dips_densities = dips_densities[mask]
    
    # Find corresponding phi2 values by matching closest phi1 values potentially
    dips_phi2 = np.zeros(len(dips_phi1))
    # eventually i will just need to return dips_ph1 and dips_phi2, the others are for plotting.
    return dips_phi1, dips_phi2, dips_densities, -inverted_densities

def find_gaps_stream_with_length(folder, positions_origin, velcoities_origin, sr, mass):
    
    left_lengths, right_lengths, _, _ = find_length_of_stream_curves(folder, positions_origin, velcoities_origin, sr, mass)
    # find peaks of left_lengths and right_lengths
    left_peaks, _ = find_peaks(left_lengths, prominence=0.1)
    right_peaks, _ = find_peaks(right_lengths, prominence=0.1)
    plt.plot(left_lengths)
    plt.savefig('left_lengths.png')
    out_file = glob.glob(os.path.join(folder, 'OUT.run1.*'))
    out_file.sort()

    fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)

    for left_peak in left_peaks:
        data = particle.Input(filename=out_file[left_peak], comp='cluster', legacy=True)

        phi1, phi2 = get_phi1_phi2(data)
        mask = (phi1 < -0.0) & (phi1 > -left_lengths[left_peak])
        phi1 = phi1[mask]
        phi2 = phi2[mask]
        
        # Make sure bin number is odd so that center is center
        bin_centers, densities, phi1, phi2 = find_density_along_stream(phi1, phi2, n_bins=21)

        inverted_densities = -densities  # Flip densities to find minima

        # Smooth the inverted densities
        smooth_inverted_densities = gaussian_filter1d(inverted_densities, sigma=1)
        minima_indices, _ = find_peaks(smooth_inverted_densities, prominence=0.1)
        
        dips_phi1 = bin_centers[minima_indices]
        dips_densities = densities[minima_indices]

        left_min = -np.pi
        # irellavent here for now
        right_max = np.pi

        mask = (dips_phi1 >= left_min) & (dips_phi1 <= right_max)
        dips_phi1 = dips_phi1[mask]
        dips_densities = dips_densities[mask]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=True)

        ax1.plot(
            phi1,
            phi2,
            marker="o",
            markeredgewidth=0,
            markersize=2.0,
            ls="none",
            alpha=0.25,
            c='black',
        )

        ax1.scatter(
            dips_phi1,
            np.zeros(len(dips_phi1)),
            marker="x",
            s = 70,
            alpha=0.75,
            c='red',
            label='Located Gaps'
        )
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlim(-np.pi, 0)
        # ax1.set_xlabel(r"$\phi_1$ [rad]")
        ax1.set_ylabel(r"$\phi_2$ / rad")
        ax1.legend(loc='upper left', frameon=False)
        ax2.plot(bin_centers, -smooth_inverted_densities, linestyle='-', c='black', label='Smoothed Density Profile')
        ax2.plot(bin_centers, densities, linestyle='--', c='blue', label='Density Profile')
        ax2.scatter(dips_phi1, -smooth_inverted_densities[minima_indices], color='red', marker='x', s=70, label='Located Gaps')
        ax2.set_xlabel(r'$\phi_1$ / rad')
        ax2.set_ylabel('Density')
        ax2.set_yscale('log')
        ax2.set_xlim(-np.pi, 0)
        ax2.set_ylim(10, 100000)
        ax2.legend(loc='upper left', frameon=False)

        plt.savefig(f'phi1_phi2_{left_peak}_left.png', dpi=600)
        plt.cla()

    # fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)

    # for left_peak in left_peaks:
        
    #     print(left_peak)
    #     data = particle.Input(filename=out_file[left_peak], comp='cluster', legacy=True)
    #     phi1, phi2 = get_phi1_phi2(data)

    #     ax.plot(
    #         phi1,
    #         phi2,
    #         marker="o",
    #         markeredgewidth=0,
    #         markersize=2.0,
    #         ls="none",
    #         alpha=0.25,
    #         c='black',
    #     )

    #     ax.set_ylim(-0.1, 0.1)
    #     ax.set_xlim(-np.pi/2, np.pi/2)
    #     ax.set_xlabel(r"$\phi_1$ [rad]")
    #     ax.set_ylabel(r"$\phi_2$ [rad]")
    #     plt.savefig(f'phi1_phi2_{left_peak}_left.png')
    #     plt.cla()

def find_mass_loss_curve(exit_data):

    last_occurrence = {}
    time_steps = []
    initial_mass_cluster = exit_data[0]['initial_mass_cluster']

    for exit in exit_data:
        particle_id = exit['particle_id']
        time_step = exit['time_step']
        last_occurrence[particle_id] = time_step 
        if time_step not in time_steps:
            time_steps.append(time_step)

    total_mass_loss = 0

    total_mass_loss_list = []

    for time in time_steps:
        mass_loss = 0
        for particle_id, time_step in last_occurrence.items():
            if time_step == time:
                mass_loss += 1
        total_mass_loss += mass_loss
        total_mass_loss_list.append(total_mass_loss)

    # return (initial_mass_cluster - np.array(total_mass_loss_list)) / initial_mass_cluster, time_steps
    return total_mass_loss_list, time_steps

def find_mass_loss_curves(folder, positions, velocities, sr, mass):

    x,y,z = positions
    vx, vy, vz = velocities
    file_name = folder
    orient_file = os.path.join(file_name, 'cluster.orient.run1')
    orient_data = read_orient(orient_file)
    exit_info = particle_leaves_sphere(file_name, orient_data, sr, mass)
    mass_loss_curve, time_steps = find_mass_loss_curve(exit_info)
    t_char = normalise_time_to_gcr([x, y, z], [vx, vy, vz])
    time_steps = np.array(time_steps) / t_char
    
    return mass_loss_curve, time_steps, t_char

if __name__ == '__main__':

    # folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/test_for_polar_orbit_2'
    folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=12'
    # These initial positions are irellavent for what I am doing, i will change this all so this doesnt need to be done later
    find_gaps_stream_with_length(folder, [1, 1, 24], [1, 1, 1], 0.1, 1e7)

