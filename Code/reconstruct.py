import numpy as np
import os
import sys
from match_observation_with_simulation import predict_orbit_of_stream
from tidal_radius import calc_tidal_radius
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
from plotting_func import set_size

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def truncated_gaussian(mean, std, lower, upper, size):

    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def reconstruct_stream(prediction_number, mass_loss_curve = None, left_lengths_curve = None, right_lengths_curve= None, best_orbits=None):
    #for now whule i dont have full data set
    best_inits = list(best_orbits.keys())[0]
    data = best_orbits[best_inits]

    print(data)
    print(data['t_idx'])
    current_tdx = data['t_idx']
    orbit_info = data['orbital_info']

    print(current_tdx)
    x = orbit_info['x']
    y = orbit_info['y']
    z = orbit_info['z']

    mass_loss = orbit_info['mass_loss_curve']
    left_lengths = orbit_info['left_lengths']
    right_lengths = orbit_info['right_lengths']

    mass_loss = mass_loss_curve
    left_lengths = left_lengths_curve
    right_lengths = right_lengths_curve

    # SOMEHOW GET SR AND MASS
    init_mass = 10e5
    sr = 0.01

    idx_to_predict = prediction_number

    # Determine the number of gaps based on idx_to_predict.
    if idx_to_predict > 150:
        n_gaps = 3
    elif idx_to_predict > 80:
        n_gaps = 2
    elif idx_to_predict > 20:
        n_gaps = 1
    else:
        n_gaps = 0

    current_mass = init_mass - mass_loss[idx_to_predict] * (init_mass / 10000)
    xcen, ycen, zcen = x[idx_to_predict], y[idx_to_predict], z[idx_to_predict]
    tidal_radius = calc_tidal_radius(xcen, ycen, zcen, current_mass)
    r_cen = np.sqrt(xcen**2 + ycen**2 + zcen**2)

    phi1_max = 2 * tidal_radius / r_cen

    left_length = left_lengths[idx_to_predict]
    right_length = right_lengths[idx_to_predict]
    mass_lost = mass_loss[idx_to_predict] * (init_mass / 10000)

    left_tail_bounds = [-left_length, -phi1_max]
    right_tail_bounds = [phi1_max, right_length]

    # This should be changed back t pure mass loss number once i figure out proper
    n_tails = int(mass_loss[idx_to_predict] )
    print(n_tails, mass_loss[idx_to_predict])
    phi1_left_tail = np.random.uniform(left_tail_bounds[0], left_tail_bounds[1], n_tails // 2)
    phi1_right_tail = np.random.uniform(right_tail_bounds[0], right_tail_bounds[1], n_tails // 2)

    suppression_radius = 0.2  # Controls the width of the suppression region
    keep_prob = 0.75         # Base probability of not keeping a star
    gap_offset=0.1

    left_gap_centers = []
    if n_gaps >= 1:
        first_gap = left_tail_bounds[1] + gap_offset * (left_tail_bounds[0] - left_tail_bounds[1])
        left_gap_centers.append(first_gap)
    if n_gaps >= 2:
        mid_gap = (left_tail_bounds[0] + left_tail_bounds[1]) / 2
        left_gap_centers.append(mid_gap)
    if n_gaps >= 3:
        third_gap = left_tail_bounds[1] + (1 - gap_offset) * (left_tail_bounds[0] - left_tail_bounds[1])
        left_gap_centers.append(third_gap)

    right_gap_centers = []
    if n_gaps >= 1:
        first_gap = right_tail_bounds[0] + gap_offset * (right_tail_bounds[1] - right_tail_bounds[0])
        right_gap_centers.append(first_gap)
    if n_gaps >= 2:
        mid_gap = (right_tail_bounds[0] + right_tail_bounds[1]) / 2
        right_gap_centers.append(mid_gap)
    if n_gaps >= 3:
        third_gap = right_tail_bounds[0] + (1 - gap_offset) * (right_tail_bounds[1] - right_tail_bounds[0])
        right_gap_centers.append(third_gap)

    # Apply gap suppression on left tail candidates:
    mask_left = np.ones_like(phi1_left_tail, dtype=bool)
    for gap in left_gap_centers:
        random_vals = np.random.uniform(0, 1, len(phi1_left_tail))
        mask_left &= random_vals > (keep_prob * np.exp(-((phi1_left_tail - gap) / suppression_radius) ** 2))
    phi1_left_tail = phi1_left_tail[mask_left]

    # Apply gap suppression on right tail candidates:
    mask_right = np.ones_like(phi1_right_tail, dtype=bool)
    for gap in right_gap_centers:
        random_vals = np.random.uniform(0, 1, len(phi1_right_tail))
        mask_right &= random_vals > (keep_prob * np.exp(-((phi1_right_tail - gap) / suppression_radius) ** 2))
    phi1_right_tail = phi1_right_tail[mask_right]

    phi2_left_tail = np.random.normal(0, phi1_max * 0.9, len(phi1_left_tail))
    phi2_right_tail = np.random.normal(0, phi1_max * 0.9, len(phi1_right_tail))

    n_in_tr = 10000 - n_tails
    phi1_cluster = truncated_gaussian(0, sr, -phi1_max, phi1_max, n_in_tr)
    phi2_cluster = truncated_gaussian(0, sr, -phi1_max, phi1_max, n_in_tr)

    phi1 = np.concatenate([phi1_left_tail, phi1_cluster, phi1_right_tail])
    phi2 = np.concatenate([phi2_left_tail, phi2_cluster, phi2_right_tail])

    fig, ax = plt.subplots(figsize=set_size(subplots=(1,3)), constrained_layout=True)
    ax.plot(
        phi1,
        phi2,
        marker="o",
        markeredgewidth=0,
        markersize=1.3,
        ls="none",
        alpha=0.45,
        c='black',
    )
    # ax.set_xlim(-np.pi, np.pi)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r"$\phi_1$ [rad]")
    ax.set_ylabel(r"$\phi_2$ [rad]")
    # plt.savefig(f'/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/ReconstructedStream/reconstructed_stream_{idx_to_predict}.png', dpi=600)
    return phi1, phi2

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
    reconstruct_stream(best_orbits, 50)