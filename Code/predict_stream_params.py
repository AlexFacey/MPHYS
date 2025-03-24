import numpy as np
from scipy.interpolate import interp1d, griddata
from find_stream_params import find_mass_loss_curves, find_length_of_stream_curves, find_bending_along_stream_curves
import matplotlib.pyplot as plt
from tidal_radius import normalise_time_to_gcr
import os
import time
import pandas as pd
from matplotlib.ticker import FuncFormatter
from plotting_func import set_size

# Resample the mass loss curve from its original normalized time grid onto a common grid.
def resample_curve(norm_time, mass_loss, common_time_grid):
    return np.interp(common_time_grid, norm_time, mass_loss)

def predict_interpolated_curve_nd(new_ind_vars, sim_data, interpolation_method='linear'):

    simulation_ind_vars = np.array(sim_data['simulation_ind_vars'])
    simulation_curves = sim_data['simulation_curves']
    
    norm_time_and_curve = []
    
    # Process each simulation: convert raw time to normalized time.
    for curve, sim_vars in zip(simulation_curves, simulation_ind_vars):
        x, z, vy, vz = sim_vars
        pos = [x, 0, z]
        vel = [0, vy, vz]
        
        # Create raw time steps from 0 to 200.
        time_steps = np.linspace(0, 200, len(curve))
        t_char = normalise_time_to_gcr(pos, vel)
        norm_time = time_steps / t_char
        norm_time = time_steps
        norm_time_and_curve.append((norm_time, curve))
    
    # For the new independent variables, compute new_t_char.
    x_new, z_new, vy_new, vz_new = new_ind_vars
    pos_new = [x_new, 0, z_new]
    vel_new = [0, vy_new, vz_new]
    new_t_char = normalise_time_to_gcr(pos_new, vel_new)
    
    max_normtimes = [norm_time[-1] for norm_time, _ in norm_time_and_curve]
    min_max_normtime = min(max_normtimes)
    # Define the common normalized time grid corresponding to raw time 0 to 200 for the new simulation.
    common_norm_grid = np.linspace(0, min_max_normtime, 200)
    
    # Resample each simulation's curve onto this common normalized grid.
    resampled_curves = []
    for norm_time, curve in norm_time_and_curve:
        resampled = np.interp(common_norm_grid, norm_time, curve)
        resampled_curves.append(resampled)
    resampled_curves = np.array(resampled_curves)
    
    # Interpolate over the multidimensional independent variable space at each normalized time point.
    predicted_curve = []
    for j in range(len(common_norm_grid)):
        data_at_t = resampled_curves[:, j]
        pred_value = griddata(simulation_ind_vars, data_at_t, new_ind_vars, method=interpolation_method)
        predicted_curve.append(pred_value)
    predicted_curve = np.squeeze(np.array(predicted_curve))
    
    # Convert the normalized time grid back to real time.
    predicted_real_time_grid = common_norm_grid * new_t_char  # This will be a grid from 0 to 200.
    
    # Optionally, if you want to enforce a fixed 0–200 grid:
    final_time_grid = np.linspace(0, 200, len(predicted_real_time_grid))
    final_predicted_curve = np.interp(final_time_grid, predicted_real_time_grid, predicted_curve)
    
    return final_time_grid, final_predicted_curve

def predict_interpolated_curve(new_ind_vars, sim_data, interpolation_method='linear'):

    simulation_ind_vars = np.array(sim_data['simulation_ind_vars'])
    simulation_curves = np.array(sim_data['simulation_curves'])


    x_new, z_new, vy_new, vz_new = new_ind_vars

    predicted_curve = []

    for j in range(simulation_curves.shape[1]):
        data_at_t = simulation_curves[:, j]
        pred_value = griddata(simulation_ind_vars, data_at_t, new_ind_vars, method=interpolation_method)
        predicted_curve.append(pred_value)
    
    predicted_curve = np.squeeze(np.array(predicted_curve))
    return predicted_curve

def predict_interpolated_curve_1d_plot(new_x, sim_data, interpolation_method='linear'):

    folders2 = ['/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=0',
            # '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=4',
            '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=8',
            # '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=12',
            '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=16',
            # '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=20',
            '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=24',
            # '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=32',
            '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=36']


    # '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/Interp_test_zs/interp_test_z=40']

    # Convert simulation data to numpy arrays
    simulation_x = np.array(sim_data['simulation_ind_vars'])
    simulation_curves = np.array(sim_data['simulation_curves'])  # shape: (n_samples, n_times)
    
    # Reshape simulation_x to a 2D array for griddata
    points = simulation_x.reshape(-1, 1)
    
    # Ensure new_x is an array, and reshape to 2D (N, 1)
    new_x = np.atleast_1d(new_x)
    new_points = new_x.reshape(-1, 1)
    
    predicted_curve = []
    # For each time step, interpolate data at new_x
    for j in range(simulation_curves.shape[1]):
        data_at_t = simulation_curves[:, j]
        # Perform 1D interpolation using griddata
        pred_value = griddata(points, data_at_t, new_points, method=interpolation_method)
        predicted_curve.append(pred_value)
    
    # Convert list to array and transpose so that each row corresponds to a new x-value
    predicted_curve = np.array(predicted_curve).T

    # If new_x was provided as a scalar, squeeze the result to return a 1D array
    if predicted_curve.shape[0] == 1:
        predicted_curve = predicted_curve.squeeze()
    
    return predicted_curve

def create_simulation_data(data_file):
    
    df = pd.read_csv(data_file)
    df = df.dropna()
    simulation_ind_vars = []
    simulation_curves = []
    
    start_time = time.time()
    # Group the DataFrame by 'init_cond'
    for sim, df_sim in df.groupby('init_cond'):

        params = {var: float(value) for token in sim.split('_') 
                                    for var, value in [token.split('=')]}

        ind_var_vector = [params['x'], params['z'], params['vy'], params['vz']]
        
        mass_loss_curve = df_sim['ml_sr=0.5_m=1e6'].values.astype(float)
        # mass_loss_curve = df_sim['l_sr=0.5_m=1e6'].values.astype(float)
        # Pad the curve if its length is less than 200
        if len(mass_loss_curve) < 200:
            pad_length = 200 - len(mass_loss_curve)
            last_val = mass_loss_curve[-1]
            mass_loss_curve = np.concatenate([mass_loss_curve, np.full(pad_length, last_val)])
        
        simulation_ind_vars.append(ind_var_vector)
        simulation_curves.append(mass_loss_curve)
        
    end_time = time.time()
    print(f"Loop execution time: {end_time - start_time} seconds")
    
    sim_data = {
        'simulation_ind_vars': np.array(simulation_ind_vars),
        'simulation_curves': simulation_curves,  
    }
    
    return sim_data

if __name__ == '__main__':


    # THIS IS FOR MASS LOSS PLOT
    data_file = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/total_data_length_and_mass.csv'
    sim_data = create_simulation_data(data_file)
    new_ind_vars = [[13, 13, 165, 50] ,[25, 23, 190, 21], [35.5, 7.5, 127.5, 20.5]]
    # [35, 16, 180, 20], [6, 39, 20, 20]]

    mass_loss_curves = []

    for new_ind_var in new_ind_vars:

        final_predicted_curve = predict_interpolated_curve(new_ind_var, sim_data, interpolation_method='linear')

        final_predicted_curve = np.array(final_predicted_curve)
        final_cleaned_curve = final_predicted_curve[~np.isnan(final_predicted_curve)]

        time_grid = np.arange(0, len(final_predicted_curve))  # [1,2,...,200]
        valid = ~np.isnan(final_predicted_curve)
        final_filled_curve = np.interp(time_grid, time_grid[valid], final_predicted_curve[valid])
        mass_loss_curves.append(np.array(final_filled_curve) * 100)
        # plt.plot(time_grid, final_filled_curve)

    mass_loss_curve_13_13_165_50, a, b = find_mass_loss_curves('/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/13_13_165_50', [13, 0 ,13], [0, 165, 50], 0.5, 1e6)
    mass_loss_curve_25_23_190_21, _, _ = find_mass_loss_curves('/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/25_23_190_21', [25, 0 ,23], [0, 190, 21], 0.5, 1e6)
    mass_loss_curve_35_7_127_20, _, _ = find_mass_loss_curves('/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/35_7_127_21', [35.5, 0 ,7.5], [0, 127.5, 20.5], 0.5, 1e6)

    mass_loss_curve_13_13_165_50 = np.array(mass_loss_curve_13_13_165_50) * 100
    mass_loss_curve_25_23_190_21 = np.array(mass_loss_curve_25_23_190_21) * 100
    mass_loss_curve_35_7_127_20 = np.array(mass_loss_curve_35_7_127_20) * 100

    fig, axes = plt.subplots(1, 3, figsize=set_size(subplots=(1,2)), sharey=True)

    ax1, ax2, ax3 = axes

    times1 = np.arange(0, len(mass_loss_curves[0])) * 0.018075
    times2 = np.arange(0, len(mass_loss_curves[1])) * 0.018075
    times3 = np.arange(0, len(mass_loss_curves[2])) * 0.018075
    print(mass_loss_curves[0])
    ax1.plot(np.arange(0, len(mass_loss_curve_13_13_165_50)) * 0.018075, mass_loss_curve_13_13_165_50, label='Simulated', linewidth=1, color='black')
    ax1.plot(times1, mass_loss_curves[0], label='Predicted', linestyle='--', color='red', linewidth=1)
    ax1.set_ylabel(r'Mass Loss [$10^6$ M$_\odot$]')
    ax1.set_xlabel('Time [Gyr]')
    ax1.set_xlim(times1[0], times1[-1])
    ax1.legend(frameon=False)

    ax2.plot(np.arange(0, len(mass_loss_curve_25_23_190_21)) * 0.018075, mass_loss_curve_25_23_190_21, label='Simulated', color='black', linewidth=1)
    ax2.plot(times2, mass_loss_curves[1], label='Predicted', linestyle='--', linewidth=1, color='red')
    ax2.set_xlabel('Time [Gyr]')
    ax2.set_xlim(times2[0], times2[-1])
    ax2.legend(frameon=False, loc='upper left')

    ax3.plot(np.arange(0, len(mass_loss_curve_35_7_127_20)) * 0.018075, mass_loss_curve_35_7_127_20, label='Simulated', color='black', linewidth=1)
    ax3.plot(times3, mass_loss_curves[2], label='Predicted', linestyle='--', linewidth=1, color='red')
    ax3.set_xlim(times3[0], times3[-1])
    ax3.set_xlabel('Time [Gyr]')
    ax3.legend(frameon=False)

    def sci_formatter(x, pos):
        return f'{x/1e6:.1f}'

    ax1.yaxis.set_major_formatter(FuncFormatter(sci_formatter))


    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig('mass_loss_curves_predictions.png', dpi = 600, bbox_inches='tight')
    # plt.savefig('mass_loss_curves_predictions.eps', format='eps', bbox_inches='tight')


    # THIS IS FOR LENGTH PLOT
    # data_file = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/total_data_length_and_mass.csv'
    # sim_data = create_simulation_data(data_file)
    # new_ind_vars = [[13, 13, 165, 50] ,[25, 23, 190, 21], [35.5, 7.5, 127.5, 20.5]]
    # # [35, 16, 180, 20], [6, 39, 20, 20]]

    # mass_loss_curves = []

    # for new_ind_var in new_ind_vars:

    #     final_predicted_curve = predict_interpolated_curve(new_ind_var, sim_data, interpolation_method='linear')

    #     final_predicted_curve = np.array(final_predicted_curve)
    #     final_cleaned_curve = final_predicted_curve[~np.isnan(final_predicted_curve)]

    #     time_grid = np.arange(0, len(final_predicted_curve))  # [1,2,...,200]
    #     valid = ~np.isnan(final_predicted_curve)
    #     final_filled_curve = np.interp(time_grid, time_grid[valid], final_predicted_curve[valid])
    #     mass_loss_curves.append(np.array(final_filled_curve) * 100)
    #     # plt.plot(time_grid, final_filled_curve)

    # mass_loss_curve_13_13_165_50, _, _, _ = find_length_of_stream_curves('/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/13_13_165_50', [13, 0 ,13], [0, 165, 50], 0.5, 1e6)
    # mass_loss_curve_25_23_190_21, _, _, _ = find_length_of_stream_curves('/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/25_23_190_21', [25, 0 ,23], [0, 190, 21], 0.5, 1e6)
    # mass_loss_curve_35_7_127_20, _, _, _= find_length_of_stream_curves('/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/35_7_127_21', [35.5, 0 ,7.5], [0, 127.5, 20.5], 0.5, 1e6)

    # mass_loss_curve_13_13_165_50 = np.array(mass_loss_curve_13_13_165_50) * 100
    # mass_loss_curve_25_23_190_21 = np.array(mass_loss_curve_25_23_190_21) * 100
    # mass_loss_curve_35_7_127_20 = np.array(mass_loss_curve_35_7_127_20) * 100

    # fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.83))

    # ax1, ax2, ax3 = axes

    # times1 = np.arange(0, len(mass_loss_curves[0])) * 0.018075
    # times2 = np.arange(0, len(mass_loss_curves[1])) * 0.018075
    # times3 = np.arange(0, len(mass_loss_curves[2])) * 0.018075

    # ax1.plot(np.arange(0, len(mass_loss_curve_13_13_165_50)) * 0.018075, mass_loss_curve_13_13_165_50, label='Simulated', linewidth=1, color='black')
    # ax1.plot(times1, mass_loss_curves[0], label='Predicted', linestyle='--', color='red', linewidth=1)
    # ax1.set_ylabel(r'Mass Loss [$10^6$ M$_\odot$]')
    # ax1.set_xlabel('Time [Gyr]')
    # ax1.set_xlim(times1[0], times1[-1])
    # ax1.legend(frameon=False)

    # ax2.plot(np.arange(0, len(mass_loss_curve_25_23_190_21)) * 0.018075, mass_loss_curve_25_23_190_21, label='Simulated', color='black', linewidth=1)
    # ax2.plot(times2, mass_loss_curves[1], label='Predicted', linestyle='--', linewidth=1, color='red')
    # ax2.set_xlabel('Time [Gyr]')
    # ax2.set_xlim(times2[0], times2[-1])
    # ax2.legend(frameon=False, loc='upper left')

    # ax3.plot(np.arange(0, len(mass_loss_curve_35_7_127_20)) * 0.018075, mass_loss_curve_35_7_127_20, label='Simulated', color='black', linewidth=1)
    # ax3.plot(times3, mass_loss_curves[2], label='Predicted', linestyle='--', linewidth=1, color='red')
    # ax3.set_xlim(times3[0], times3[-1])
    # ax3.set_xlabel('Time [Gyr]')
    # ax3.legend(frameon=False)

    # # def sci_formatter(x, pos):
    # #     return f'{x/1e6:.1f}'

    # # ax1.yaxis.set_major_formatter(FuncFormatter(sci_formatter))


    # plt.subplots_adjust(hspace=0, wspace=0)
    # plt.savefig('left_length_curves_predictions.png', dpi = 500, bbox_inches='tight')