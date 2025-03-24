import numpy as np
import pandas as pd
import glob
import os
import sys
sys.path.append(r'/home/afacey/MPhysProj/MySims/exptool')
from exptool.io import particle
from collections import defaultdict
from predict_stream_params import predict_mass_loss, predict_stream_lengths
from find_stream_params import find_mass_loss_curves, find_length_of_stream_curves
import math

def adjust_curve_length(curve, T):

    curve = np.array(curve)  
    L = len(curve)
    if L == T:
        return curve
    elif L > T:

        start = (L - T) // 2
        return curve[start:start + T]
    else:

        pad_length = T - L
        return np.concatenate([curve, np.full(pad_length, np.nan)])

def extract_and_round_sim(sim):

    tokens = sim.split('_')
    new_tokens = []
    for token in tokens:
        if token.startswith('sr=') or token.startswith('mass=') or token.startswith('batch='):
            break
        if '=' in token:
            key, val = token.split('=', 1)
            try:
                fval = float(val)
                new_tokens.append(f"{key}={fval:.1f}")
            except ValueError:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return "_".join(new_tokens)

def generate_dataframe_just_orbit():

    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/BigCollectionOfSims'
    orbital_data_folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/find_grid_orbitals_orig_init'

    out_files_orbit = glob.glob(os.path.join(orbital_data_folder, 'OUT.run1.*'))
    out_files_orbit.sort() 
    
    name_path = os.path.join(orbital_data_folder, 'all_orbits_ids_and_names.csv')
    orbital_data_ids = pd.read_csv(name_path)
    

    names = orbital_data_ids['init_string'].values 
    N = len(names)  
    T = len(out_files_orbit)

    conv_factor = np.sqrt(1/4.3e-6)
    
    x_list, y_list, z_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []

    for out_file in out_files_orbit:
        print(out_file)
        data = particle.Input(filename=out_file, comp='star', legacy=True)

        x_list.append(data.xpos)
        y_list.append(data.ypos)
        z_list.append(data.zpos)
        vx_list.append(data.xvel / conv_factor)
        vy_list.append(data.yvel / conv_factor)
        vz_list.append(data.zvel / conv_factor)

    x_array = np.array(x_list, dtype=np.float16)
    y_array = np.array(y_list, dtype=np.float16)
    z_array = np.array(z_list,  dtype=np.float16)
    vx_array = np.array(vx_list, dtype=np.float16)
    vy_array = np.array(vy_list, dtype=np.float16)
    vz_array = np.array(vz_list, dtype=np.float16)

    x_flat = x_array.T.flatten()
    y_flat = y_array.T.flatten()
    z_flat = z_array.T.flatten()
    vx_flat = vx_array.T.flatten()
    vy_flat = vy_array.T.flatten()
    vz_flat = vz_array.T.flatten()

    names_repeated = np.repeat(names, T)

    final_df = pd.DataFrame({
        'init_cond': names_repeated,
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'vx': vx_flat,
        'vy': vy_flat,
        'vz': vz_flat
    })

    output_csv_path = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/only_inits_orbits.csv'
    final_df.to_csv(output_csv_path, index=False)
    return final_df

def generate_dataframe_mass_loss():

    ##### GET ORBIT DATA #####
    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/BigCollectionOfSims'
    orbital_data_folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/orbit_for_initial_number_sims'

    out_files_orbit = glob.glob(os.path.join(orbital_data_folder, 'OUT.run1.*'))
    out_files_orbit.sort()
    
    name_path = os.path.join(orbital_data_folder, 'all_orbits_ids_and_names.csv')
    orbital_data_ids = pd.read_csv(name_path)
    
    names = orbital_data_ids['init_string'].values
    N = len(names)
    T = len(out_files_orbit)

    conv_factor = np.sqrt(1/4.3e-6)
    
    x_list, y_list, z_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []


    for out_file in out_files_orbit:
        data = particle.Input(filename=out_file, comp='star', legacy=True)

        x_list.append(data.xpos)
        y_list.append(data.ypos)
        z_list.append(data.zpos)
        vx_list.append(data.xvel / conv_factor)
        vy_list.append(data.yvel / conv_factor)
        vz_list.append(data.zvel / conv_factor)

    x_array = np.array(x_list, dtype=np.float16)
    y_array = np.array(y_list, dtype=np.float16)
    z_array = np.array(z_list,  dtype=np.float16)
    vx_array = np.array(vx_list, dtype=np.float16)
    vy_array = np.array(vy_list, dtype=np.float16)
    vz_array = np.array(vz_list, dtype=np.float16)

    x_flat = x_array.T.flatten()
    y_flat = y_array.T.flatten()
    z_flat = z_array.T.flatten()
    vx_flat = vx_array.T.flatten()
    vy_flat = vy_array.T.flatten()
    vz_flat = vz_array.T.flatten()

    ##### GET MASS LOSS DATA #####
    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/BigCollectionOfSims'
    mass_loss_sr001_m1e6 = []
    mass_loss_sr001_m1e7 = []
    mass_loss_sr01_m1e6 = []
    mass_loss_sr01_m1e7 = []
    mass_loss_sr1_m1e6 = []
    mass_loss_sr1_m1e7 = []

    mass_loss_sr001_m1e6_sim = []
    mass_loss_sr001_m1e7_sim = []
    mass_loss_sr01_m1e6_sim = []
    mass_loss_sr01_m1e7_sim = []
    mass_loss_sr1_m1e6_sim = []
    mass_loss_sr1_m1e7_sim = []

    l_length_sr001_m1e6 = []
    l_length_sr001_m1e7 = []
    l_length_sr01_m1e6 = []
    l_length_sr01_m1e7 = []
    l_length_sr1_m1e6 = []
    l_length_sr1_m1e7 = []

    r_length_sr001_m1e6 = []
    r_length_sr001_m1e7 = []
    r_length_sr01_m1e6 = []
    r_length_sr01_m1e7 = []
    r_length_sr1_m1e6 = []
    r_length_sr1_m1e7 = []

    rtol = 1e-3
    for folder in os.listdir(base_path):
        print(folder)
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for sim in os.listdir(folder_path):
                sim_path = os.path.join(folder_path, sim)
                mass_loss_curve, norm_time, t_char, sr, mass = get_mass_loss(sim_path, sim)
                right_lengths = np.arange(0, 200)
                left_lengths = np.arange(0, 200)

                sim = sim.split('sr=_')[0]
                if sr == 0.5:
                    continue
                elif sr == 0.01 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr001_m1e6.append(mass_loss_curve)
                    l_length_sr001_m1e6.append(left_lengths)
                    r_length_sr001_m1e6.append(right_lengths)
                    mass_loss_sr001_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 0.01 and math.isclose(mass, 10000000.0, rel_tol=rtol):
                    mass_loss_sr001_m1e7.append(mass_loss_curve)
                    l_length_sr001_m1e7.append(left_lengths)
                    r_length_sr001_m1e7.append(right_lengths)
                    mass_loss_sr001_m1e7_sim.append(extract_and_round_sim(sim))
                elif sr == 0.1 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr01_m1e6.append(mass_loss_curve)
                    l_length_sr01_m1e6.append(left_lengths)
                    r_length_sr01_m1e6.append(right_lengths)
                    mass_loss_sr01_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 0.1 and math.isclose(mass, 10000000.0, rel_tol=rtol):
                    mass_loss_sr01_m1e7.append(mass_loss_curve)
                    l_length_sr01_m1e7.append(left_lengths)
                    r_length_sr01_m1e7.append(right_lengths)
                    mass_loss_sr01_m1e7_sim.append(extract_and_round_sim(sim))
                elif sr == 1 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr1_m1e6.append(mass_loss_curve)
                    l_length_sr1_m1e6.append(left_lengths)
                    r_length_sr1_m1e6.append(right_lengths)
                    mass_loss_sr1_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 1 and math.isclose(mass, 10000000.0, rel_tol=rtol):
                    mass_loss_sr1_m1e7.append(mass_loss_curve)
                    l_length_sr1_m1e7.append(left_lengths)
                    r_length_sr1_m1e7.append(right_lengths)
                    mass_loss_sr1_m1e7_sim.append(extract_and_round_sim(sim))

    names_repeated = np.repeat(names, T)

    final_df = pd.DataFrame({
        'init_cond': names_repeated,
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'vx': vx_flat,
        'vy': vy_flat,
        'vz': vz_flat
    })

    def build_mass_loss_dict(sim_list, curve_list):
        d = {}
        for sim, curve in zip(sim_list, curve_list):
            base = sim.split('_sr=')[0]
            d[base] = curve
        return d

    mass_loss_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, mass_loss_sr001_m1e6)
    mass_loss_dict_sr001_m1e7 = build_mass_loss_dict(mass_loss_sr001_m1e7_sim, mass_loss_sr001_m1e7)
    mass_loss_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim,  mass_loss_sr01_m1e6)
    mass_loss_dict_sr01_m1e7  = build_mass_loss_dict(mass_loss_sr01_m1e7_sim,  mass_loss_sr01_m1e7)
    mass_loss_dict_sr1_m1e6   = build_mass_loss_dict(mass_loss_sr1_m1e6_sim,   mass_loss_sr1_m1e6)
    mass_loss_dict_sr1_m1e7   = build_mass_loss_dict(mass_loss_sr1_m1e7_sim,   mass_loss_sr1_m1e7)

    l_length_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, l_length_sr001_m1e6)
    l_length_dict_sr001_m1e7 = build_mass_loss_dict(mass_loss_sr001_m1e7_sim, l_length_sr001_m1e7)
    l_length_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim,  l_length_sr01_m1e6)
    l_length_dict_sr01_m1e7  = build_mass_loss_dict(mass_loss_sr01_m1e7_sim,  l_length_sr01_m1e7)
    l_length_dict_sr1_m1e6   = build_mass_loss_dict(mass_loss_sr1_m1e6_sim,   l_length_sr1_m1e6)
    l_length_dict_sr1_m1e7   = build_mass_loss_dict(mass_loss_sr1_m1e7_sim,   l_length_sr1_m1e7)

    col1 = "ml_sr=0.01_m=1e6"
    col2 = "ml_for=0.01_m=1e7"
    col3 = "ml_sr=0.1_m=1e6"
    col4 = "ml_sr=0.1_m=1e7"
    col5 = "ml_sr=1_m=1e6"
    col6 = "ml_sr=1_m=1e7"
    col7 = "l_sr=0.01_m=1e6"
    col8 = "l_sr=0.01_m=1e7"
    col9 = "l_sr=0.1_m=1e6"
    col10 = "l_sr=0.1_m=1e7"
    col11 = "l_sr=1_m=1e6"
    col12 = "l_sr=1_m=1e7"
    col13 = "r_sr=0.01_m=1e6"
    col14 = "r_sr=0.01_m=1e7"
    col15 = "r_sr=0.1_m=1e6"
    col16 = "r_sr=0.1_m=1e7"
    col17 = "r_sr=1_m=1e6"
    col18 = "r_sr=1_m=1e7"

    # Initialize these columns in final_df with NaN.
    for col in [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18]:
        final_df[col] = np.nan

    for sim in final_df['init_cond'].unique():
        mask = final_df['init_cond'] == sim
        
        if sim in mass_loss_dict_sr001_m1e6:

            ml_curve = mass_loss_dict_sr001_m1e6[sim]
            l_curve = l_length_dict_sr001_m1e6[sim]
            r_curve = l_length_dict_sr001_m1e6[sim]
            final_df.loc[mask, col1] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col7] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col13] = adjust_curve_length(r_curve, T)


        if sim in mass_loss_dict_sr001_m1e7:

            ml_curve = mass_loss_dict_sr001_m1e7[sim]
            l_curve = l_length_dict_sr001_m1e7[sim]
            r_curve = l_length_dict_sr001_m1e7[sim]
            final_df.loc[mask, col2] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col8] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col14] = adjust_curve_length(r_curve, T)


        if sim in mass_loss_dict_sr01_m1e6:

            ml_curve = mass_loss_dict_sr01_m1e6[sim]
            l_curve = l_length_dict_sr01_m1e6[sim]
            r_curve = l_length_dict_sr01_m1e6[sim]
            final_df.loc[mask, col3] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col9] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col15] = adjust_curve_length(r_curve, T)

        if sim in mass_loss_dict_sr01_m1e7:

            ml_curve = mass_loss_dict_sr01_m1e7[sim]
            l_curve = l_length_dict_sr01_m1e7[sim]
            r_curve = l_length_dict_sr01_m1e7[sim]
            final_df.loc[mask, col4] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col10] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col16] = adjust_curve_length(r_curve, T)

        if sim in mass_loss_dict_sr1_m1e6:

            ml_curve = mass_loss_dict_sr1_m1e6[sim]
            l_curve = l_length_dict_sr1_m1e6[sim]
            r_curve = l_length_dict_sr1_m1e6[sim]
            final_df.loc[mask, col5] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col11] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col17] = adjust_curve_length(r_curve, T)

        if sim in mass_loss_dict_sr1_m1e7:

            ml_curve = mass_loss_dict_sr1_m1e7[sim]
            l_curve = l_length_dict_sr1_m1e7[sim]
            r_curve = l_length_dict_sr1_m1e7[sim]
            final_df.loc[mask, col6] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col12] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col18] = adjust_curve_length(r_curve, T)

    output_csv_path = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/total_data_2.csv'
    final_df.to_csv(output_csv_path, index=False)
    return final_df

def generate_dataframe_lengths():

    ##### GET ORBIT DATA #####
    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/BigCollectionOfSims'
    orbital_data_folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/orbit_for_initial_number_sims'

    out_files_orbit = glob.glob(os.path.join(orbital_data_folder, 'OUT.run1.*'))
    out_files_orbit.sort() 

    name_path = os.path.join(orbital_data_folder, 'all_orbits_ids_and_names.csv')
    orbital_data_ids = pd.read_csv(name_path)
    
    names = orbital_data_ids['init_string'].values 
    N = len(names)         
    T = len(out_files_orbit) 
    conv_factor = np.sqrt(1/4.3e-6)
    
    x_list, y_list, z_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []

    for out_file in out_files_orbit:
        data = particle.Input(filename=out_file, comp='star', legacy=True)

        x_list.append(data.xpos)
        y_list.append(data.ypos)
        z_list.append(data.zpos)
        vx_list.append(data.xvel / conv_factor)
        vy_list.append(data.yvel / conv_factor)
        vz_list.append(data.zvel / conv_factor)


    x_array = np.array(x_list, dtype=np.float16)
    y_array = np.array(y_list, dtype=np.float16)
    z_array = np.array(z_list,  dtype=np.float16)
    vx_array = np.array(vx_list, dtype=np.float16)
    vy_array = np.array(vy_list, dtype=np.float16)
    vz_array = np.array(vz_list, dtype=np.float16)

    x_flat = x_array.T.flatten()
    y_flat = y_array.T.flatten()
    z_flat = z_array.T.flatten()
    vx_flat = vx_array.T.flatten()
    vy_flat = vy_array.T.flatten()
    vz_flat = vz_array.T.flatten()

    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/BigCollectionOfSims'
    mass_loss_sr001_m1e6 = []
    mass_loss_sr001_m1e7 = []
    mass_loss_sr01_m1e6 = []
    mass_loss_sr01_m1e7 = []
    mass_loss_sr1_m1e6 = []
    mass_loss_sr1_m1e7 = []

    mass_loss_sr001_m1e6_sim = []
    mass_loss_sr001_m1e7_sim = []
    mass_loss_sr01_m1e6_sim = []
    mass_loss_sr01_m1e7_sim = []
    mass_loss_sr1_m1e6_sim = []
    mass_loss_sr1_m1e7_sim = []

    l_length_sr001_m1e6 = []
    l_length_sr001_m1e7 = []
    l_length_sr01_m1e6 = []
    l_length_sr01_m1e7 = []
    l_length_sr1_m1e6 = []
    l_length_sr1_m1e7 = []

    r_length_sr001_m1e6 = []
    r_length_sr001_m1e7 = []
    r_length_sr01_m1e6 = []
    r_length_sr01_m1e7 = []
    r_length_sr1_m1e6 = []
    r_length_sr1_m1e7 = []

    rtol = 1e-3
    for folder in os.listdir(base_path):
        print(folder)
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for sim in os.listdir(folder_path):
                sim_path = os.path.join(folder_path, sim)
                mass_loss_curve, norm_time, t_char, sr, mass, positions_origin, velocities_origin = get_mass_loss(sim_path, sim)
                print(positions_origin)
                left_lengths, right_lengths, time_steps, t_chars = find_length_of_stream_curves(sim_path, positions_origin, velocities_origin, sr, mass)
                print(f'Len left: {len(left_lengths)}')
                # right_lengths = adasdasdasd
                sim = sim.split('sr=_')[0]
                if sr == 0.5:
                    continue
                elif sr == 0.01 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr001_m1e6.append(mass_loss_curve)
                    l_length_sr001_m1e6.append(left_lengths)
                    r_length_sr001_m1e6.append(right_lengths)
                    mass_loss_sr001_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 0.01 and math.isclose(mass, 10000000.0, rel_tol=rtol):
                    mass_loss_sr001_m1e7.append(mass_loss_curve)
                    l_length_sr001_m1e7.append(left_lengths)
                    r_length_sr001_m1e7.append(right_lengths)
                    mass_loss_sr001_m1e7_sim.append(extract_and_round_sim(sim))
                elif sr == 0.1 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr01_m1e6.append(mass_loss_curve)
                    l_length_sr01_m1e6.append(left_lengths)
                    r_length_sr01_m1e6.append(right_lengths)
                    mass_loss_sr01_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 0.1 and math.isclose(mass, 10000000.0, rel_tol=rtol):
                    mass_loss_sr01_m1e7.append(mass_loss_curve)
                    l_length_sr01_m1e7.append(left_lengths)
                    r_length_sr01_m1e7.append(right_lengths)
                    mass_loss_sr01_m1e7_sim.append(extract_and_round_sim(sim))
                elif sr == 1 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr1_m1e6.append(mass_loss_curve)
                    l_length_sr1_m1e6.append(left_lengths)
                    r_length_sr1_m1e6.append(right_lengths)
                    mass_loss_sr1_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 1 and math.isclose(mass, 10000000.0, rel_tol=rtol):
                    mass_loss_sr1_m1e7.append(mass_loss_curve)
                    l_length_sr1_m1e7.append(left_lengths)
                    r_length_sr1_m1e7.append(right_lengths)
                    mass_loss_sr1_m1e7_sim.append(extract_and_round_sim(sim))

    names_repeated = np.repeat(names, T)

    final_df = pd.DataFrame({
        'init_cond': names_repeated,
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'vx': vx_flat,
        'vy': vy_flat,
        'vz': vz_flat
    })

    def build_mass_loss_dict(sim_list, curve_list):
        d = {}
        for sim, curve in zip(sim_list, curve_list):
            base = sim.split('_sr=')[0]
            d[base] = curve
        return d

    mass_loss_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, mass_loss_sr001_m1e6)
    mass_loss_dict_sr001_m1e7 = build_mass_loss_dict(mass_loss_sr001_m1e7_sim, mass_loss_sr001_m1e7)
    mass_loss_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim,  mass_loss_sr01_m1e6)
    mass_loss_dict_sr01_m1e7  = build_mass_loss_dict(mass_loss_sr01_m1e7_sim,  mass_loss_sr01_m1e7)
    mass_loss_dict_sr1_m1e6   = build_mass_loss_dict(mass_loss_sr1_m1e6_sim,   mass_loss_sr1_m1e6)
    mass_loss_dict_sr1_m1e7   = build_mass_loss_dict(mass_loss_sr1_m1e7_sim,   mass_loss_sr1_m1e7)

    l_length_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, l_length_sr001_m1e6)
    l_length_dict_sr001_m1e7 = build_mass_loss_dict(mass_loss_sr001_m1e7_sim, l_length_sr001_m1e7)
    l_length_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim,  l_length_sr01_m1e6)
    l_length_dict_sr01_m1e7  = build_mass_loss_dict(mass_loss_sr01_m1e7_sim,  l_length_sr01_m1e7)
    l_length_dict_sr1_m1e6   = build_mass_loss_dict(mass_loss_sr1_m1e6_sim,   l_length_sr1_m1e6)
    l_length_dict_sr1_m1e7   = build_mass_loss_dict(mass_loss_sr1_m1e7_sim,   l_length_sr1_m1e7)

    col1 = "ml_sr=0.01_m=1e6"
    col2 = "ml_for=0.01_m=1e7"
    col3 = "ml_sr=0.1_m=1e6"
    col4 = "ml_sr=0.1_m=1e7"
    col5 = "ml_sr=1_m=1e6"
    col6 = "ml_sr=1_m=1e7"
    col7 = "l_sr=0.01_m=1e6"
    col8 = "l_sr=0.01_m=1e7"
    col9 = "l_sr=0.1_m=1e6"
    col10 = "l_sr=0.1_m=1e7"
    col11 = "l_sr=1_m=1e6"
    col12 = "l_sr=1_m=1e7"
    col13 = "r_sr=0.01_m=1e6"
    col14 = "r_sr=0.01_m=1e7"
    col15 = "r_sr=0.1_m=1e6"
    col16 = "r_sr=0.1_m=1e7"
    col17 = "r_sr=1_m=1e6"
    col18 = "r_sr=1_m=1e7"

    for col in [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18]:
        final_df[col] = np.nan

    for sim in final_df['init_cond'].unique():
        mask = final_df['init_cond'] == sim
        
        if sim in mass_loss_dict_sr001_m1e6:

            ml_curve = mass_loss_dict_sr001_m1e6[sim]
            l_curve = l_length_dict_sr001_m1e6[sim]
            r_curve = l_length_dict_sr001_m1e6[sim]
            final_df.loc[mask, col1] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col7] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col13] = adjust_curve_length(r_curve, T)


        if sim in mass_loss_dict_sr001_m1e7:

            ml_curve = mass_loss_dict_sr001_m1e7[sim]
            l_curve = l_length_dict_sr001_m1e7[sim]
            r_curve = l_length_dict_sr001_m1e7[sim]
            final_df.loc[mask, col2] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col8] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col14] = adjust_curve_length(r_curve, T)


        if sim in mass_loss_dict_sr01_m1e6:

            ml_curve = mass_loss_dict_sr01_m1e6[sim]
            l_curve = l_length_dict_sr01_m1e6[sim]
            r_curve = l_length_dict_sr01_m1e6[sim]
            final_df.loc[mask, col3] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col9] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col15] = adjust_curve_length(r_curve, T)

        if sim in mass_loss_dict_sr01_m1e7:

            ml_curve = mass_loss_dict_sr01_m1e7[sim]
            l_curve = l_length_dict_sr01_m1e7[sim]
            r_curve = l_length_dict_sr01_m1e7[sim]
            final_df.loc[mask, col4] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col10] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col16] = adjust_curve_length(r_curve, T)

        if sim in mass_loss_dict_sr1_m1e6:

            ml_curve = mass_loss_dict_sr1_m1e6[sim]
            l_curve = l_length_dict_sr1_m1e6[sim]
            r_curve = l_length_dict_sr1_m1e6[sim]
            final_df.loc[mask, col5] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col11] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col17] = adjust_curve_length(r_curve, T)

        if sim in mass_loss_dict_sr1_m1e7:

            ml_curve = mass_loss_dict_sr1_m1e7[sim]
            l_curve = l_length_dict_sr1_m1e7[sim]
            r_curve = l_length_dict_sr1_m1e7[sim]
            final_df.loc[mask, col6] = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col12] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col18] = adjust_curve_length(r_curve, T)

    output_csv_path = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/total_data_lengths_tester.csv'
    final_df.to_csv(output_csv_path, index=False)
    return final_df

def get_mass_loss(folder_path, sim):
    params = sim.split('_')
    x = float(params[0].split('=')[1])
    y = float(params[1].split('=')[1])
    z = float(params[2].split('=')[1])
    vx = float(params[3].split('=')[1])
    vy = float(params[4].split('=')[1])
    vz = float(params[5].split('=')[1])
    sr = float(params[6].split('=')[1])
    mass = float(params[7].split('=')[1])
    positions = [x, y, z]
    velocities = [vx, vy, vz]
    mass_loss_curve, norm_time, t_char = find_mass_loss_curves(folder_path, positions, velocities, sr, mass)
    mass_loss_curve, norm_time, t_char = np.arange(0, 200), np.arange(0, 200), np.ones(200)
    return mass_loss_curve, norm_time, t_char, sr, mass, positions, velocities

def generate_dataframe_lengths_mass_loss_both():

    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/NewBigCollectionOfSims'
    orbital_data_folder = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/orbit_for_initial_number_sims'

    out_files_orbit = glob.glob(os.path.join(orbital_data_folder, 'OUT.run1.*'))
    out_files_orbit.sort()
    
    name_path = os.path.join(orbital_data_folder, 'all_orbits_ids_and_names.csv')
    orbital_data_ids = pd.read_csv(name_path)
    

    names = orbital_data_ids['init_string'].values
    N = len(names)
    T = len(out_files_orbit)

    conv_factor = np.sqrt(1/4.3e-6)
    
    x_list, y_list, z_list = [], [], []
    vx_list, vy_list, vz_list = [], [], []

    for out_file in out_files_orbit:
        data = particle.Input(filename=out_file, comp='star', legacy=True)

        x_list.append(data.xpos)
        y_list.append(data.ypos)
        z_list.append(data.zpos)
        vx_list.append(data.xvel / conv_factor)
        vy_list.append(data.yvel / conv_factor)
        vz_list.append(data.zvel / conv_factor)

    x_array = np.array(x_list, dtype=np.float16)
    y_array = np.array(y_list, dtype=np.float16)
    z_array = np.array(z_list,  dtype=np.float16)
    vx_array = np.array(vx_list, dtype=np.float16)
    vy_array = np.array(vy_list, dtype=np.float16)
    vz_array = np.array(vz_list, dtype=np.float16)

    x_flat = x_array.T.flatten()
    y_flat = y_array.T.flatten()
    z_flat = z_array.T.flatten()
    vx_flat = vx_array.T.flatten()
    vy_flat = vy_array.T.flatten()
    vz_flat = vz_array.T.flatten()

    base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/NewBigCollectionOfSims'
    mass_loss_sr001_m1e5 = []
    mass_loss_sr001_m1e6 = []
    mass_loss_sr01_m1e5 = []
    mass_loss_sr01_m1e6 = []
    mass_loss_sr05_m1e5 = []
    mass_loss_sr05_m1e6 = []

    mass_loss_sr001_m1e5_sim = []
    mass_loss_sr001_m1e6_sim = []
    mass_loss_sr01_m1e5_sim = []
    mass_loss_sr01_m1e6_sim = []
    mass_loss_sr05_m1e5_sim = []
    mass_loss_sr05_m1e6_sim = []

    l_length_sr001_m1e5 = []
    l_length_sr001_m1e6 = []
    l_length_sr01_m1e5 = []
    l_length_sr01_m1e6 = []
    l_length_sr05_m1e5 = []
    l_length_sr05_m1e6 = []

    r_length_sr001_m1e5 = []
    r_length_sr001_m1e6 = []
    r_length_sr01_m1e5 = []
    r_length_sr01_m1e6 = []
    r_length_sr05_m1e5 = []
    r_length_sr05_m1e6 = []

    rtol = 1e-3
    counter = 0
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for sim in os.listdir(folder_path):
                params = sim.split('_')
                vz = float(params[5].split('=')[1])
                if vz < 0:
                    continue
                else:
                    counter += 0
                    print(sim)
                    print(counter)
                    sim_path = os.path.join(folder_path, sim)
                    mass_loss_curve, norm_time, t_char, sr, mass, positions_origin, velocities_origin = get_mass_loss(sim_path, sim)
                    print(positions_origin)
                    left_lengths, right_lengths, time_steps, t_chars = find_length_of_stream_curves(sim_path, positions_origin, velocities_origin, sr, mass)
                    print(f'Len of files: {len(left_lengths)}, {len(right_lengths), len(mass_loss_curve)}')
                    
                # right_lengths = adasdasdasd
                sim = sim.split('sr=_')[0]

                if sr == 0.01 and math.isclose(mass, 100000.0, rel_tol=rtol):
                    mass_loss_sr001_m1e5.append(mass_loss_curve)
                    l_length_sr001_m1e5.append(left_lengths)
                    r_length_sr001_m1e5.append(right_lengths)
                    mass_loss_sr001_m1e5_sim.append(extract_and_round_sim(sim))
                elif sr == 0.01 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr001_m1e6.append(mass_loss_curve)
                    l_length_sr001_m1e6.append(left_lengths)
                    r_length_sr001_m1e6.append(right_lengths)
                    mass_loss_sr001_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 0.1 and math.isclose(mass, 100000.0, rel_tol=rtol):
                    mass_loss_sr01_m1e5.append(mass_loss_curve)
                    l_length_sr01_m1e5.append(left_lengths)
                    r_length_sr01_m1e5.append(right_lengths)
                    mass_loss_sr01_m1e5_sim.append(extract_and_round_sim(sim))
                elif sr == 0.1 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr01_m1e6.append(mass_loss_curve)
                    l_length_sr01_m1e6.append(left_lengths)
                    r_length_sr01_m1e6.append(right_lengths)
                    mass_loss_sr01_m1e6_sim.append(extract_and_round_sim(sim))
                elif sr == 0.5 and math.isclose(mass, 100000.0, rel_tol=rtol):
                    mass_loss_sr05_m1e5.append(mass_loss_curve)
                    l_length_sr05_m1e5.append(left_lengths)
                    r_length_sr05_m1e5.append(right_lengths)
                    mass_loss_sr05_m1e5_sim.append(extract_and_round_sim(sim))
                elif sr == 0.5 and math.isclose(mass, 1000000.0, rel_tol=rtol):
                    mass_loss_sr05_m1e6.append(mass_loss_curve)
                    l_length_sr05_m1e6.append(left_lengths)
                    r_length_sr05_m1e6.append(right_lengths)
                    mass_loss_sr05_m1e6_sim.append(extract_and_round_sim(sim))


    names_repeated = np.repeat(names, T)

    final_df = pd.DataFrame({
        'init_cond': names_repeated,
        'x': x_flat,
        'y': y_flat,
        'z': z_flat,
        'vx': vx_flat,
        'vy': vy_flat,
        'vz': vz_flat
    })

    def build_mass_loss_dict(sim_list, curve_list):
        d = {}
        for sim, curve in zip(sim_list, curve_list):
            base = sim.split('_sr=')[0]
            d[base] = curve
        return d

    mass_loss_dict_sr001_m1e5 = build_mass_loss_dict(mass_loss_sr001_m1e5_sim, mass_loss_sr001_m1e5)
    mass_loss_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, mass_loss_sr001_m1e6)
    mass_loss_dict_sr01_m1e5  = build_mass_loss_dict(mass_loss_sr01_m1e5_sim,  mass_loss_sr01_m1e5)
    mass_loss_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim,  mass_loss_sr01_m1e6)
    mass_loss_dict_sr05_m1e5  = build_mass_loss_dict(mass_loss_sr05_m1e5_sim,  mass_loss_sr05_m1e5)
    mass_loss_dict_sr05_m1e6  = build_mass_loss_dict(mass_loss_sr05_m1e6_sim,  mass_loss_sr05_m1e6)

    l_length_dict_sr001_m1e5 = build_mass_loss_dict(mass_loss_sr001_m1e5_sim, l_length_sr001_m1e5)
    l_length_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, l_length_sr001_m1e6)
    l_length_dict_sr01_m1e5  = build_mass_loss_dict(mass_loss_sr01_m1e5_sim,  l_length_sr01_m1e5)
    l_length_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim,  l_length_sr01_m1e6)
    l_length_dict_sr05_m1e5  = build_mass_loss_dict(mass_loss_sr05_m1e5_sim,  l_length_sr05_m1e5)
    l_length_dict_sr05_m1e6  = build_mass_loss_dict(mass_loss_sr05_m1e6_sim,  l_length_sr05_m1e6)

    r_length_dict_sr001_m1e5 = build_mass_loss_dict(mass_loss_sr001_m1e5_sim, r_length_sr001_m1e5)
    r_length_dict_sr001_m1e6 = build_mass_loss_dict(mass_loss_sr001_m1e6_sim, r_length_sr001_m1e6)
    r_length_dict_sr01_m1e5  = build_mass_loss_dict(mass_loss_sr01_m1e5_sim, r_length_sr01_m1e5)
    r_length_dict_sr01_m1e6  = build_mass_loss_dict(mass_loss_sr01_m1e6_sim, r_length_sr01_m1e6)
    r_length_dict_sr05_m1e5  = build_mass_loss_dict(mass_loss_sr05_m1e5_sim, r_length_sr05_m1e5)
    r_length_dict_sr05_m1e6  = build_mass_loss_dict(mass_loss_sr05_m1e6_sim, r_length_sr05_m1e6)



    col1  = "ml_sr=0.01_m=1e5"
    col2  = "ml_sr=0.01_m=1e6"
    col3  = "ml_sr=0.1_m=1e5"
    col4  = "ml_sr=0.1_m=1e6"
    col5  = "ml_sr=0.5_m=1e5"
    col6  = "ml_sr=0.5_m=1e6"

    col7  = "l_sr=0.01_m=1e5"
    col8  = "l_sr=0.01_m=1e6"
    col9  = "l_sr=0.1_m=1e5"
    col10 = "l_sr=0.1_m=1e6"
    col11 = "l_sr=0.5_m=1e5"
    col12 = "l_sr=0.5_m=1e6"

    col13 = "r_sr=0.01_m=1e5"
    col14 = "r_sr=0.01_m=1e6"
    col15 = "r_sr=0.1_m=1e5"
    col16 = "r_sr=0.1_m=1e6"
    col17 = "r_sr=0.5_m=1e5"
    col18 = "r_sr=0.5_m=1e6"

    for col in [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18]:
        final_df[col] = np.nan

    for sim in final_df['init_cond'].unique():
        mask = final_df['init_cond'] == sim
        
        if sim in mass_loss_dict_sr001_m1e5:
            ml_curve = mass_loss_dict_sr001_m1e5[sim]
            l_curve  = l_length_dict_sr001_m1e5[sim]
            r_curve  = r_length_dict_sr001_m1e5[sim]
            final_df.loc[mask, col1]  = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col7]  = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col13] = adjust_curve_length(r_curve, T)
        
        if sim in mass_loss_dict_sr001_m1e6:
            ml_curve = mass_loss_dict_sr001_m1e6[sim]
            l_curve  = l_length_dict_sr001_m1e6[sim]
            r_curve  = r_length_dict_sr001_m1e6[sim]
            final_df.loc[mask, col2]  = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col8]  = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col14] = adjust_curve_length(r_curve, T)
        
        if sim in mass_loss_dict_sr01_m1e5:
            ml_curve = mass_loss_dict_sr01_m1e5[sim]
            l_curve  = l_length_dict_sr01_m1e5[sim]
            r_curve  = r_length_dict_sr01_m1e5[sim]
            final_df.loc[mask, col3]  = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col9]  = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col15] = adjust_curve_length(r_curve, T)
        
        if sim in mass_loss_dict_sr01_m1e6:
            ml_curve = mass_loss_dict_sr01_m1e6[sim]
            l_curve  = l_length_dict_sr01_m1e6[sim]
            r_curve  = r_length_dict_sr01_m1e6[sim]
            final_df.loc[mask, col4]  = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col10] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col16] = adjust_curve_length(r_curve, T)
        
        if sim in mass_loss_dict_sr05_m1e5:
            ml_curve = mass_loss_dict_sr05_m1e5[sim]
            l_curve  = l_length_dict_sr05_m1e5[sim]
            r_curve  = r_length_dict_sr05_m1e5[sim]
            final_df.loc[mask, col5]  = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col11] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col17] = adjust_curve_length(r_curve, T)
        
        if sim in mass_loss_dict_sr05_m1e6:
            ml_curve = mass_loss_dict_sr05_m1e6[sim]
            l_curve  = l_length_dict_sr05_m1e6[sim]
            r_curve  = r_length_dict_sr05_m1e6[sim]
            final_df.loc[mask, col6]  = adjust_curve_length(ml_curve, T)
            final_df.loc[mask, col12] = adjust_curve_length(l_curve, T)
            final_df.loc[mask, col18] = adjust_curve_length(r_curve, T)



    output_csv_path = '/home/afacey/MPhysProj/MySims/FirstFinalPythonFiles/total_data_length_and_mass_all_final.csv'
    print(f'Saving file to {output_csv_path}')
    final_df.to_csv(output_csv_path, index=False)
    print('File saved.')
    return final_df

def get_mass_loss(folder_path, sim):
    params = sim.split('_')
    x = float(params[0].split('=')[1])
    y = float(params[1].split('=')[1])
    z = float(params[2].split('=')[1])
    vx = float(params[3].split('=')[1])
    vy = float(params[4].split('=')[1])
    vz = float(params[5].split('=')[1])
    sr = float(params[6].split('=')[1])
    mass = float(params[7].split('=')[1])
    positions = [x, y, z]
    velocities = [vx, vy, vz]
    if vz > 0:
        mass_loss_curve, norm_time, t_char = find_mass_loss_curves(folder_path, positions, velocities, sr, mass)
    else:
        mass_loss_curve, norm_time, t_char = np.arange(0, 200), np.arange(0, 200), np.ones(200)
    return mass_loss_curve, norm_time, t_char, sr, mass, positions, velocities

if __name__ == '__main__':
    # df = generate_dataframe_just_orbit()
    df = generate_dataframe_lengths_mass_loss_both()
    # print(f'The number of unique initial conditions is: {df["init_cond"].nunique()}')
    # print(df.head(10))
    # condition = 'x=32.6_y=0.0_z=0.0_vx=0.0_vy=128.0_vz=20.0'
    # filtered_df = df[df['init_cond'] == condition]
    # print(filtered_df)


    # base_path = '/home/afacey/MPhysProj/MySims/AutomatedClusterCreation/BigCollectionOfSims'
    # for folder in os.listdir(base_path):
    #     folder_path = os.path.join(base_path, folder)
    #     if os.path.isdir(folder_path):
    #         get_mass_loss(folder_path)
    # create_mass_loss_file(base_path)