import shutil
import os
import subprocess
from make_model_file import plummer_density, makemodel
import numpy as np
import yaml
from rotate_initial_condtions import rotate_coordinates
import sys
import argparse

def create_folder(new_directory):
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    else:
        raise SystemExit(f'Stopping script because the directory {new_directory} already exists.')
    return new_directory

def create_yaml_file(offset, new_directory, scale_radius, rvals):
    
    if scale_radius == 0.01:
        nsteps = 500000
        dtime = 1.5e-8
        nint = 2500
    else:
        nsteps = 50000
        dtime = 1.5e-7
        nint = 250

    data = {
        'Global': {
            'outdir': '.',
            'ldlibdir': '/home/mpetersen/lib/user/',
            'runtag': 'run1',
            'nsteps': nsteps,
            'dtime': dtime,
            'multistep': 0,
            'dynfracA': 0.01,
            'dynfracV': 0.000625,
            'VERBOSE': 1,
        },
        'Components': [
            {
                'name': 'cluster',
                'parameters': {
                    'nlevel': 1, 'indexing': True, 'EJ': 2, 'EJdryrun': False, 
                    'EJkinE': False, 'EJdiag': True, 'nEJkeep': 1, 'nEJwant': 100,
                    'EJx0': offset[0], 'EJy0': offset[1], 'EJz0': offset[2], 
                    'EJu0': offset[3], 'EJv0': offset[4], 'EJw0': offset[5]
                },
                'bodyfile': 'automated_cluster.bods',
                'force': {
                    'id': 'sphereSL',
                    'parameters': {
                        'numr': int(len(rvals)),
                        'rmin': float(min(rvals)),
                        'rmax': float(max(rvals)),
                        'Lmax': 2,
                        'nmax': 12,
                        'rmapping': float(scale_radius),
                        'self_consistent': True,
                        'modelname': 'automated_cluster.model',
                        'cachename': 'automated_cluster.cache'
                    }
                }
            }
        ],
        'Output': [
            {'id': 'outlog', 'parameters': {'nint': 1}},
            {'id': 'outpsn', 'parameters': {'nint': nint}},
            {'id': 'outcoef', 'parameters': {'nint': 1, 'name': 'cluster'}}
        ],
        'External': [
            {'id': 'usermw', 'parameters': {'G': 1.0}}
        ],
        'Interaction': {}
    }
    #'EJkinE': True
    # Ensure the directory exists
    os.makedirs(new_directory, exist_ok=True)
    file_path = os.path.join(new_directory, 'automated_config.yml')

    with open(file_path, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

    with open(file_path, 'r') as file:
        content = file.read()

    # Replace the 'parameters' section in 'Components' to be on a single line with quotes around each key
    content = content.replace(
        "parameters:\n      nlevel: 1\n      indexing: true\n      EJ: 2\n      EJdryrun: false\n      EJkinE: true\n      EJdiag: true\n      nEJkeep: 1\n      nEJwant: 1000\n      EJx0: {}\n      EJy0: {}\n      EJz0: {}\n      EJu0: {}\n      EJv0: {}\n      EJw0: {}".format(
            offset[0], offset[1], offset[2], offset[3], offset[4], offset[5]
        ),
        "parameters: {'nlevel': 1, 'indexing': true, 'EJ': 2, 'EJdryrun': false, 'EJkinE': true, 'EJdiag': true, 'nEJkeep': 1, 'nEJwant': 1000, 'EJx0': " + str(offset[0]) + ", 'EJy0': " + str(offset[1]) + ", 'EJz0': " + str(offset[2]) + ", 'EJu0': " + str(offset[3]) + ", 'EJv0': " + str(offset[4]) + ", 'EJw0': " + str(offset[5]) + "}"
    )

    with open(file_path, 'w') as file:
        file.write(content)

def create_bods_file(n_stars, mass, offset, scale_radius, filename, new_directory):
    
    original_dir = os.getcwd()

    try:
        os.chdir('/home/afacey/MPhysProj/MySims/PlummerPlus')
        command = ['python3', 'PlummerPlus.py', '-n', str(n_stars), '-M', str(mass), '-init'] + [str(x) for x in offset] + ['-o', filename, '-scale', str(scale_radius)]
        print(f'Running command: {" ".join(command)}')  # Print the command
        subprocess.run(command, check=True)
        new_folder_path = os.path.join(original_dir, new_directory)
        shutil.move(filename, new_folder_path)

        file_path = os.path.join(new_folder_path, filename)
        print(f'This is the file path {file_path}')
        with open(file_path, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write(f'{n_stars}     0     0\n' + content)
    finally:
        os.chdir(original_dir)

def create_model_file(scale_radius, mass, new_directory, rvals):
    new_directory = os.path.join(os.getcwd(), new_directory)
    os.chdir(new_directory)
    pfile = 'automated_cluster.model'
    R,D,M,P = makemodel(plummer_density,1.,[scale_radius, mass],rvals = rvals, pfile=pfile, plabel=pfile)

def convert_to_g_1_coords(coords):
    scale = np.sqrt(1/4.3e-6)
    initial_conditions = [coords[0], coords[1], coords[2], \
                          float(coords[3] * scale), float(coords[4] * scale), float(coords[5] * scale)]
    return initial_conditions

if __name__ == '__main__':

    # Check for correct number of arguments
    if len(sys.argv) < 2:
        raise SystemExit(f'Usage: {sys.argv[0]} -f <new_directory> -o <offsets> -s <scale_radius> -n <n_stars> -m <mass>')

    parser = argparse.ArgumentParser(description='Create a cluster in MW.')
    parser.add_argument('-f', '--new_directory', type=str, required=True, help='The name of the new directory to create.')
    parser.add_argument('-o', '--offset', nargs=6, type=float, required=True, help='Offsets: x y z vx vy vz')
    parser.add_argument('-s', '--scale_radius', type=float, default=0.5, help='The scale radius of the cluster.')
    parser.add_argument('-n', '--n_stars', type=int, default=10000, help='The number of stars in the cluster.')
    parser.add_argument('-m', '--mass', type=float, default=10e5, help='The mass of the cluster.')

    args = parser.parse_args()

    new_directory = args.new_directory
    offset = args.offset
    n_stars = args.n_stars
    mass = args.mass
    scale_radius = args.scale_radius
    # rvals = 10.**np.linspace(-5.,scale_radius / 10,2000)
    rvals = 10.**np.linspace(np.log10(scale_radius)-4.,np.log10(scale_radius)+3,4000)
    print(rvals[-1])
    offset = convert_to_g_1_coords(offset)
    filename = 'automated_cluster.bods'
    base_directory = os.getcwd()

    folder = create_folder(new_directory)
    create_yaml_file(offset, new_directory, scale_radius, rvals)
    create_bods_file(n_stars, mass, offset, scale_radius, filename, new_directory)
    create_model_file(scale_radius, mass, new_directory, rvals)
    os.chdir(base_directory)


    # new_directory_list = ['30kpc_0deg', '30kpc_15deg', '30kpc_30deg', '30kpc_45deg', '30kpc_60deg', '30kpc_90deg_220']
    # offset_angles = [0, 15, 30, 45, 60, 90]
    # offset_init = [30, 0, 0 , 0, 220, 0]
    # n_stars = 10000
    # mass = 10e5
    # scale_radius = 0.5
    # base_directory = os.getcwd()

    # new_directory = new_directory_list[-1]
    # offset_angle = offset_angles[-1]
    # offset = rotate_coordinates(offset_init, offset_angle)
    # offset = convert_to_g_1_coords(offset)
    # print(offset)
    # filename = 'automated_cluster.bods'
    # base_directory = os.getcwd()

    # folder = create_folder(new_directory)
    # create_yaml_file(offset, new_directory, scale_radius)
    # create_bods_file(n_stars, mass, offset, filename, new_directory)
    # create_model_file(scale_radius, mass, new_directory)
    # os.chdir(base_directory)

    # for new_directory, offset_angle in zip(new_directory_list, offset_angles):
    #     offset = rotate_coordinates(offset_init, offset_angle)
    #     offset = convert_to_g_1_coords(offset)
    #     print(offset)
    #     filename = 'automated_cluster.bods'
    #     base_directory = os.getcwd()

    #     folder = create_folder(new_directory)
    #     create_yaml_file(offset, new_directory, scale_radius,)
    #     create_bods_file(n_stars, mass, offset, filename, new_directory)
    #     create_model_file(scale_radius, mass, new_directory)
    #     os.chdir(base_directory)























    # new_directory = '30kpc'

    # n_stars = 10000
    # mass = 10e5
    # scale_radius = 0.5



    # offset = convert_to_g_1_coords([30, 0, 0 , 0, 150, 0])
    # filename = 'automated_cluster.bods'
    # base_directory = os.getcwd()

    # folder = create_folder(new_directory)
    # create_yaml_file(offset, new_directory, scale_radius,)
    # create_bods_file(n_stars, mass, offset, filename, new_directory)
    # create_model_file(scale_radius, mass, new_directory)
        
    # offset = [7.274214773276849, 0.2236482173152161, 15.804014889594473, -20149.033537170348, -43769.2854289064, -7773.638532340813]