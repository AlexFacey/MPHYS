import numpy as np

G = 4.3e-6
M_halo  = 5.4e11
rs_halo = 15.62

M_disk  = 6.8e10
a_disk  = 3.0
b_disk  = 0.28

M_nucl  = 1.71e9
c_nucl  = 0.07

M_bulge = 5e9
c_bulge = 1.0

def HN_bulge_dphi_dr(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return - (G*M_bulge*r) / (r**2 + c_bulge**2)**2

def HN_nucl_dphi_dr(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return - (G*M_nucl*r) / (r**2 + c_nucl**2)**2

def NFW_dphi_dr(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return - (G*M_halo) / r * (np.log( 1 + (r/rs_halo) )/r - (1/(rs_halo + r)))

# Maybe needs a z? probs not
def MN_dphi_dR(x,y):
    r = np.sqrt(x**2 + y**2)
    n = 3.0/2.0
    return - (G*M_disk*r) / r**2

# this needs to be checked to see if it works in comparison to old
def calc_v_c(x, y, z):
    # ask if it makes sense to have y,z not = 0
    dPhidR_total = NFW_dphi_dr(x, y, z) + HN_nucl_dphi_dr(x, y, z) + HN_bulge_dphi_dr(x, y, z) + MN_dphi_dR(x, y)
    R = np.sqrt(x**2 + y**2 + z**2)
    V_c = np.sqrt(- R * dPhidR_total)
    return V_c

def calc_tidal_radius(x, y, z, M_clus):

    V_c = calc_v_c(x, y, z)
    R = np.sqrt(x**2 + y**2 + z**2)
    omega = V_c / R
    r_tid = (G*M_clus / omega**2)**(1/3)
    return r_tid

def normalise_time_to_gcr(positions, velocities):

    x, y, z = positions

    V_c = calc_v_c(x, y, z)

    L = np.cross(positions, velocities)
    L_mag = np.linalg.norm(L)

    R_g = L_mag / V_c

    t_char = R_g / V_c

    return t_char
    
