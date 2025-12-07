# import statements
import numpy as np
import math as m
from sympy import symbols, solve, Eq

# spatial properties
def spatial_prop(RA, DEC, u, v, r_half, alpha, a, e):
    """ Determine the spatial properties of a star in a dwarf galaxy given its 
    center RA/DEC, half-light radius, position angle, semi-major axis, and eccentricity.
    Parameters
    ----------
        RA (float) : right ascension of star [deg]
        DEC (float) : declination of star [deg]
        u (float) : center RA of dwarf galaxy [deg]
        v (float) : center DEC of dwarf galaxy [deg]
        r_half (float) : half-light radius of dwarf galaxy [deg]
        alpha (float) : position angle of dwarf galaxy [deg]
                        (defined as clockwise angle of semimajor axis from the vertical)
        a (float) : semi-major axis [deg]
        e (float) : eccentricity (0 to 1)
    Returns
    -------
        r (float) : distance to center of dwarf galaxy [deg]
        elliptical_r (float) : distance to center of dwarf galaxy recalculated from elliptical radii values [deg]
                                    (check to verify -- should approximately equal r)
        circ_hl_radius (float) : circular half-light radius [r_half]
        ell_hl_radius (float) : elliptical half-light radius [r_half]
        proj_dist (float) : projected major-axis distance [r_half]
        pos_ang_deg (float) : position angle of star with respect to the semi-major axis (ccw) [deg]
    """
    # set origin at center RA/DEC of dwarf galaxy
    x = RA - u
    y = DEC - v
        
    # convert to polar
    r = (x ** 2 + y ** 2) ** (1/2)
    theta = np.arctan2(y, x)
    circ_hl_radius = r / r_half # radius in half-light radii

    # rotate based on position angle (alpha)
    x_rot = r * np.cos(theta + alpha)
    y_rot = r * np.sin(theta + alpha)
    
    # determine position angle wrt semi-major axis
    pos_ang_rad = theta + alpha
    pos_ang_deg = m.degrees(pos_ang_rad)
    if pos_ang_rad < 0:
        pos_ang_deg += 360
    
    # calculate projected major−axis distance from center
    proj_dist = x_rot / r_half

    # calculate semi-major, minor axis of ellipse containing point
    x = symbols('x')
    equation = Eq((x_rot / x) ** 2 + (y_rot / ((x ** 2 - ((e ** 2) * (x ** 2))) ** (1/2))) ** 2, 1)
    solution = solve(equation, x)[0]

    major = abs(solution)
    minor = (major ** 2 - ((e ** 2) * (major ** 2))) ** (1/2)
    
    # calculate half-light radius and elliptical radius (should approximately equal r)
    ell_hl_radius = float(major / a)
    elliptical_r = float((major * minor) / (((major ** 2 * np.sin(theta + alpha) ** 2) + (minor ** 2 * np.cos(theta + alpha) ** 2)) ** (1/2)))
    
    # return values
    return r, elliptical_r, circ_hl_radius, ell_hl_radius, proj_dist, pos_ang_deg

# save variables
def save_variable(name, value, file_path = "wil1_properties.py"):
    lines = []
    found = False
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        pass

    with open(file_path, "w") as f:
        for line in lines:
            if line.strip().startswith(f"{name} ="):
                f.write(f"{name} = {repr(value)}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"{name} = {repr(value)}\n")