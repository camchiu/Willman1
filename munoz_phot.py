# import statements
import numpy as np
import pandas as pd
from astropy.io import ascii
import sfdmap
import astropy.units as u
import membership_utils

# read in munoz photometry
wil1_phot1 = ascii.read("data/munoz_final_W1.phot")

wil1_phot1.rename_column('col2', 'RA')
wil1_phot1.rename_column('col3', 'DEC')
wil1_phot1.rename_column('col4', 'g')
wil1_phot1.rename_column('col5', 'gerr')
wil1_phot1.rename_column('col6', 'r')
wil1_phot1.rename_column('col7', 'rerr')

wil1_phot = wil1_phot1.to_pandas(index = "col1")

# spatial properties
ra_w1  = 162.3436                 # center RA
dec_w1 = 51.0501                  # center DEC

alpha = np.radians(90-73)         # position angle
n = 0.47                          # ellipticity

r_half = 2.51/60                  # half-light radius [deg]
r_half_err = 0.22/60              # half-light radius error [deg]

ecc = (1 - ((n - 1) ** 2)) ** (1/2)                      # eccentricity
aax = r_half                                             # semi-major axis length [deg]
bax = (aax ** 2 - ((ecc ** 2) * (aax ** 2))) ** (1/2)    # semi-minor axis length [deg]

# delta ra cos dec, delta dec
wil1_phot["delta_ra_cos_dec"] = (wil1_phot["RA"] - ra_w1) * np.cos(np.deg2rad(dec_w1))
wil1_phot["delta_dec"] = (wil1_phot["DEC"] - dec_w1)

# empty columns
wil1_phot["radius"]         = np.nan      # distance to center of Wil1 [deg]
wil1_phot["radius_ell"]     = np.nan      # distance to center, calculated from elliptical radii [deg]
wil1_phot["hl_radius_circ"] = np.nan      # distance to center of Wil1 [half-light radii]
wil1_phot["hl_radius_ell"]  = np.nan      # half-light elliptical radius
wil1_phot["r_proj"]         = np.nan      # projected half-light radius
wil1_phot["pos_angle"]      = np.nan      # position angle (deg)

# preliminary spatial cut
wil1_phot = wil1_phot[(abs(wil1_phot["delta_ra_cos_dec"]) < 0.4) & ((abs(wil1_phot["delta_dec"]) < 0.4))]

# calculate spatial properties for each star
for index, row in wil1_phot.iterrows():
    radius, ell_radius, hl_radius_circular, hl_radius_elliptical, proj_distance, p_angle = \
        membership_utils.spatial_prop(row["delta_ra_cos_dec"], row["delta_dec"], 0, 0, r_half, -alpha, aax, ecc)
    
    wil1_phot.loc[index, "radius"]         = radius
    wil1_phot.loc[index, "radius_ell"]     = ell_radius
    wil1_phot.loc[index, "hl_radius_circ"] = hl_radius_circular
    wil1_phot.loc[index, "hl_radius_ell"]  = hl_radius_elliptical
    wil1_phot.loc[index, "r_proj"]         = proj_distance
    wil1_phot.loc[index, "pos_angle"]      = p_angle

# correct for extinction
sdf_ext = sfdmap.SFDMap('data/sfddata-master')
wil1_phot["EBV"] = sdf_ext.ebv(wil1_phot['RA'] * u.deg, wil1_phot['DEC'] * u.deg)

Ag = 3.237 * wil1_phot['EBV']
Ar = 2.176 * wil1_phot['EBV']
wil1_phot['rmag_o'] = wil1_phot['r'] - Ar
wil1_phot['gmag_o'] = wil1_phot['g'] - Ag

# open isochrone
iso_data = pd.read_csv("data/isochrone_data_0001.csv")

# definition of P_CMD
def prob_cmd(min_dist, data_sigma, iso_sigma):
    p_cmd = np.exp(-(min_dist ** 2) / (2 * (iso_sigma ** 2 + data_sigma ** 2)))
    return p_cmd

# empty columns
wil1_phot["perp_dist_iso"]  = np.nan      # perpendicular distance to isochrone
wil1_phot["cmd_marker"]     = np.nan      # cmd marker

# cmd marker
for index_phot, row_phot in wil1_phot.iterrows():
    if row_phot["hl_radius_ell"] <= 6:
        # color, magnitude data
        r_mag = wil1_phot.loc[index_phot, "rmag_o"]
        gr_mag = wil1_phot.loc[index_phot, "gmag_o"] - r_mag

        # add error data in quadrature
        r_mag_err = wil1_phot.loc[index_phot, "rerr"]
        g_mag_err = wil1_phot.loc[index_phot, "gerr"]
        gr_mag_err = (r_mag_err ** 2 + g_mag_err ** 2) ** (0.5)
    
        # calculate distance from star's cmd location to each point in isochrone
        iso_dists = np.sqrt((r_mag - iso_data["r"]) ** 2 + (gr_mag - iso_data["gr"])**2)
        d_min = np.min(iso_dists)
        wil1_phot.loc[index_phot, "perp_dist_iso"] = d_min
    
        # set isochrone error, to account for the metallicity spread + age uncertainty (estimated using grid of isochrones)
        iso_stdev = 0.15
    
        if d_min < 4 * iso_stdev: # calculate P_CMD from perpendicular distance
            prob = prob_cmd(d_min, gr_mag_err, iso_stdev)
            wil1_phot.loc[index_phot, "cmd_marker"] = prob
    
        else: # P_CMD = 0 if more than 4 * iso_stdev away from isochrone
            wil1_phot.loc[index_phot, "cmd_marker"] = 0
wil1_phot.loc[37380, "cmd_marker"] = 1 # manually add one HB star

# save to csv file
wil1_phot.to_csv("data/munoz_wil1_phot_evaluated.csv")