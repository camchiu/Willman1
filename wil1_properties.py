import numpy as np

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

# distance
munoz_dist = 38                 # original distance estimate [kpc] (Martin2008, used in Munoz+2018)
dist = 38.55                    # distance [kpc]       (Durbin+2025)
dist_err = 0.45                 # distance error [kpc] (Durbin+2025, private communication)
dm_wil1 = 5 * np.log10(dist * 1000 / 10)                  # distance modulus
dm_wil1_err = 5 * (1 / (np.log(10) * dist)) * dist_err    # distance modulus error

r_half_pc = r_half * (np.pi / 180) * (dist * 10**3)                                          # half-light radius [pc]
r_half_pc_err = r_half_pc * (((r_half_err / r_half) ** 2 + (dist_err / dist) ** 2) ** (0.5)) # half-light radius error [pc]

# photometric properties
Mr_munoz = -2.74                # absolute r-band mag       (Munoz+2018)
Mr_err_munoz =  0.64            # absolute r-band mag error (Munoz+2018)

mr_munoz = Mr_munoz + 5 * np.log10(munoz_dist * 1000 / 10)      # apparent r-band mag
mr_err_munoz = Mr_err_munoz                                     # apparent r-band mag error

Mr = mr_munoz - dm_wil1         # recalculate absolute r-band mag using new distance estimate
Mr_err = (mr_err_munoz ** 2 + dm_wil1_err ** 2) ** (0.5)        # absolute r-band mag error

MV_munoz = -2.53                # absolute V-band mag (Munoz+2018)
MV_err_munoz = 0.74             # absolute V-band mag error (Munoz+2018)

mV_munoz = MV_munoz + 5 * np.log10(munoz_dist * 1000 / 10)      # apparent V-band mag
mV_err_munoz = MV_err_munoz                                     # apparent V-band mag error

MV = mV_munoz - dm_wil1         # recalculate absolute r-band mag using new distance estimate
MV_err = (mV_err_munoz ** 2 + dm_wil1_err ** 2) ** (0.5)        # absolute r-band mag error

v_avg2011 = -12.8
v_disp2011 = 4.8
pmra_pace = 0.255
pmra_err_pace = 0.08516454661418682
pmdec_pace = -1.11
pmdec_err_pace = 0.09580187889597991
pmra_w1 = [0.30350398138015605, 0.05087948259212696]
pmdec_w1 = [-1.0971768087861844, 0.07417568619696469]
feh_w1 = [-2.2351552416682106, 0.1684906636888024, 0.17065740903892346]
feh_disp_w1 = [0.3980117233077811, 0.1690246861562551, 0.12040491711922485]
MV_app = [15.37, 0.74]
feh_avg_2011 = -2.1
feh_disp_2011 = 0.45
w1_met = [-2.239071059256534, 0.144053895827831, 0.14640328935337932]
w1_met_sigma = [0.3592645331804941, 0.15528119660725137, 0.1086466172368904]
vsys = [-13.03994616473921, 1.1083444447390285]
vdisp = [4.688559339564653, 1.499989774645, 1.3102894662261315]
mass_w1 = [575442.5309510666, 325632.3536481163, 371696.69120681135]
