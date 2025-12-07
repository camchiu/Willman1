# import necessary packages
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

import multiprocessing

import astropy.table as table
from astropy import units as u
from astropy.coordinates import SkyCoord

from galpy.orbit import Orbit
from galpy.potential import ChandrasekharDynamicalFrictionForce
from galpy.potential import HernquistPotential, MovingObjectPotential, NonInertialFrameForce
from galpy.potential import (evaluateRforces, evaluatephitorques, evaluatezforces)

print("imports done")

# open milky way dwarf galaxies data
dsph_mw = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/dwarf_mw.csv')
mw_dwarfs = dsph_mw.to_pandas()

mw_dwarfs["peri"], mw_dwarfs["peri16"], mw_dwarfs["peri50"], mw_dwarfs["peri84"] = np.nan, np.nan, np.nan, np.nan
mw_dwarfs["apo"], mw_dwarfs["apo16"], mw_dwarfs["apo50"], mw_dwarfs["apo84"] = np.nan, np.nan, np.nan, np.nan
mw_dwarfs["ecc"], mw_dwarfs["ecc16"], mw_dwarfs["ecc50"], mw_dwarfs["ecc84"] = np.nan, np.nan, np.nan, np.nan
mw_dwarfs["fperi"], mw_dwarfs["fperi16"], mw_dwarfs["fperi50"], mw_dwarfs["fperi84"] = np.nan, np.nan, np.nan, np.nan
mw_dwarfs["Lz"], mw_dwarfs["Lz16"], mw_dwarfs["Lz50"], mw_dwarfs["Lz84"] = np.nan, np.nan, np.nan, np.nan
mw_dwarfs["E"], mw_dwarfs["E16"], mw_dwarfs["E50"], mw_dwarfs["E84"] = np.nan, np.nan, np.nan, np.nan

# open milky way globular clusters data
gc_disk = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/gc_disk.csv')
mw_gcs1 = gc_disk.to_pandas()

gc_harris = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/gc_harris.csv')
mw_gcs2 = gc_harris.to_pandas()

gc_ufsc = table.Table.read('https://raw.githubusercontent.com/apace7/local_volume_database/main/data/gc_ufsc.csv')
mw_gcs3 = gc_ufsc.to_pandas()

mw_clusters = pd.concat([mw_gcs1, mw_gcs2, mw_gcs3], ignore_index = True)

mw_clusters["peri"], mw_clusters["peri16"], mw_clusters["peri50"], mw_clusters["peri84"] = np.nan, np.nan, np.nan, np.nan
mw_clusters["apo"], mw_clusters["apo16"], mw_clusters["apo50"], mw_clusters["apo84"] = np.nan, np.nan, np.nan, np.nan
mw_clusters["ecc"], mw_clusters["ecc16"], mw_clusters["ecc50"], mw_clusters["ecc84"] = np.nan, np.nan, np.nan, np.nan
mw_clusters["fperi"], mw_clusters["fperi16"], mw_clusters["fperi50"], mw_clusters["fperi84"] = np.nan, np.nan, np.nan, np.nan
mw_clusters["Lz"], mw_clusters["Lz16"], mw_clusters["Lz50"], mw_clusters["Lz84"] = np.nan, np.nan, np.nan, np.nan
mw_clusters["E"], mw_clusters["E16"], mw_clusters["E50"], mw_clusters["E84"] = np.nan, np.nan, np.nan, np.nan

print("data open")

# MW Potential with LMC
from galpy.potential import MWPotential2014

# calculate LMC orbit
o_lmc = Orbit.from_name('LMC')
MWPotential2014[2] *= 1.5 # halo 50% bigger so LMC is in bound orbit

lmc_mass = 1.38 * 10.**11 * u.Msun # total mass
lmc_rhalf = 8.7 * u.kpc # r_half
cdf_lmc = ChandrasekharDynamicalFrictionForce(GMs = lmc_mass, rhm = lmc_rhalf, dens = MWPotential2014)

time_step_lmc = np.linspace(0., -10., 1001)*u.Gyr
o_lmc.integrate(time_step_lmc, MWPotential2014 + cdf_lmc, method = 'dop853_c')

# model LMC potential
lmcpot = HernquistPotential(amp = 2 * lmc_mass, a = lmc_rhalf/(1. + np.sqrt(2.))) #rhm = (1+sqrt(2)) a
moving_lmcpot = MovingObjectPotential(o_lmc, pot = lmcpot)

# calculate acceleration due to LMC at the origin
loc_origin = 1e-4 # Small offset in R to avoid numerical issues
ax = lambda t: evaluateRforces(moving_lmcpot, loc_origin, 0., phi = 0., t = t, use_physical = False)
ay = lambda t: evaluatephitorques(moving_lmcpot, loc_origin, 0., phi = 0., t = t, use_physical = False) / loc_origin
az = lambda t: evaluatezforces(moving_lmcpot, loc_origin, 0., phi = 0., t = t, use_physical = False)

# interpolate
if o_lmc.time(use_physical = False)[0] > o_lmc.time(use_physical = False)[1]:
    t_intunits = o_lmc.time(use_physical = False)[::-1] # need to reverse the order for interp
else:
    t_intunits = o_lmc.time(use_physical = False)

ax4int = np.array([ax(t) for t in t_intunits])
ax_int = lambda t: np.interp(t, t_intunits, ax4int)

ay4int = np.array([ay(t) for t in t_intunits])
ay_int = lambda t: np.interp(t, t_intunits, ay4int)

az4int = np.array([az(t) for t in t_intunits])
az_int = lambda t: np.interp(t, t_intunits, az4int)

# set up non-inertial reference frame
nip = NonInertialFrameForce(a0 = [ax_int, ay_int, az_int])

print("potential set up")

# run single orbit
def orbit_params(row, num = 1000):
    # 6D phase space measurements
    ra, dec = row["ra"], row["dec"]
    dist = row["distance"]
    dist_err = (row["distance_ep"] + row["distance_em"]) / 2
    pmra, pmdec = row["pmra"], row["pmdec"]
    pmra_err = (row["pmra_ep"] + row["pmra_em"]) / 2
    pmdec_err = (row["pmdec_ep"] + row["pmdec_em"]) / 2
    vel = row["vlos_systemic"]
    vel_err = (row["vlos_systemic_ep"] + row["vlos_systemic_em"]) / 2

    if np.isnan(dist_err):
        dist_err = 0
    if np.isnan(pmra_err):
        pmra_err, pmdec_err = 0, 0
    if np.isnan(vel_err):
        vel_err = 0

    # create samples
    ras = np.full((num), ra)
    decs = np.full((num), dec)
    dists = scipy.stats.norm.rvs(loc = dist, scale = dist_err, size = num) #normal dist
    pmras = scipy.stats.norm.rvs(loc = pmra, scale = pmra_err, size = num) #normal dist
    pmdecs = scipy.stats.norm.rvs(loc = pmdec, scale = pmdec_err, size = num) #normal dist
    vels = scipy.stats.norm.rvs(loc = vel, scale = vel_err, size = num) #normal dist

    # create coordinates
    coords = SkyCoord(ra = ra * u.degree, dec = dec * u.degree, distance = dist * u.kpc,
                      pm_ra_cosdec = pmra * u.mas/u.yr, pm_dec = pmdec * u.mas/u.yr, radial_velocity = vel * u.km/u.s)
    samp_coords = SkyCoord(ra = ras * u.degree, dec = decs * u.degree, distance = dists * u.kpc,
                           pm_ra_cosdec = pmras * u.mas/u.yr, pm_dec = pmdecs * u.mas/u.yr, radial_velocity = vels * u.km/u.s)
    
    # calculate orbit
    obj_orbit = Orbit(coords)
    orbit_samps = Orbit(samp_coords)
    
    # define time step
    t_step = np.linspace(0, -10, 1001) * u.Gyr
    t_step2 = np.linspace(0, -1, 101) * u.Gyr

    # integrate orbit
    obj_orbit.integrate(t_step, MWPotential2014) # + nip + moving_lmcpot
    orbit_samps.integrate(t_step2, MWPotential2014) # + nip + moving_lmcpot
    
    # orbital parameters
    pericenter, apocenter, eccentricity = obj_orbit.rperi(), obj_orbit.rap(), obj_orbit.e()
    fperi_obj = (obj_orbit.r() - obj_orbit.rperi()) / (obj_orbit.rap() - obj_orbit.rperi())
    Lz, E = (-obj_orbit.Lz()  * (u.km / u.s) * u.kpc).to(u.kpc**2 / u.Myr), (obj_orbit.E() * (u.km/u.s)**2).to(u.kpc**2 / u.Myr **2)
    
    pericenters, apocenters, eccentricities = orbit_samps.rperi(), orbit_samps.rap(), orbit_samps.e()
    
    peri16, peri50, peri84 = np.percentile(pericenters, 16), np.percentile(pericenters, 50), np.percentile(pericenters, 84)
    apo16, apo50, apo84 = np.percentile(apocenters, 16), np.percentile(apocenters, 50), np.percentile(apocenters, 84)
    ecc16, ecc50, ecc84 = np.percentile(eccentricities, 16), np.percentile(eccentricities, 50), np.percentile(eccentricities, 84)
    
    fperi = (orbit_samps.r() - orbit_samps.rperi()) / (orbit_samps.rap() - orbit_samps.rperi())
    Lz_samp = (- orbit_samps.Lz() * (u.km / u.s) * u.kpc).to(u.kpc**2 / u.Myr)
    E_samp = (orbit_samps.E() * (u.km/u.s)**2).to(u.kpc**2 / u.Myr **2)

    fperi16, fperi50, fperi84 = np.percentile(fperi, 16), np.percentile(fperi, 50), np.percentile(fperi, 84)
    Lz16, Lz50, Lz84 = np.percentile(Lz_samp.value, 16), np.percentile(Lz_samp.value, 50), np.percentile(Lz_samp.value, 84)
    E16, E50, E84 = np.percentile(E_samp.value, 16), np.percentile(E_samp.value, 50), np.percentile(E_samp.value, 84)
    
    params = [pericenter, peri16, peri50, peri84,
              apocenter, apo16, apo50, apo84, 
              eccentricity, ecc16, ecc50, ecc84,
              fperi_obj, fperi16, fperi50, fperi84, 
              Lz.value, Lz16, Lz50, Lz84, 
              E.value, E16, E50, E84]

    # save to row df
    row["peri"], row["peri16"], row["peri50"], row["peri84"] = params[0], params[1], params[2], params[3]
    row["apo"], row["apo16"], row["apo50"], row["apo84"] = params[4], params[5], params[6], params[7]
    row["ecc"], row["ecc16"], row["ecc50"], row["ecc84"] = params[8], params[9], params[10], params[11]
    row["fperi"], row["fperi16"], row["fperi50"], row["fperi84"] = params[12], params[13], params[14], params[15]
    row["Lz"], row["Lz16"], row["Lz50"], row["Lz84"] = params[16], params[17], params[18], params[19]
    row["E"], row["E16"], row["E50"], row["E84"] = params[20], params[21], params[22], params[23]

    print(row["name"])

    return row

# run orbit for chunks
def loop_orbit(chunk):
    for index, row in chunk.iterrows():
        orbit_row = orbit_params(row)
        chunk.loc[index] = orbit_row
    return chunk

# orbits using parallel processing
def parallel_process(dataframe, num_processes = 36):
    # split df into chunks for parallel processing
    chunks = np.array_split(dataframe, num_processes)
    
    # create a Pool of worker processes
    with multiprocessing.Pool() as pool:
        # calculate orbits of each chunk in parallel
        results = pool.map(loop_orbit, chunks)
        
    # concatenate results and return final df
    orbit_df = pd.concat(results)
    return orbit_df

# run multiprocessing and save files
if __name__ == "__main__":
    mw_dwarfs_orbits = parallel_process(dataframe = mw_dwarfs)
    mw_dwarfs_orbits.to_csv("evaluated_mwdg.csv")

    mw_clusters_orbits = parallel_process(dataframe = mw_clusters)
    mw_clusters_orbits.to_csv("evaluated_mwgc.csv")