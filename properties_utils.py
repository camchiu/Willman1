# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm
from scipy.special import logsumexp
import astropy.units as u
import astropy.constants as const
import pandas as pd

import emcee
import corner

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

######################################################
############# velocity/dispersion MCMC ###############
######################################################

# likelihood probability
def lnprob_v(theta, vel, vel_err):
    # "v" or "v_zero"
 
    lp = lnprior_v(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_v(theta, vel, vel_err)

# likelihood priors
def lnprior_v(theta):
    if (-40. < theta[0] < 10.) & (0. < theta[1] < 10.):
        return 0.0
    
    return -np.inf

# likelihood probability
def lnlike_v(theta, vel, vel_err):
    # Gaussian 
    term1 = -0.5*np.sum(np.log(vel_err**2 + theta[1]**2))
    term2 = -0.5*np.sum((vel - theta[0])**2/(vel_err**2 + theta[1]**2))
    term3 = -0.5*np.size(vel)* np.log(2*np.pi)

    lnl  = term1+term2+term3
    
    return lnl

# initialize walkers
def initialize_walkers_v(vr_guess, sig_guess):

    # v, sig
    ndim, nwalkers = 2, 20
    p0 =  np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    # vr 
    p0[:,0] = (p0[:,0] * 40. - 20.) + vr_guess
    
    # initial walker guesses
    p0[:,1] = (p0[:,1] * 6. - 3.) + sig_guess

    return ndim,nwalkers,p0

# run sampler
def run_sampler_v(sampler, p0, max_n):

    pos, prob, state = sampler.run_mcmc(p0, max_n,progress=False)
    
    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100
        
    return sampler, convg, burnin

# MCMC
def mcmc_velocity(vel, vel_err, vr_guess, sig_guess, label, figname, max_n):
    ndim, nwalkers, p0  = initialize_walkers_v(vr_guess,sig_guess)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_v, args=(vel, vel_err))
    sampler, convg, burnin = run_sampler_v(sampler, p0, max_n)
    theta = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]

    # PLOT CORNER
    labels = [r'$v_{W1}$', r'$\sigma_{v}$']
    ndim=2
    samples = sampler.chain[:, 2*burnin:, :].reshape((-1, ndim))
    fig     = corner.corner(samples, labels = labels, show_titles = True, quantiles=[0.16, 0.5, 0.84],
                               label_kwargs = {"labelpad": 5, "size": 18})
    fig.gca().annotate(label, xy=(1.0, 1.0), xycoords="figure fraction", xytext=(-20, -10), 
                           textcoords="offset points", ha="right", va="top", fontsize = 14)
    if figname != 0:
        fig.savefig(figname, dpi = 600, transparent = True)
    
    return sampler, theta

# determine velocity + velocity disp
def calc_velocity(vel,vel_err, vr_guess, sig_guess, label, figname, nn = 5000):

    sampler, theta = mcmc_velocity(vel, vel_err, vr_guess, sig_guess, label, figname, max_n = nn)

    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        print(tau,burnin, converged)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100

    if (burnin < 100):
        burnin = 100
    print(burnin, convg)

    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            vr = mcmc[1] 
            vr_err16 = mcmc[0]
            vr_err84= mcmc[2] 
            vr_err   =(vr_err84 - vr_err16)/2.   # APPROX
        if (ii==1):
            sigma = mcmc[1] 
            sigma_err16 = mcmc[0]
            sigma_err84 = mcmc[2]

            sigma_err_m   = (sigma - sigma_err16)
            sigma_err_p   = (sigma_err84 - sigma)

    return vr, vr_err, sigma, sigma_err_p,sigma_err_m, sampler

# weighted rolling average
def weighted_rolling_average(radius, velocity, weights, window_size = 7):
    # calculate length of rolling avg df
    n_rolling = len(radius) - (window_size - 1)

    # empty arrays for rolling radius, velocity
    rolling_radius = np.zeros(n_rolling)
    rolling_velocity = np.zeros(n_rolling)

    for ii in range(n_rolling):
        radius_window = radius[ii:ii+window_size]
        velocity_window = velocity[ii:ii+window_size]

        weight_window = weights[ii:ii+window_size]

        rolling_radius[ii] = np.sum(radius_window * weight_window) / np.sum(weight_window)
        rolling_velocity[ii] = np.sum(velocity_window * weight_window) / np.sum(weight_window)
    
    return rolling_radius, rolling_velocity

# weighted rolling average bootstrap
def weighted_rolling_bootstrap(n_boot, window_size, array, r_col = "hl_radius_ell"):
    # calculate regular rolling avg
    sorted_secure_members = array.sort_values(r_col)

    radius = sorted_secure_members[r_col]
    velocity = sorted_secure_members["v"]
    weights = sorted_secure_members["prob_member"]

    r, v = weighted_rolling_average(radius, velocity, weights)

    # df to store velocity bootstrapping samples
    vel_samp_df = pd.DataFrame()

    # bootstrapping
    for nn in range(n_boot):
        sample = array.sample(len(array), replace = True)

        sorted_sample = sample.sort_values(r_col)

        samp_radius = sorted_sample[r_col]
        samp_velocity = sorted_sample["v"]
        samp_weights = sorted_sample["prob_member"]

        rolling_radius, rolling_velocity = weighted_rolling_average(samp_radius, samp_velocity, samp_weights, window_size)
        rolling_velocity_interp = np.interp(r, rolling_radius, rolling_velocity)

        new_df = pd.DataFrame(rolling_velocity_interp, columns = [nn]).T
        vel_samp_df = pd.concat([vel_samp_df, new_df])

    vel, vel_lower, vel_upper, vel_stdev = np.array([]), np.array([]), np.array([]), np.array([])

    for ii in vel_samp_df.columns.tolist():
        item_mean = np.median(vel_samp_df.loc[:, ii])
        item_lower = np.percentile(vel_samp_df.loc[:, ii], 16)
        item_upper = np.percentile(vel_samp_df.loc[:, ii], 84)
        item_stdev = np.std(vel_samp_df.loc[:, ii])

        vel = np.append(vel, item_mean)
        vel_lower = np.append(vel_lower, item_lower)
        vel_upper = np.append(vel_upper, item_upper)
        vel_stdev = np.append(vel_stdev, item_stdev)
    
    return r, v, vel, vel_lower, vel_upper, vel_stdev

#######################################
######## TWO-GAUSSIAN MCMC ############
#######################################

def lnprior_mix(theta):
    mu1, sig1, mu2, sig2, f = theta
    if not (0 < f < 1 and 0 < sig1 < 200 and 0 < sig2 < 10 and -100 < mu1 < 100 and -30 < mu2 < 10):
        return -np.inf

    return 0.0

def lnlike_mix(theta, vel, vel_err):
    mu1, sig1, mu2, sig2, f = theta

    var1 = sig1**2 + vel_err**2
    var2 = sig2**2 + vel_err**2

    # Log PDFs for both components
    lnL1 = -0.5 * ((vel - mu1)**2 / (var1) + np.log(2 * np.pi * (var1)))
    lnL2 = -0.5 * ((vel - mu2)**2 / (var2) + np.log(2 * np.pi * (var2)))

    # Combine in linear space via log-sum-exp
    lnL = np.sum(logsumexp([np.log(1-f) + lnL1, np.log(f) + lnL2], axis=0))
    return lnL

def lnprob_mix(theta, vel, vel_err):
    lp = lnprior_mix(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_mix(theta, vel, vel_err)

def initialize_walkers_mix(priors, nwalkers=50):
    ndim = 5
    p0 = np.zeros((nwalkers, ndim))
    for i, key in enumerate(['mu1', 'sig1', 'mu2', 'sig2', 'f']):
        mu, sigma = priors[key]
        p0[:, i] = mu + sigma * np.random.randn(nwalkers)
    return ndim, nwalkers, p0

def mcmc_mixture(vel, vel_err, priors, nsteps=5000, boot=False):
    ndim, nwalkers, p0 = initialize_walkers_mix(priors)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_mix, args=(vel, vel_err))
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Compute burn-in automatically
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
    except:
        burnin = 200

    for ii in [2, 3]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==2):
            vr = mcmc[1] 
            vr_err16 = mcmc[0]
            vr_err84 = mcmc[2] 
            vr_err   = (vr_err84 - vr_err16)/2.   # APPROX
        if (ii==3):
            sigma = mcmc[1] 
            sigma_err16 = mcmc[0]
            sigma_err84 = mcmc[2]

            sigma_err_m   = (sigma - sigma_err16)
            sigma_err_p   = (sigma_err84 - sigma)

    samples = sampler.get_chain(discard = burnin, flat = True)
    return sampler, samples, vr, vr_err, sigma, sigma_err_p,sigma_err_m

def plot_mixture_results(samples, vel, label = None, figname = 0):
    labels = [r"$v_{\rm MW}$", r"$\sigma_{\rm MW}$", r"$v_{\rm W1}$", r"$\sigma_{\rm W1}$", r"$f$"]
    fig = corner.corner(samples, labels = labels, show_titles = True, quantiles=[0.16, 0.5, 0.84],
                               label_kwargs = {"labelpad": 5, "size": 28}, title_kwargs = {"fontsize": 16})
    if label != None:
        fig.gca().annotate(label, xy=(1.0, 1.0), xycoords="figure fraction", xytext=(-20, -10), 
                           textcoords="offset points", ha="right", va="top", fontsize = 14)
    plt.show()
    if figname != 0:
        fig.savefig(figname, dpi = 600, transparent = True)

    # posterior medians
    theta_best = np.median(samples, axis=0)
    print("Posterior medians:", dict(zip(labels, theta_best)))

    # plot fit
    mu1, sig1, mu2, sig2, f = theta_best
    vgrid = np.linspace(np.min(vel), np.max(vel), 300)
    pdf1 = (1-f) * norm.pdf(vgrid, mu1, sig1) 
    pdf2 = f * norm.pdf(vgrid, mu2, sig2)

    plt.hist(vel, bins=25, density=True, alpha=0.5, label="Data")
    plt.plot(vgrid, pdf1, 'r-', lw=2, label="MW bkgd")
    plt.plot(vgrid, pdf2, 'b-', lw=2, label="W1")
    plt.xlim(-200, 200)
    plt.legend()
    plt.show()

    return theta_best

#######################################
######## THREE-GAUSSIAN MCMC ############
#######################################

def lnprior_mix3(theta):
    mu1, sig1, mu2, sig2, mu3, sig3, f1, f2 = theta
    if not (0 < f1 < 1 and 0 < f2 < 1 and 0 < f1+f2 < 0.9 and 0 < sig1 < 200 and 0 < sig2 < 40 and 0 < sig3 < 20 and -100 < mu1 < 100 and -30 < mu2 < 30 and -30 < mu3 < 10):
        return -np.inf

    return 0.0

def lnlike_mix3(theta, vel, vel_err):
    mu1, sig1, mu2, sig2, mu3, sig3, f1, f2 = theta

    var1 = sig1**2 + vel_err**2
    var2 = sig2**2 + vel_err**2
    var3 = sig3**2 + vel_err**2

    # Log PDFs for both components
    lnL1 = -0.5 * ((vel - mu1)**2 / (var1) + np.log(2 * np.pi * (var1)))
    lnL2 = -0.5 * ((vel - mu2)**2 / (var2) + np.log(2 * np.pi * (var2)))
    lnL3 = -0.5 * ((vel - mu3)**2 / (var3) + np.log(2 * np.pi * (var3)))

    # Combine in linear space via log-sum-exp
    lnL = np.sum(logsumexp([np.log(1 - f1 - f2) + lnL1, np.log(f1) + lnL2, np.log(f2) + lnL3], axis=0))
    return lnL

def lnprob_mix3(theta, vel, vel_err):
    lp = lnprior_mix3(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_mix3(theta, vel, vel_err)

def initialize_walkers_mix3(priors, nwalkers=50):
    ndim = 8
    p0 = np.zeros((nwalkers, ndim))
    for i, key in enumerate(['mu1', 'sig1', 'mu2', 'sig2', 'mu3', 'sig3', 'f1', 'f2']):
        mu, sigma = priors[key]
        p0[:, i] = mu + sigma * np.random.randn(nwalkers)
    return ndim, nwalkers, p0

def mcmc_mixture3(vel, vel_err, priors, nsteps = 5000):
    ndim, nwalkers, p0 = initialize_walkers_mix3(priors)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_mix3, args=(vel, vel_err))
    sampler.run_mcmc(p0, nsteps, progress=True)

    # Compute burn-in automatically
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
    except:
        burnin = 200

    for ii in [2, 3]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==2):
            vr = mcmc[1] 
            vr_err16 = mcmc[0]
            vr_err84 = mcmc[2] 
            vr_err   = (vr_err84 - vr_err16)/2.   # APPROX
        if (ii==3):
            sigma = mcmc[1] 
            sigma_err16 = mcmc[0]
            sigma_err84 = mcmc[2]

            sigma_err_m   = (sigma - sigma_err16)
            sigma_err_p   = (sigma_err84 - sigma)

    samples = sampler.get_chain(discard = burnin, flat = True)
    return sampler, samples, vr, vr_err, sigma, sigma_err_p,sigma_err_m

def plot_mixture_results3(samples, vel, label = None, figname = 0):
    labels = [r"$v_{\rm MW}$", r"$\sigma_{\rm MW}$", r"$v_{\rm W1-1}$", r"$\sigma_{\rm W1-1}$", r"$v_{\rm W1-2}$", r"$\sigma_{\rm W1-2}$", r"$f1$", r"$f2$"]
    fig = corner.corner(samples, labels = labels, show_titles = True, quantiles = [0.16, 0.5, 0.84],
                               label_kwargs = {"labelpad": 5, "size": 18})
    if label != None:
        fig.gca().annotate(label, xy=(1.0, 1.0), xycoords="figure fraction", xytext=(-20, -10), 
                           textcoords="offset points", ha="right", va="top", fontsize = 14)
    plt.show()
    if figname != 0:
        fig.savefig(figname, dpi = 600, transparent = True)

    # posterior medians
    theta_best = np.median(samples, axis=0)
    print("Posterior medians:", dict(zip(labels, theta_best)))

    # plot fit
    mu1, sig1, mu2, sig2, mu3, sig3, f1, f2 = theta_best
    vgrid = np.linspace(np.min(vel), np.max(vel), 300)
    pdf1 = (1-f1-f2) * norm.pdf(vgrid, mu1, sig1) 
    pdf2 = f1 * norm.pdf(vgrid, mu2, sig2)
    pdf3 = f2 * norm.pdf(vgrid, mu3, sig3)

    plt.hist(vel, bins=25, density=True, alpha=0.5, label="Data")
    plt.plot(vgrid, pdf1, 'r-', lw=2, label="MW bkgd")
    plt.plot(vgrid, pdf2, 'b-', lw=2, label="W1-1")
    plt.plot(vgrid, pdf3, 'g-', lw=2, label="W1-1")
    plt.xlim(-200, 200)
    plt.legend()
    plt.show()

    return theta_best

################## mass estimate #####################
def calc_mass(vdisp, v_lower, v_upper, rh, rh_err, p = 1):
    """Calculate mass from velocity dispersion of DG.
    Parameters
    ----------
        vdisp   : velocity dispersion [km/s]
        v_lower : lower uncertainty [km/s]
        v_upper : upper uncertainty [km/s]
        rh      : half-light radius [pc]
        rh_err  : half-light radius uncertainty [pc]
    Returns
    -------
        Mhalf_lower : lower uncertainty on mass estimate [Msol]
        Mhalf       : mass estimate [Msol]
        Mhalf_upper : upper uncertainty on mass estimate [Msol]
    """
    # kms
    kms = u.km / u.s
    
    # velocity dispersion and rhalf
    vdisp_lower, vdisp, vdisp_upper = v_lower * kms, vdisp * kms, v_upper * kms
    rhalf, rhalf_err = rh * u.pc, rh_err * u.pc
    
    # mass estimate
    Mhalf = 4 * (const.G ** -1) * (vdisp ** 2) * rhalf

    a = (rhalf_err / rhalf)
    b_lower = (vdisp_lower / vdisp)
    b_upper = (vdisp_upper / vdisp)

    Mhalf_lower = Mhalf * ((a**2) + 4 * (b_lower**2))**0.5
    Mhalf_upper = Mhalf * ((a**2) + 4 * (b_upper**2))**0.5

    Mhalf = Mhalf.to(u.solMass).value
    Mhalf_lower = Mhalf_lower.to(u.solMass).value
    Mhalf_upper = Mhalf_upper.to(u.solMass).value

    if p:
        print("M = {:.3E} + {:.3E} - {:.3E} Msol".format(Mhalf, Mhalf_upper, Mhalf_lower))
    
    # output
    return Mhalf_lower, Mhalf, Mhalf_upper

################# mass to light ratio ################
def mass_to_light(mass_lower, mass, mass_upper, lum, lum_err, lum_band = "V"):
    ml = mass / lum
    ml_lower = ml * (((mass_lower / mass) ** 2 + (lum_err / lum) ** 2) ** 0.5)
    ml_upper = ml * (((mass_upper / mass) ** 2 + (lum_err / lum) ** 2) ** 0.5)

    print("(M/L)_" + lum_band + " = {:.0f} + {:.0f} - {:.0f} Msol".format(ml, ml_upper, ml_lower))
    
    return ml_lower, ml, ml_upper

######################################################
################ proper motion MCMC ##################
######################################################

def lnprob_pm(theta, pmra, pmra_err, pmdec, pmdec_err):
    lp = lnprior_pm(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_pm(theta, pmra, pmra_err, pmdec, pmdec_err)

def lnprior_pm(theta):

    if (-1 < theta[0] < 4) & (-3 < theta[1] < 1):
        return 0.0
    
    return -np.inf

def lnlike_pm(theta, pmra, pmra_err, pmdec, pmdec_err):
    # Gaussian 
    term1 = -0.5 * np.sum(((theta[0] - pmra) / pmra_err) ** 2)
    term2 = -0.5 * np.sum(((theta[1] - pmdec) / pmdec_err) ** 2)
    lnl   = term1 + term2
    return lnl

def initialize_walkers_pm(pmra_guess, pmdec_guess):
    ndim, nwalkers = 2, 20
    p0 =  np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    # pmra, pmdec
    p0[:,0] = (p0[:,0] * 2 - 0.5) + pmra_guess
    p0[:,1] = (p0[:,1] * 2 - 1.5) + pmdec_guess
    
    return ndim, nwalkers, p0

def run_sampler_pm(sampler, p0, max_n):

    pos, prob, state = sampler.run_mcmc(p0, max_n, progress = False)
    
    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100
        
    return sampler, convg, burnin

def mcmc_pm(pmra, pmra_err, pmdec, pmdec_err, pmra_guess, pmdec_guess, label, figname, max_n = 5000):

    ndim, nwalkers,p0   = initialize_walkers_pm(pmra_guess, pmdec_guess)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_pm, args = (pmra, pmra_err, pmdec, pmdec_err))

    sampler, convg, burnin = run_sampler_pm(sampler, p0, max_n)
    #print(burnin, convg)
    theta = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]

    # PLOT CORNER
    labels=[r'$\mu_{\alpha} \cos(\delta)$', r'$\mu {}_{\alpha}$']
    ndim=2
    samples = sampler.chain[:, 2*burnin:, :].reshape((-1, ndim))
    fig     = corner.corner(samples, labels = labels, show_titles = True, quantiles=[0.16, 0.5, 0.84],
                               label_kwargs = {"labelpad": 5, "size": 18})
    fig.gca().annotate(label, xy=(1.0, 1.0), xycoords="figure fraction", xytext=(-20, -10), 
                           textcoords="offset points", ha="right", va="top", fontsize = 14)
    fig.savefig(figname, dpi = 600, transparent = True)
    
    return sampler, theta

def calc_pm(pmra, pmra_err, pmdec, pmdec_err, pmra_guess, pmdec_guess, label, figname):

    sampler, theta = mcmc_pm(pmra, pmra_err, pmdec, pmdec_err, pmra_guess, pmdec_guess, label, figname)

    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        #print(tau,burnin, converged)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100

    if (burnin < 75):
        burnin = 75

    # print(burnin, convg)

    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            pmra = mcmc[1] 
            pmra_err16 = mcmc[0]
            pmra_err84 = mcmc[2] 
            pmra_err   = (pmra_err84 - pmra_err16)/2.   # APPROX
        if (ii==1):
            pmdec = mcmc[1] 
            pmdec_err16 = mcmc[0]
            pmdec_err84 = mcmc[2] 
            pmdec_err   = (pmdec_err84 - pmdec_err16)/2.   # APPROX

            # pmdec_err_m   = (sigma - sigma_err16)
            # pmdec_err_p   = (sigma_err84 - sigma)

    return pmra, pmra_err, pmdec, pmdec_err, sampler


######################################################
################# metallicity MCMC ###################
######################################################

def lnprob_met(theta, met, met_err):
 
    lp = lnprior_met(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike_met(theta, met, met_err)

def lnprior_met(theta):
    if (-3.5 < theta[0] < -1.5) & (0. < theta[1] < 0.75):
        return 0.0
    
    return -np.inf

def lnlike_met(theta, met, met_err):

    # Gaussian 
    term1 = -0.5*np.sum(np.log(met_err**2 + theta[1]**2))
    term2 = -0.5*np.sum((met - theta[0])**2/(met_err**2 + theta[1]**2))
    term3 = -0.5*np.size(met)* np.log(2*np.pi)
    
    lnl  = term1+term2+term3
    
    return lnl

def initialize_walkers_met(met_guess, msig_guess):
    # met, msig
    ndim, nwalkers = 2, 20
    p0 =  np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))

    # met, met err
    p0[:,0] = np.random.uniform(met_guess-1, met_guess+1, nwalkers)
    p0[:,1] = np.random.uniform(msig_guess-0.3, msig_guess+0.3, nwalkers)

    return ndim,nwalkers,p0

def run_sampler_met(sampler, p0, max_n):
    pos, prob, state = sampler.run_mcmc(p0, max_n,progress=False)
    
    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100
        
    return sampler, convg, burnin

def mcmc_metdisp(met, met_err, met_guess, msig_guess, label, figname, max_n = 5000, plot=1):

    ndim, nwalkers,p0   = initialize_walkers_met(met_guess, msig_guess)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_met, args=(met, met_err))

    sampler, convg, burnin = run_sampler_met(sampler, p0, max_n)
    #print(burnin, convg)
    theta = [np.mean(sampler.chain[:, burnin:, i])  for i in [0,1]]

    if (plot == 1):
        # PLOT CORNER
        labels = [r'$[Fe/H]_{W1}$', r'$\sigma_{[Fe/H]}$']
        ndim=2
        samples = sampler.chain[:, 2*burnin:, :].reshape((-1, ndim))
        fig     = corner.corner(samples, labels = labels, show_titles = True, quantiles=[0.16, 0.5, 0.84],
                               label_kwargs = {"labelpad": 5, "size": 18})
        fig.gca().annotate(label, xy=(1.0, 1.0), xycoords="figure fraction", xytext=(-20, -10), 
                           textcoords="offset points", ha="right", va="top", fontsize = 14)
        fig.savefig(figname, dpi = 600, transparent = True)
    
    return sampler, theta

def calc_metdisp(met, met_err, met_guess, msig_guess, label, figname, plot=1):
    sampler, theta = mcmc_metdisp(met, met_err, met_guess, msig_guess, label, figname, plot=plot)

    try:
        tau    = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        converged = np.all(tau * 100 < sampler.iteration)
        #print(tau,burnin, converged)
        convg = np.sum(converged)
    except:
        convg=0
        burnin=100

    if (burnin < 75):
        burnin = 75

    #print(burnin, convg)

    for ii in [0,1]:
        mcmc = np.percentile(sampler.chain[:,burnin:, ii], [16, 50, 84])
        if (ii==0):
            met = mcmc[1] 
            met_err16 = mcmc[0]
            met_err84 = mcmc[2] 
            met_err_m = (met - met_err16)
            met_err_p = (met_err84 - met)

        if (ii==1):
            sigma = mcmc[1]
            sigma_err16 = mcmc[0]
            sigma_err84 = mcmc[2]
            sigma_err_m   = (sigma - sigma_err16)
            sigma_err_p   = (sigma_err84 - sigma)

    return met, met_err_m, met_err_p, sigma, sigma_err_p, sigma_err_m, sampler

################# mass to light ratio ################
def calc_rhalf_pc(rhalf, rhalf_err, dist, dist_err): 
    """Calculate rhalf [pc] from rhalf [arcmin] and distance [kpc].
    Parameters
    ----------
        rhalf     : half-light radius [arcmin]
        rhalf_err : half-light radius uncertainty [arcmin]
        dist      : distance [kpc]
        dist_err  : distance uncertainty [kpc]
    Returns
    -------
        rhalf_pc     : half-light radius [pc]
        rhalf_pc_err : half-light radius uncertainty [pc]
    """
    # rhalf in arc min, dist in kpc
    rhalf_deg = rhalf / 60
    rhalf_pc = rhalf_deg * (np.pi / 180) * (dist * 1e3)    # half-light radius [pc]

    a = rhalf_pc * rhalf_err / rhalf
    b = rhalf_pc * dist_err / dist

    rhalf_pc_err = (a ** 2 + b **2) ** (0.5)

    return rhalf_pc, rhalf_pc_err