"""
Compute the autocorrelation of timeseries and identify dominant rotational period of stellar light curves.
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress

from timeseriestools import find_peaks_and_troughs

@njit
def compute_acf(y):
    """Compute ACF of evenly spaced data. Speed up with numba.
    Refer to Shumway & Stoffer 2010 for review on autocorrelation""" 

    n = len(y) 
    r = np.zeros(n)
    ybar = np.mean(y)
    SSTo = 0
    for i in range(0, n):
        SSTo += (y[i] - ybar)**2
    if SSTo == 0: # values are all zero
        raise ValueError('Cannot compute ACF of horizontal line')
        
    for k in range(0, n):
        r_k = 0 
        for i in range(0, n-k):
            r_k += (y[i]-ybar) * (y[i+k]-ybar)
        r[k] = (r_k/SSTo)
    return r

def smooth_acf(acf, window_size=56, fwhm=18):
    """Perform a Gaussian smooth on the ACF.
    window_size and fwhm (full width half max) are provided in units of lag"""
    sigma = fwhm/2.355
    truncate=window_size/sigma

    acf_smoothed = gaussian_filter(acf, sigma=sigma, truncate=truncate)
    return acf_smoothed

class PeakNotFoundException(Exception):
    """Peak with minimum local peak height (LPH) not founds"""
    pass

def _identify_rotational_period(lags, acf_smoothed, min_lph=0.1, version=2014, 
                                ax=None, verbose=False):
    """Search for dominant period inside a smoothed ACF"""

    indices, is_peak = find_peaks_and_troughs(acf_smoothed)

    """Compute period and local peak heights (LPHs) of peaks"""
    peak_periods, local_peak_heights = [], []
    for i in range(0, len(indices)-1):
        if is_peak[i]:
            peak_periods.append(lags[indices[i]])
            local_peak_heights.append(0.5 * ((acf_smoothed[indices[i]] - acf_smoothed[indices[i-1]]) + 
                                             (acf_smoothed[indices[i]] - acf_smoothed[indices[i+1]])))

    """Estimate dominant period with taller (larger LPH) of the first two peaks"""
    dominant_period = peak_periods[0] if local_peak_heights[0] > local_peak_heights[1]  else peak_periods[1]
    domiant_period_lph = max(local_peak_heights[0], local_peak_heights[1])

    if domiant_period_lph < min_lph:
        raise PeakNotFoundException('acf_mma: The dominant period\'s LPH was too small')

    if verbose:
        print(f'acf_mma: Dominant period: {dominant_period}')

    """Search for harmonics of the dominant period"""
    harmonic_periods, harmonic_period_lphs = [], []
    for peak_period, lph in zip(peak_periods, local_peak_heights):
        """2013 paper only requires first 10 harmonics, while 2014 paper only requires 4"""
        if version==2013 and len(harmonic_periods) == 10:
            break
        if version==2014 and len(harmonic_periods) == 4:
            break

        nearest_harmonic = round(peak_period/dominant_period) * dominant_period # closest integer multiple of dominant peak
        if np.abs(peak_period - nearest_harmonic) < 0.20 * dominant_period: # check within 20% of dominant period
            if len(harmonic_periods)>0:
                """Check if there isn't another peak correspoding to this harmonic"""
                if peak_period - harmonic_periods[-1] > 0.40 * dominant_period:
                    harmonic_periods.append(peak_period)
                    harmonic_period_lphs.append(lph)
            else:
                harmonic_periods.append(peak_period)
                harmonic_period_lphs.append(lph)
    
    if verbose:
        print(f'acf_mma: Peaks that are harmonics of dominant period: {harmonic_periods}')
    
    """Estimate rotational period from harmonic peaks"""
    P_rot = None
    P_rot_err = None

    if version==2013:
        period_intervals = np.diff(harmonic_periods)
        P_rot = np.median(period_intervals)
        median_absolute_deviation = np.median(np.abs(period_intervals-P_rot))
        P_rot_err = 1.483 * median_absolute_deviation/np.sqrt(len(harmonic_periods)-1)
    elif version==2014:
        y = np.concatenate([[0], harmonic_periods])
        x = np.round(y/dominant_period)
        lrr = linregress(x, y, alternative='greater')
        P_rot = lrr.slope
        P_rot_err = lrr.stderr

    """Plot"""
    if ax is not None:
        """Plot dominant period"""
        if verbose:
            ax.axvline(peak_periods[0], color='black', linestyle='--')
            ax.axvline(peak_periods[1], color='black', linestyle='--')
        
        """Show intervals for harmonics of the dominant period to be found"""
        if verbose:
            for i in range(1,11):
                plt.axvspan(xmin=dominant_period * i - 0.20*dominant_period,
                            xmax=dominant_period * i + 0.20*dominant_period, alpha=0.3, color='orange')

    return (P_rot, P_rot_err)

def acf_mma(t, y, version=2014, min_lph = 0.1,
            plot=False, ax=None, verbose=False):
    """
    Use autocorrelation to search for rotational period in stellar light curve.
    See McQuillan et al. 2013 and 2014
    
    Parameters
        t, y: time series with time in units of days. 
        version: Which year of the paper to base the algorithm off (2013, 2014)
        min_lph: Minimum local peak height of the dominant period.
    Returns
        (P_rot, P_rot_err): Rotational period estimate and uncertainty
        flag: what smoothing was necessary to find a peak with LPH > min_lph
    """
    
    y = np.copy(y) 
    y -= np.mean(y)
    n = len(y)
    cadence = np.min(np.diff(t))
    lags = cadence * np.arange(n)
    acf = compute_acf(y)
    if plot:
        if ax is None:
            plt.figure(figsize=(12,6), constrained_layout=True)
            ax = plt.gca()
        plt.sca(ax)

    """Make sure time series is evenly spaced."""
    if np.max(np.diff(t)) - np.min(np.diff(t)) > 1e-3:
        print('acf_mma: Warning, data is likely not evenly spaced. Need to interpolate.')

    flag = 'default'
    try:
        """Try to smooth acf to best reveal a peak on the order of days"""
        acf_smoothed = smooth_acf(acf, window_size=56, fwhm=18/48/cadence)
        (P_rot, P_rot_err) = _identify_rotational_period(lags, acf_smoothed, min_lph=min_lph, version=version, ax=ax, verbose=verbose)
    except PeakNotFoundException:
        try:
            if verbose:
                print(f'acf_mma: Could not find periods with LPH>{min_lph}. Trying no smoothing')
            acf_smoothed = np.copy(acf)
            (P_rot, P_rot_err) = _identify_rotational_period(lags, acf_smoothed, min_lph=min_lph, version=version, ax=ax, verbose=verbose)
            flag = 'no smoothing'
        except PeakNotFoundException:
            try:
                if verbose:
                    print(f'acf_mma: Still could not find periods with LPH>{min_lph}. Trying hard smoothing')
                acf_smoothed = smooth_acf(acf, window_size=200, fwhm=60)
                (P_rot, P_rot_err) = _identify_rotational_period(lags, acf_smoothed, min_lph=min_lph, version=version, ax=ax, verbose=verbose)
                flag = 'hard smoothing'
            except PeakNotFoundException:
                if verbose:
                    print('acf_mma: Did not detect any rotational period!')
                (P_rot, P_rot_err) = np.nan, np.nan
                flag = 'failure'
    
    if verbose:
        print(f'acf_mma: P_rot = {P_rot}±{P_rot_err}')

    """Plot"""
    if plot:
        plt.plot(lags, acf, alpha=0.5, color='k', linestyle='-.')
        plt.plot(lags, acf_smoothed, color='k')
        if not np.isnan(P_rot):
            plt.axvline(P_rot, color='red', zorder=1000)
            plt.axvspan(P_rot - P_rot_err, P_rot + P_rot_err, color='red', alpha=0.3, zorder=1000)
            # plt.axvline(27, linestyle='-.') # Sun's rotational period

    return (P_rot, P_rot_err), flag