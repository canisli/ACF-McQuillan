"""Tools for working with time series"""
import numpy as np
from astropy.stats import sigma_clip as astropy_sigma_clip

def map_and_interpolate_gaps(t, y, dt=None, 
                            fill_large_gaps=False, min_gap_size=np.inf, fill_value=0):
    """Remap data to make the cadence constant. 
    The new cadence can be specified is by default the median cadence.
    Option to fill large gaps with a specified value instead of interpolating"""

    if dt is None:
        dt = np.nanmedian(np.diff(t))
    t_mapped = np.arange(np.nanmin(t), np.nanmax(t) + dt, dt)
    y_mapped = np.interp(t_mapped, t, y)

    """Find large gaps in original data and replace in mapped data"""
    gaps = np.diff(t)
    gap_indices = np.where(gaps > min_gap_size)[0]
    for i in gap_indices:
        mask = np.logical_and(t[i] < t_mapped, t_mapped < t[i+1])
        y_mapped[mask] = fill_value

    return t_mapped, y_mapped

def find_peaks_and_troughs(y):
    """Return the indices of local maxima and minima in a signal by comparing with neighboring values"""

    indices = []
    is_peak = []
    for i in range(1, len(y)-1):
        if (y[i]-y[i-1]) * (y[i+1]-y[i]) < 0:
            indices.append(i)
            is_peak.append(y[i] - y[i-1] > 0)
    return np.array(indices), np.array(is_peak)

def remove_nans(t, *y, verbose=False):
    nan_mask = np.logical_or.reduce([np.isnan(t), *[np.isnan(yi) for yi in y]])
    unmasked = np.invert(nan_mask)
    if verbose:
        nan_count = np.sum(nan_mask)
        print(f'remove_nans: Removed {nan_count}/{len(t)}≈{round(nan_count/len(t)*100, 1)}% nan values from original data')
    if len(y) == 0:
        return np.array(t)[unmasked]
    else:
        return np.array(t)[unmasked], *[np.array(yi)[unmasked] for yi in y]

def sigma_clip(t, *y, sigma=5, verbose=False):
    clipped_mask = astropy_sigma_clip(y[0], sigma=sigma, masked=True).mask
    unclipped_mask = np.invert(clipped_mask)
    if verbose:
        num_clipped = np.sum(clipped_mask)
        num_total = len(t)
        percentage = round(num_clipped/num_total*100, 1)
        print(f'sigma_clip: {sigma}σ clipped {num_clipped}/{num_total}≈{percentage}% of the unmasked data')
    
    return t[unclipped_mask],*[yi[unclipped_mask] for yi in y] 