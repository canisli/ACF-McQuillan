# ACF-McQuillan
This is my implementation of the autocorrelation-based period detection algorithm described in [McQuillan et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1203M/abstract) and [McQuillan et al. 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M/abstract). 
Autocorrelation functions (ACFs) have been shown to be well-suited for detecting rotational periods in stellar light curves.

See `example.ipynb` for a walkthrough of how it's used. The algorithm is in `acf_mma.py`, with some supporting methods in `timeseriestools.py`. 

ACFs assume evenly spaced data. This can be obtained by applying `map_and_interpolate_gaps` in `timeseriestools.py`, which remaps the data to make the cadence constant, linearly interpolating temporal gaps (e.g. t=802 to t=809 in the plot below).

![timeseries](https://github.com/canisli/ACF-McQuillan/assets/73449574/ac6b8479-300f-4a1c-a4f7-e799a032bcf9)
![ACF](https://github.com/canisli/ACF-McQuillan/assets/73449574/099902b7-8247-41df-b94e-dcfe13bb9c64)

## How it works
Aftering computing the ACF, the algorithm first determines the dominant period, defined as whichever of the first two ACF peaks has a larger local peak height (defined as the average distance from peak to adjacent troughs). In the example above, the second peak at ~18 days was chosen. Then, the algorithm looks for harmonics of this dominant period, i.e. peaks near integer multiples of the dominant period (orange shaded regions). Finally, the rotation period and its uncertainty are computed by fitting a line to the period of each peak vs. its harmonic number.

## Dependencies
* numpy
* scipy
* numba
