# ACF-McQuillan
This is my implementation of the autocorrelation-based period detection algorithm described in [McQuillan et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.1203M/abstract) and [McQuillan et al. 2014](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...24M/abstract). 
Autocorrelation functions have been shown to be well-suited for detecting rotational periods in stellar light curves.

See `example.ipynb` for a walkthrough of how it's used. The algorithm is in `acf_mma.py`, with some supporting methods in `timeseriestools.py`.

![timeseries](https://github.com/canisli/ACF-McQuillan/assets/73449574/ac6b8479-300f-4a1c-a4f7-e799a032bcf9)
![ACF](https://github.com/canisli/ACF-McQuillan/assets/73449574/099902b7-8247-41df-b94e-dcfe13bb9c64)

**Dependencies**
* numpy
* scipy
* numba (to speed up computations)
