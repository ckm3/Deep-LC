import numpy as np
from uncertainties import unumpy

def light_curve_preparation(time, flux, flux_err=None, bands=None, magnitude_or_flux="flux"):
    """
    Prepare light curve data for modeling.

    Parameters
    ----------
    time : array-like
        Time values of the observations in the unit of day.
    flux : array-like
        Flux or magnitude values of the observations.
    flux_err : array-like, optional
        Error values of the flux or magnitude measurements.
    bands : array-like, optional
        Bandpass filter IDs of the observations. Starts from 1.
    magnitude_or_flux : str, optional
        Whether the input data is in flux or magnitude units. Default is "flux".

    Returns
    -------
    A numpy array with shape (N, 2) for time and flux input
    or (N, 3) for time, flux, and bands input,
    or (N, 4) for time, flux, flux_err, and bands input.

    """
    # Convert to numpy array
    time = np.ascontiguousarray(time)
    flux = np.ascontiguousarray(flux)

    # Check input and make sure they are not empty
    if len(time) == 0 or len(flux) == 0:
        raise ValueError("Input data cannot be empty.")
    
    if len(time) != len(flux):
        raise ValueError("time and flux must have the same length.")
    
    if flux_err is not None:
        flux_err = np.ascontiguousarray(flux_err)
        if len(time) != len(flux_err):
            raise ValueError("time and flux_err must have the same length.")
    
    if bands is not None:
        bands = np.ascontiguousarray(bands, dtype=int)
        if len(time) != len(bands):
            raise ValueError("time and bands must have the same length.")
        
    if magnitude_or_flux not in ["magnitude", "flux"]:
        raise ValueError("magnitude_or_flux must be either 'magnitude' or 'flux'.")


    # Convert magnitude to flux if input is in magnitude
    if magnitude_or_flux == "magnitude":
        if bands is not None:
            multiband_list = []
            if flux_err is None:
                flux_err = np.zeros_like(flux)
            for i in np.unique(bands):
                mag_array = unumpy.uarray(
                    flux[bands==i], flux_err[bands==i]
                )
                flux_from_mag = (
                    10 ** ((mag_array - np.nanmedian(mag_array)) / -2.5)
                    - 1
                )
                temp_flux = unumpy.nominal_values(flux_from_mag)
                temp_flux_err = unumpy.std_devs(flux_from_mag)
                multiband_list.append(np.array([time[bands==i], temp_flux, temp_flux_err, bands[bands==i]]).T)

            processed_array = np.concatenate(multiband_list, axis=0)
            if np.all(flux_err==0):
                processed_array = processed_array[:, [0,1,3]].copy()
        else:
            flux_from_mag = 10 ** ((flux - np.nanmedian(flux)) / -2.5) - 1
            processed_array = np.array([time, flux_from_mag]).T
    else:
        if bands is not None:
            multiband_list = []
            if flux_err is None:
                flux_err = np.zeros_like(flux)
            for i in np.unique(bands):
                flux_median = np.nanmedian(flux[bands==i])
                if np.isclose(flux_median, 0, rtol=1e-5):
                    temp_flux = flux[bands==i]
                    temp_flux_err = flux_err[bands==i]
                else:
                    flux_uarray = unumpy.uarray(flux[bands==i], flux_err[bands==i])
                    rel_flux = (flux_uarray - np.nanmedian(flux_uarray)) / np.abs(np.nanmedian(flux_uarray))
                    temp_flux = unumpy.nominal_values(rel_flux)
                    temp_flux_err = unumpy.std_devs(rel_flux)
                multiband_list.append(np.array([time[bands==i], temp_flux, temp_flux_err, bands[bands==i]]).T)
            processed_array = np.concatenate(multiband_list, axis=0)
            if np.all(flux_err==0):
                processed_array = processed_array[:, [0,1,3]].copy()
        else:
            processed_array = np.array([time, flux]).T
    
    # Sort processed_array by time
    sort_idx = np.argsort(processed_array[:, 0])
    processed_array = processed_array[sort_idx]

    return processed_array



if __name__ == "__main__":
    time = np.linspace(2.4, 12.54, 1000)
    flux = np.random.normal(12, 1, 1000)
    flux_err = np.random.uniform(0, 0.1, 1000)
    bands = np.random.choice([1, 2, 3], 1000)
    
    lc_array = light_curve_preparation(time, flux, flux_err=None, bands=bands, magnitude_or_flux="magnitude")

    pass