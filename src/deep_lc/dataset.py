import numpy as np
import torch
from torchvision import transforms
from astropy.timeseries import LombScargle
from gatspy import periodic


def lc_to_image_array(
    time_list,
    flux_list,
    figure_pixel_size=(256, 512),
    figure_edge=5,
    square_size=5,
    cumulative=False,
):
    grad = np.zeros([figure_pixel_size[0], figure_pixel_size[1]])

    y_min = np.inf
    y_max = -np.inf
    t_min = np.inf
    t_max = -np.inf
    for time, flux in zip(time_list, flux_list):
        # continue if it's empty
        if len(time) == 0:
            continue
        # check if flux is an ndarray
        if not isinstance(flux, np.ndarray):
            flux = np.asarray(flux)
            time = np.asarray(time)
        length = len(flux[~np.isnan(flux)])
        if length <= 1:
            continue  # skip this segment
        if np.nanmin(flux) < y_min:
            y_min = np.nanmin(flux)
        if np.nanmax(flux) > y_max:
            y_max = np.nanmax(flux)
        if np.nanmin(time) < t_min:
            t_min = np.nanmin(time)
        if np.nanmax(time) > t_max:
            t_max = np.nanmax(time)

    if y_min == y_max:
        for flux in flux_list:
            flux[~np.isnan(flux)] = 0
        y_min, y_max = -1, 1

    for i, (time, flux) in enumerate(zip(time_list, flux_list)):
        # continue if it's empty
        if len(time) == 0:
            continue
        length = len(flux[~np.isnan(flux)])
        if length <= 1:
            continue  # skip this segment
        length = len(flux[~np.isnan(flux)])
        if length <= 1:
            continue
        xcoord = np.rint(
            (flux - y_min)
            / (y_max - y_min)
            * (figure_pixel_size[0] - figure_edge * 2 - 1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            xcoord = np.clip(xcoord, 0, figure_pixel_size[0] - 1).astype(int) + figure_edge

        ycoord = np.rint(
            (time - t_min)
            / (t_max - t_min)
            * (figure_pixel_size[1] - figure_edge * 2 - 1)
        )
        ycoord = np.clip(ycoord, 0, figure_pixel_size[1] - 1).astype(int) + figure_edge
        square_size -= i
        # loop over the core coordinates and select squares around them
        for x, y in zip(xcoord, ycoord):
            # calculate the boundaries of the square around this core
            xmin = max(0, x - square_size // 2)
            xmax = min(grad.shape[0], x + square_size // 2 + 1)
            ymin = max(0, y - square_size // 2)
            ymax = min(grad.shape[1], y + square_size // 2 + 1)

            # update the values in the selected square
            if cumulative:
                grad[xmin:xmax, ymin:ymax] += 1 - i / len(time_list)
            else:
                grad[xmin:xmax, ymin:ymax] = 1 - i / len(time_list)

    return np.ascontiguousarray(np.flipud(grad))


def ps_to_img(freq_list, power_list, figure_pixel_size=(256, 512), figure_edge=5):
    grad = np.zeros([figure_pixel_size[0], figure_pixel_size[1]])
    flat_freq_list = [item for sublist in freq_list for item in sublist]
    flat_power_list = [item for sublist in power_list for item in sublist]
    for freq, power in zip(freq_list, power_list):
        if not isinstance(power, np.ndarray):
            power = np.asarray(power)
        length = len(power)
        if np.isnan(power).all() or length <= 1:
            continue
        y_min, y_max = np.nanmin(flat_power_list), np.nanmax(flat_power_list)
        if y_min == y_max:
            power[~np.isnan(power)] = 0
            y_min, y_max = -1, 1

        xcoord = np.rint(
            (power - y_min)
            / (y_max - y_min)
            * (figure_pixel_size[0] - figure_edge * 2 - 1)
        )
        xcoord = np.clip(xcoord, 0, figure_pixel_size[0] - 1).astype(int) + figure_edge
        ycoord = np.rint(
            (freq - np.nanmin(flat_freq_list))
            / (np.nanmax(flat_freq_list) - np.nanmin(flat_freq_list))
            * (figure_pixel_size[1] - figure_edge * 2 - 1)
        )
        ycoord = np.round(ycoord).astype(int) + figure_edge
        # Iterate through each adjacent pair of points and draw lines between them
        for i in range(length - 1):
            x1, y1 = xcoord[i], ycoord[i]
            x2, y2 = xcoord[i + 1], ycoord[i + 1]

            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx, sy = 1 if x1 < x2 else -1, 1 if y1 < y2 else -1
            err = dx - dy

            while x1 != x2 or y1 != y2:
                grad[x1, y1] += 1
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        grad[grad > 0] = 1
    return np.ascontiguousarray(np.flipud(grad))


def bin_timeseries(time, values, nbins):
    # Assign data points to bins
    bin_edges = np.linspace(np.nanmin(time), np.nanmax(time), nbins + 1)
    bin_index = np.digitize(time, bin_edges)

    # Calculate mean value for each bin
    # fill nan to 0 for values
    values[np.isnan(values)] = 0
    
    # cancel the warning here
    with np.errstate(divide="ignore", invalid="ignore"):
        binned_values = np.bincount(bin_index, weights=values) / np.bincount(bin_index)
    # refill 0 to nan
    binned_values[binned_values == 0] = np.nan
    # Create new time array
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bin_centers, binned_values[1:-1]


def plot_phase_curve(
    time_list, flux_list, period, figure_pixel_size=(256, 256), *kwargs
):
    if period == 0:
        return np.zeros(figure_pixel_size), 0
    phase_list = []
    new_flux_list = []
    var_ranges = []
    for time_data, flux_data in zip(time_list, flux_list):
        if not isinstance(time_data, np.ndarray):
            time_data = np.asarray(time_data)
        if not isinstance(flux_data, np.ndarray):
            flux_data = np.asarray(flux_data)
        flux = flux_data[~np.isnan(flux_data)]
        time_data = time_data[~np.isnan(flux_data)]
        if time_data.size == 0:
            continue
        # new_flux_data = flux_data[flux_data<3*np.nanstd(flux_data) + np.nanmean(flux_data)]
        # time_data = time_data[flux_data<3*np.nanstd(flux_data) + np.nanmean(flux_data)]
        phase, new_flux_data = bin_timeseries((time_data / period) % 2, flux, 500)
        new_flux_list.append(new_flux_data)
        phase_list.append(phase)
        var_ranges.append((np.nanmax(new_flux_data)-np.nanmin(new_flux_data)) if ~np.isnan(new_flux_data).all() else 0)

    img = lc_to_image_array(
        phase_list,
        new_flux_list,
        figure_pixel_size=figure_pixel_size,
        square_size=5,
        cumulative=False,
        *kwargs,
    )
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, np.mean(var_ranges)


def fap_level(fap, t, y, y_err, filts, periods, n_random=100):
    model = periodic.LombScargleMultiband()
    power_max = []
    rng = np.random.default_rng(42)

    for _ in range(n_random):
        s = rng.integers(0, len(y), len(y))  # sample with replacement
        model.fit(t, y[s], y_err[s], filts[s])
        power = model.periodogram(periods)
        power_max.append(power.max())

    return np.quantile(power_max, 1 - fap, method="nearest")


def fold_lightcurve(light_curve, period):
    """_summary_

    Parameters
    ----------
    light_curve : _ (N, 2) array for time and flux,
        or (N, 3) array for time, flux, and filter,
        or (N, 4) array for time, flux, flux_error and filter, currently, we only support two filters.
    period : float
        period to fold a light curve
    """
    if light_curve.shape[1] == 2:
        time_data = light_curve[:, 0]
        flux_data = light_curve[:, 1]
        phase, flux = [(time_data / period) % 2], [flux_data]
    else:
        phase = []
        flux = []
        n_bands = len(np.unique(light_curve[:, 3]))
        for i in range(n_bands):
            time_data = light_curve[light_curve[:, 3] == i+1][:, 0]
            flux_data = light_curve[light_curve[:, 3] == i+1][:, 1]
            phase.append((time_data / period) % 2)
            flux.append(flux_data)
    return phase, flux


def light_curve_preprocess(light_curve, multiband_FAP=False):
    """Preprocess the light curve data.

    Parameters
    ----------
    light_curve : (N, 2) array for time and flux,
        or (N, 3) array for time, flux, and filter,
        or (N, 4) array for time, flux, flux_error and filter, currently, we only support two filters.
    """

    processed_lc = np.asarray(light_curve)

    if processed_lc.shape[1] == 2 or 3 or 4:
        pass
    else:
        raise ValueError(
            "The input light curve data is not valid. Please check the shape of the input data."
        )

    transformed_lc_data = processed_lc.copy()

    if processed_lc.shape[1] == 2:
        transformed_lc_data[:, 0] -= transformed_lc_data[0, 0]

        # check if more than 6 data points
        if sum(~np.isnan(transformed_lc_data[:, 1])) < 6:
            raise ValueError(
                "The input light curve data is not valid. Please check the length of the input data."
            )

        time_length = transformed_lc_data[-1, 0] - transformed_lc_data[0, 0]
        lgamp = np.log10(
            np.nanmax(transformed_lc_data[:, 1]) - np.nanmin(transformed_lc_data[:, 1])
        )
        time_diff = np.diff(transformed_lc_data[:, 0])
        minimum_cadence = time_diff[time_diff>0.0013].min()

        lc_param = torch.tensor([time_length, minimum_cadence, lgamp]).float()

        lc_time, lc_flux = transformed_lc_data[:, 0], transformed_lc_data[:, 1]
        lc_img = lc_to_image_array([lc_time], [lc_flux])
        lc_img = transforms.ToTensor()(lc_img)

        ls_lc = transformed_lc_data[~np.isnan(transformed_lc_data[:, 1])]

        ls = LombScargle(ls_lc[:, 0], ls_lc[:, 1], normalization="standard")
        freq, power = ls.autopower(
            minimum_frequency=2 / (ls_lc[-1, 0] - ls_lc[0, 0]),
            maximum_frequency=1 / minimum_cadence / 2,
            samples_per_peak=5,
        )
        freq_at_max_power = freq[np.argmax(power)]
        period_at_max_power = 1 / freq_at_max_power
        fap100 = ls.false_alarm_level(
            1e-2,
            minimum_frequency=2 / (ls_lc[-1, 0] - ls_lc[0, 0]),
            maximum_frequency=1 / minimum_cadence / 2,
        ).item()

        freq = freq[power > 0]
        power = power[power > 0]

        if fap100 > power.max():
            ps_img = np.zeros((256, 512))
        else:
            ps_img = ps_to_img(
                [np.log10(freq), [-3.5, 3.5]],
                [np.log10(power), [np.log10(fap100), np.log10(fap100)]],
            )
        ps_img = transforms.ToTensor()(ps_img)

        phase_img, var_ranges = plot_phase_curve(
            [ls_lc[:, 0]],
            [ls_lc[:, 1]],
            period_at_max_power * (fap100 < power.max()),
        )
        phase_img = transforms.ToTensor()(phase_img)
        
        ps_param = torch.tensor(
            [
                freq_at_max_power,
                period_at_max_power,
                np.log10(fap100),
                np.log10(var_ranges if var_ranges > 0 else 1),
            ]
        ).float()

    else:
        if processed_lc.shape[1] == 3:
            # insert a column for flux error
            transformed_lc_data = np.insert(transformed_lc_data, 2, 1e-3, axis=1)
        
        # make sure the color diamension is int type
        transformed_lc_data[:, 3] = transformed_lc_data[:, 3].astype(int)

        filter_mask1 = (transformed_lc_data[:, 3] == 1) & (
            ~np.isnan(transformed_lc_data[:, 1])
        )
        filter_mask2 = (transformed_lc_data[:, 3] == 2) & (
            ~np.isnan(transformed_lc_data[:, 1])
        )
        # check if more than 6 data points
        if sum(filter_mask1) < 6 and sum(filter_mask2) < 6:
            raise ValueError(
                "The input light curve data is not valid. Please check the length of the input data."
            )

        if sum(filter_mask1) > 1:
            time_length_1 = (
                transformed_lc_data[filter_mask1][-1, 0]
                - transformed_lc_data[filter_mask1][0, 0]
            )

            lgamp_1 = np.log10(
                np.nanmax(transformed_lc_data[filter_mask1][:, 1])
                - np.nanmin(transformed_lc_data[filter_mask1][:, 1])
            )
            lc_time_1, lc_flux_1 = (
                transformed_lc_data[filter_mask1][:, 0],
                transformed_lc_data[filter_mask1][:, 1],
            )
            diff_t1 = np.diff(transformed_lc_data[filter_mask1][:, 0])
            minimum_cadence1 = diff_t1[diff_t1>(2/60/24)].min()
        else:
            time_length_1 = 0
            lgamp_1 = np.nan
            lc_time_1 = []
            lc_flux_1 = []
            minimum_cadence1 = 1e3
        if sum(filter_mask2) > 1:
            time_length_2 = (
                transformed_lc_data[filter_mask2][-1, 0]
                - transformed_lc_data[filter_mask2][0, 0]
            )

            lgamp_2 = np.log10(
                np.nanmax(transformed_lc_data[filter_mask2][:, 1])
                - np.nanmin(transformed_lc_data[filter_mask2][:, 1])
            )
            lc_time_2, lc_flux_2 = (
                transformed_lc_data[filter_mask2][:, 0],
                transformed_lc_data[filter_mask2][:, 1],
            )
            diff_t2 = np.diff(transformed_lc_data[filter_mask2][:, 0])
            minimum_cadence2 = diff_t2[diff_t2>0.0013].min()
        else:
            time_length_2 = 0
            lgamp_2 = np.nan
            lc_time_2 = []
            lc_flux_2 = []
            minimum_cadence2 = 1e3

        time_length = max(time_length_1, time_length_2)

        if np.isnan(lgamp_1) and ~np.isnan(lgamp_2):
            lgamp = lgamp_2
        elif np.isnan(lgamp_2) and ~np.isnan(lgamp_1):
            lgamp = lgamp_1
        elif ~np.isnan(lgamp_1) and ~np.isnan(lgamp_2):
            lgamp = (lgamp_2 + lgamp_1) / 2

        # minimum cadence should calculated by each filter
        minimum_cadence = max(min(minimum_cadence1, minimum_cadence2), 1e-3)

        lc_param = torch.tensor([time_length, minimum_cadence, lgamp]).float()

        try:
            lc_img = lc_to_image_array([lc_time_1, lc_time_2], [lc_flux_1, lc_flux_2])
        except TypeError:
            print([lc_time_1, lc_time_2], [lc_flux_1, lc_flux_2])
        lc_img = transforms.ToTensor()(lc_img)

        # calculate power spectrum
        model = periodic.LombScargleMultiband()
        ls_lc = transformed_lc_data[~np.isnan(transformed_lc_data[:, 1])].copy()
        model.fit(ls_lc[:, 0], ls_lc[:, 1], ls_lc[:, 2], ls_lc[:, 3])

        freqs = np.arange(
            2 / (np.nanmax(ls_lc[:, 0]) - np.nanmin(ls_lc[:, 0])),
            1 / minimum_cadence / 2,
            2 * np.pi / (np.nanmax(ls_lc[:, 0]) - np.nanmin(ls_lc[:, 0])) / 5,
        )
        periods = 1 / freqs
        power = model.periodogram(periods)


        if multiband_FAP:
            fap100 = fap_level(
                1e-2, ls_lc[:, 0], ls_lc[:, 1], ls_lc[:, 2], ls_lc[:, 3], periods
            )
        else:
            fap100 = 0

        freq_at_max_power = freqs[np.nanargmax(power)]
        period_at_max_power = 1 / freq_at_max_power

        freq = freqs[power > 0]
        power = power[power > 0]

        if fap100 == 0:
            ps_img = ps_to_img(
                [np.log10(freq), [-3.5, 3.5]],
                [np.log10(power), [np.nan, np.nan]],
            )
        else:
            ps_img = ps_to_img(
                [np.log10(freq), [-3.5, 3.5]],
                [np.log10(power), [np.log10(fap100), np.log10(fap100)]],
            )
        ps_img = transforms.ToTensor()(ps_img)

        phase_img, var_ranges = plot_phase_curve(
            [lc_time_1, lc_time_2],
            [lc_flux_1, lc_flux_2],
            period_at_max_power,
        )
        phase_img = transforms.ToTensor()(phase_img)

        ps_param = torch.tensor(
            [
                freq_at_max_power,
                period_at_max_power,
                np.log10(fap100) if fap100 != 0 else fap100,
                np.log10(var_ranges if var_ranges > 0 else 1),
            ]
        ).float()

    return (
        lc_img,
        ps_img,
        phase_img,
        lc_param,
        ps_param,
        transformed_lc_data,
        np.array([freq, power]).T,
    )


def data_collate_fn(batch):
    img = torch.stack([item[0] for item in batch])
    ps_img = torch.stack([item[1] for item in batch])
    phase_img = torch.stack([item[2] for item in batch])
    lc_param = torch.stack([item[3] for item in batch])
    ps_param = torch.stack([item[4] for item in batch])
    label = torch.tensor([item[5] for item in batch])
    lc_data = [item[6] for item in batch]
    ps_data = [item[7] for item in batch]
    params_data = torch.tensor([item[8] for item in batch])
    return img, ps_img, phase_img, lc_param, ps_param, label, lc_data, ps_data


class LocalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lcs,
        labels,
        parameters=None,
    ):
        self.lcs = lcs
        self.labels = labels
        self.parameters = parameters

    def __len__(self):
        return len(self.lcs)

    def __getitem__(self, idx):
        lc = self.lcs[idx]
        params = self.parameters[idx]
        label = self.labels[idx]

        (lc_img,
        ps_img,
        phase_img,
        lc_param,
        ps_param,
        lc_data,
        ps_data,) = light_curve_preprocess(lc)

        return lc_img, ps_img, phase_img, lc_param, ps_param, label, lc_data, ps_data, params