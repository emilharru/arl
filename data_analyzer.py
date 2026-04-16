import numpy as np
from scipy.stats import median_abs_deviation
from scipy.signal import correlate, welch

def compute_channel_statistics(data, sampling_rate=100.0):
    """
    Computes a comprehensive dictionary of summary statistics for a 2D array
    of channel data (shape: [samples, time_steps]).
    """
    if data.size == 0:
        return {}


    num_samples, length = data.shape


    is_missing = np.isnan(data)
    missingness_rate = np.mean(is_missing)

    max_gaps = []
    for row in is_missing:

        runs = np.diff(np.where(np.concatenate(([row[0]], row[:-1] != row[1:], [True])))[0])[::2]
        if row[0]:
           max_gap = np.max(runs) if len(runs) > 0 else 0
        else:
           runs = np.diff(np.where(np.concatenate(([row[0]], row[:-1] != row[1:], [True])))[0])[1::2]
           max_gap = np.max(runs) if len(runs) > 0 else 0
        max_gaps.append(max_gap)

    avg_max_gap = np.mean(max_gaps) if max_gaps else 0.0


    data_clean = np.nan_to_num(data, nan=0.0)


    global_median = np.median(data_clean)

    global_mad = median_abs_deviation(data_clean.flatten(), scale='normal')


    p5 = np.percentile(data_clean, 5)
    p95 = np.percentile(data_clean, 95)


    diffs = np.diff(data_clean, axis=1)
    flatline_rate = np.mean(diffs == 0.0)


    roughness = np.std(diffs)


    lags_to_check = [1, 5, 10]
    acf_lags = {f"lag_{l}": [] for l in lags_to_check}
    zero_crossings = []

    for row in data_clean:

        row_centered = row - np.mean(row)
        if np.all(row_centered == 0):
             for l in lags_to_check:
                 acf_lags[f"lag_{l}"].append(0.0)
             zero_crossings.append(0)
             continue


        acf = correlate(row_centered, row_centered, mode='full')
        acf = acf[len(acf)//2:]

        acf = acf / acf[0] if acf[0] != 0 else acf

        for l in lags_to_check:
            val = acf[l] if l < len(acf) else 0.0
            acf_lags[f"lag_{l}"].append(val)


        sign_changes = np.where(np.diff(np.signbit(acf)))[0]
        if len(sign_changes) > 0:
            zero_crossings.append(int(sign_changes[0] + 1))
        else:
            zero_crossings.append(length)

    avg_acf = {k: np.mean(v) for k, v in acf_lags.items()}
    avg_zero_crossing = np.mean(zero_crossings)


    low_band_fracs = []
    mid_band_fracs = []
    high_band_fracs = []
    centroids = []

    for row in data_clean:
        if np.all(row == 0):
            low_band_fracs.append(0)
            mid_band_fracs.append(0)
            high_band_fracs.append(0)
            centroids.append(0)
            continue


        freqs, psd = welch(row, fs=sampling_rate, nperseg=min(length, 256))
        total_power = np.sum(psd)

        if total_power == 0:
            low_band_fracs.append(0)
            mid_band_fracs.append(0)
            high_band_fracs.append(0)
            centroids.append(0)
            continue


        nyquist = sampling_rate / 2.0
        low_thresh = nyquist / 3.0
        mid_thresh = (nyquist / 3.0) * 2.0

        low_power = np.sum(psd[freqs < low_thresh])
        mid_power = np.sum(psd[(freqs >= low_thresh) & (freqs < mid_thresh)])
        high_power = np.sum(psd[freqs >= mid_thresh])

        low_band_fracs.append(low_power / total_power)
        mid_band_fracs.append(mid_power / total_power)
        high_band_fracs.append(high_power / total_power)


        centroid = np.sum(freqs * psd) / total_power
        centroids.append(centroid)

    return {
        "length": length,
        "sampling_rate": sampling_rate,
        "jitter_var": 0.0,
        "missingness_rate": float(missingness_rate),
        "max_gap_avg": float(avg_max_gap),
        "median": float(global_median),
        "mad": float(global_mad),
        "percentile_05": float(p5),
        "percentile_95": float(p95),
        "clipping_flatline_rate": float(flatline_rate),
        "roughness_std_diff": float(roughness),
        "acf_lags": {k: float(v) for k, v in avg_acf.items()},
        "first_zero_crossing_lag": float(avg_zero_crossing),
        "bandpower_fraction_low": float(np.mean(low_band_fracs)),
        "bandpower_fraction_mid": float(np.mean(mid_band_fracs)),
        "bandpower_fraction_high": float(np.mean(high_band_fracs)),
        "spectral_centroid": float(np.mean(centroids))
    }
