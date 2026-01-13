import concurrent.futures
import multiprocessing
from pathlib import Path

import numpy as np
import psutil
import scipy.fft
import scipy.signal
import scipy.stats

from .reader import Reader

LF_CUTOFF_HZ = 10
GAMMA_FREQ_RANGE = (60, 100)


def rms(x, axis=-1):
    return np.sqrt(np.mean(x**2, axis=axis))


def apply_cmr(raw):
    median_trace = np.median(raw, axis=0)
    return raw - median_trace[np.newaxis, :]


def detrend(x, nmed):
    """
    Subtract the trend from a vector
    The trend is a median filtered version of the said vector with tapering
    """
    ntap = int(np.ceil(nmed / 2))
    xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]
    xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
    return x - xf


def channels_similarity(raw, nmed=0):
    """
    Computes the similarity based on zero-lag crosscorrelation of each channel with the median trace
    """

    def fxcor(x, y):
        return scipy.fft.irfft(
            scipy.fft.rfft(x) * np.conj(scipy.fft.rfft(y)), n=raw.shape[-1]
        )

    def nxcor(x, ref):
        ref = ref - np.mean(ref)
        apeak = fxcor(ref, ref)[0]
        x = x - np.mean(x, axis=-1)[:, np.newaxis]  # remove dc component
        return fxcor(x, ref)[:, 0] / apeak

    ref = np.median(raw, axis=0)
    xcor = nxcor(raw, ref)

    if nmed > 0:
        xcor = detrend(xcor, nmed) + 1
    return xcor


def process_chunk(
    bin_file_path,
    t0,
    batch_duration,
    spike_threshold,
    apply_cmr_flag,
    debug,
    channel_subset=None,
):
    """
    Worker function to process a single chunk of data.
    """
    # Re-open reader in worker process
    with Reader(bin_file_path) as sr:
        fs = sr.fs

        # Determine which channels to read
        if channel_subset is not None:
            channels_to_read = channel_subset
            nc = len(channels_to_read)
        else:
            # Default: read all AP channels (exclude sync if at end)
            nc_total = sr.nc - sr.nsync
            channels_to_read = slice(0, nc_total)
            nc = nc_total

        # Pre-calculate filter coefficients (same as before)
        sos_hp = scipy.signal.butter(
            N=3, Wn=300 / fs * 2, btype="highpass", output="sos"
        )
        sos_lp = scipy.signal.butter(
            N=3, Wn=LF_CUTOFF_HZ / fs * 2, btype="lowpass", output="sos"
        )

        sl = slice(int(t0 * fs), int((t0 + batch_duration) * fs))

        # Read data
        # sr[slice, channels] returns (ns, nc), we want (nc, ns)
        raw = sr[sl, channels_to_read].T

        raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]

        # --- data streams ---
        # 1. activity stream (strictly cmr + hp)
        raw_cmr = apply_cmr(raw)
        hf_activity = scipy.signal.sosfiltfilt(sos_hp, raw_cmr, axis=1)

        # 3. noise stream (strictly raw)
        hf_noise = scipy.signal.sosfiltfilt(sos_hp, raw, axis=1)

        # 4. feature/plot stream (configurable)
        if apply_cmr_flag:
            raw_features = raw_cmr
            hf_features = hf_activity
        else:
            raw_features = raw
            hf_features = hf_noise

        lf_features = scipy.signal.sosfiltfilt(sos_lp, raw_features, axis=1)

        # --- calculation ---

        # spikes / activity
        mad = np.median(np.abs(hf_activity), axis=1)
        thresholds = spike_threshold * mad

        chunk_spikes = np.zeros(nc)
        chunk_amplitudes = [[] for _ in range(nc)]

        for ch in range(nc):
            # find negative threshold crossings
            crossings = np.where(hf_activity[ch, :] < thresholds[ch])[0]
            if len(crossings) > 0:
                refractory_samples = int(fs / 1000)
                diff = np.diff(crossings)
                valid = np.r_[True, diff > refractory_samples]

                num_valid = np.sum(valid)
                chunk_spikes[ch] = num_valid

                # collect amplitudes
                for idx in crossings[valid]:
                    end_search = min(idx + int(fs / 1000), hf_activity.shape[1])
                    if end_search > idx:
                        amp = np.min(hf_activity[ch, idx:end_search])
                        chunk_amplitudes[ch].append(amp)
                    else:
                        chunk_amplitudes[ch].append(hf_activity[ch, idx])

        current_chunk_median_fr = np.median(chunk_spikes / batch_duration)

        # features
        n_samples = raw.shape[-1]
        fscale, psd_cmr = scipy.signal.welch(raw_cmr * 1e6, fs=fs, nperseg=n_samples)

        mask_gamma = (fscale >= GAMMA_FREQ_RANGE[0]) & (fscale <= GAMMA_FREQ_RANGE[1])
        if np.sum(mask_gamma) > 0:
            power_gamma = np.mean(psd_cmr[:, mask_gamma], axis=-1)
        else:
            power_gamma = np.zeros(nc)

        # noisy/dead detection (using raw / hf_noise)
        xcor_raw = channels_similarity(raw)
        xcor_hf_val = detrend(xcor_raw, 11)

        fscale_raw, psd_raw = scipy.signal.welch(raw * 1e6, fs=fs, nperseg=n_samples)
        psd_hf_threshold = 0.02
        psd_hf_val = np.mean(psd_raw[:, fscale_raw > (fs / 2 * 0.8)], axis=-1)

        # surface detection (using configurable streams)
        xcorf_features = channels_similarity(hf_features)
        xcor_lf_val = xcorf_features - detrend(xcorf_features, 11) - 1

        # mav (activity - strictly preprocessed)
        mean_abs_volt = np.mean(np.abs(hf_activity), axis=-1)

        # Use actual channel indices if available, else 0..nc
        if channel_subset is not None:
            feat_ind = np.array(channel_subset)
        else:
            feat_ind = np.arange(nc)

        chunk_xfeats = {
            "ind": feat_ind,
            "rms_raw": rms(raw_features),
            "rms_lf": rms(lf_features),
            "power_gamma": power_gamma,
            "xcor_hf": xcor_hf_val,
            "xcor_lf": xcor_lf_val,
            "psd_hf": psd_hf_val,
            "mean_abs_volt": mean_abs_volt,
        }

        # detect bad channels (per chunk)
        similarity_threshold = (-0.5, 1)

        idead = np.where(similarity_threshold[0] > chunk_xfeats["xcor_hf"])[0]
        inoisy = np.where(
            np.logical_or(
                chunk_xfeats["psd_hf"] > psd_hf_threshold,
                chunk_xfeats["xcor_hf"] > similarity_threshold[1],
            )
        )[0]

        # outside brain detection (gradient of lf coherence)
        signal_noisy = chunk_xfeats["xcor_lf"]
        window_size = 25
        kernel = np.ones(window_size) / window_size
        signal_filtered = np.convolve(signal_noisy, kernel, mode="same")
        diff_x = np.diff(signal_filtered)
        indx = np.where(diff_x < -0.02)[0]

        ichannels = np.zeros(nc)
        if indx.size > 0:
            indx_threshold = np.floor(np.median(indx)).astype(int)
            threshold_val = signal_noisy[indx_threshold]
            ioutside = np.where(signal_noisy < threshold_val)[0]
            # check contiguity at top
            if ioutside.size > 0 and ioutside[-1] == (nc - 1):
                a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
                ioutside = ioutside[a == np.max(a)]
                ichannels[ioutside] = 3

        ichannels[idead] = 1
        ichannels[inoisy] = 2

        debug_info = None
        if debug:
            debug_info = {"gradient_drops": len(indx) if indx.size > 0 else 0}

        return {
            "chunk_spikes": chunk_spikes,
            "amplitudes": chunk_amplitudes,
            "median_fr": current_chunk_median_fr,
            "xfeats": chunk_xfeats,
            "channel_labels": ichannels,
            "debug_info": debug_info,
            "t0": t0,  # return t0 to identify best chunk later
        }


def analyze_recording(
    bin_file,
    n_batches=20,
    batch_duration=0.4,
    spike_threshold=-5.0,
    apply_cmr_flag=False,
    debug=False,
    time_slice=None,
    channel_subset=None,
):
    """
    Unified analysis function.

    :param time_slice: Optional tuple ("seconds"|"proportion", start, end).
                       Restricts analysis to this time window.
    :param channel_subset: Optional list of absolute channel indices to analyze.
    :return: channel_flags, xfeats_median, raw_best_chunk, fs, firing_rates, spike_amplitudes
    """
    # handle bin_file object or path
    if isinstance(bin_file, Reader):
        # if it's already a reader, we need the path to reopen in workers
        bin_file_path = bin_file.file_bin
        sr = bin_file
        # if we are passed a Reader, we assume it's open, but let's check
        if not sr.is_open:
            sr.open()
    else:
        bin_file_path = Path(bin_file)
        sr = Reader(bin_file_path)

    # Determine NC based on subset
    if channel_subset is not None:
        nc = len(channel_subset)
    else:
        nc = sr.nc - sr.nsync

    fs = sr.fs

    # Calculate n_jobs
    # Estimate memory: ~111MB per job for 0.4s chunk. Scale linearly.
    # Base: 385 ch * 30000 Hz * 0.4s * 4 bytes * 6 copies approx 111MB
    # Scale by channel count if reduced
    ch_ratio = nc / (sr.nc - sr.nsync)
    estimated_mem_per_job = (sr.nc - sr.nsync) * ch_ratio * fs * batch_duration * 4 * 6
    available_mem = psutil.virtual_memory().available
    max_jobs_mem = int(available_mem / estimated_mem_per_job)
    max_jobs_cpu = multiprocessing.cpu_count() - 1

    n_jobs = max(1, min(max_jobs_cpu, max_jobs_mem))
    print(
        f"Parallel processing with {n_jobs} jobs (Mem limit: {max_jobs_mem}, CPU limit: {max_jobs_cpu})"
    )

    xfeats_accumulator = {}

    total_spikes = np.zeros(nc)
    total_duration = 0
    all_amplitudes = [[] for _ in range(nc)]

    channel_labels_all = np.zeros((nc, n_batches))

    max_chunk_median_fr = -1.0
    best_chunk_t0 = -1

    # determine time range
    t_start_abs = 0
    t_end_abs = sr.rl

    if time_slice:
        mode, t1, t2 = time_slice
        if mode == "proportion":
            t_start_abs = t1 * sr.rl
            t_end_abs = t2 * sr.rl
        else:  # seconds
            t_start_abs = t1
            t_end_abs = t2

    t_start_abs = max(0, min(t_start_abs, sr.rl))
    t_end_abs = max(0, min(t_end_abs, sr.rl))

    if t_end_abs <= t_start_abs:
        print(
            f"Warning: Invalid time slice {t_start_abs}-{t_end_abs}. Using full duration."
        )
        t_start_abs = 0
        t_end_abs = sr.rl

    scan_end = max(t_start_abs, t_end_abs - batch_duration)

    print(
        f"Analyzing {n_batches} chunks of {batch_duration}s in range {t_start_abs:.1f}-{t_end_abs:.1f}s..."
    )

    chunk_starts = np.linspace(t_start_abs, scan_end, n_batches)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                process_chunk,
                bin_file_path,
                t0,
                batch_duration,
                spike_threshold,
                apply_cmr_flag,
                debug,
                channel_subset,
            ): i
            for i, t0 in enumerate(chunk_starts)
        }

        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                res = future.result()

                # Aggregate spikes
                total_spikes += res["chunk_spikes"]
                total_duration += batch_duration  # this assumes all chunks succeed

                # Aggregate amplitudes
                for ch in range(nc):
                    all_amplitudes[ch].extend(res["amplitudes"][ch])

                # Collect features
                chunk_xfeats = res["xfeats"]
                if not xfeats_accumulator:
                    for k in chunk_xfeats.keys():
                        xfeats_accumulator[k] = []
                for k, v in chunk_xfeats.items():
                    xfeats_accumulator[k].append(v)

                # Collect labels
                channel_labels_all[:, i] = res["channel_labels"]

                # Check for best chunk
                if res["median_fr"] > max_chunk_median_fr:
                    max_chunk_median_fr = res["median_fr"]
                    best_chunk_t0 = res["t0"]

                if debug:
                    dinfo = res.get("debug_info")
                    if dinfo:
                        drops = dinfo.get("gradient_drops", 0)
                        if drops > 0:
                            print(f"Debug (Chunk {i}):  Gradient drops: {drops}")
                        else:
                            print(f"Debug (Chunk {i}):  No gradient drops.")

            except Exception as e:
                print(f"Chunk {i} processing failed: {e}")

    # Final aggregation
    firing_rates = total_spikes / total_duration

    spike_amplitudes = np.zeros(nc)
    for ch in range(nc):
        if len(all_amplitudes[ch]) > 0:
            spike_amplitudes[ch] = np.median(all_amplitudes[ch])
        else:
            spike_amplitudes[ch] = 0

    if channel_subset is not None:
        final_ind = np.array(channel_subset)
    else:
        final_ind = np.arange(nc)

    xfeats_median = {"ind": final_ind}
    for k, v_list in xfeats_accumulator.items():
        if k != "ind":
            xfeats_median[k] = np.median(np.stack(v_list, axis=1), axis=1)

    channel_flags, _ = scipy.stats.mode(channel_labels_all, axis=1)
    channel_flags = channel_flags.flatten()

    num_outside = np.sum(channel_flags == 3)
    if num_outside > 0:
        print(
            f"Final aggregated detection: {num_outside} channels marked as outside brain"
        )

    # Read best chunk raw data for plotting
    best_chunk_raw = None
    if best_chunk_t0 != -1:
        sl = slice(int(best_chunk_t0 * fs), int((best_chunk_t0 + batch_duration) * fs))

        # Determine read channels again locally
        if channel_subset is not None:
            channels_to_read = channel_subset
        else:
            nc_total = sr.nc - sr.nsync
            channels_to_read = slice(0, nc_total)

        raw = sr[sl, channels_to_read].T
        raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]

        if apply_cmr_flag:
            best_chunk_raw = apply_cmr(raw)
        else:
            best_chunk_raw = raw

    # If we created the reader locally, close it.
    if not isinstance(bin_file, Reader):
        sr.close()

    return (
        channel_flags,
        xfeats_median,
        best_chunk_raw,
        fs,
        firing_rates,
        spike_amplitudes,
    )


def find_surface_channel(channel_labels):
    """
    Find the brain surface channel from the channel labels.

    The surface channel is defined as the lowest channel number (smallest index)
    in the contiguous block of channels labeled as "outside brain" (label 3)
    that are at the top of the probe (highest channel numbers).

    :param channel_labels: numpy array of channel labels (0=good, 1=dead, 2=noisy, 3=outside)
    :return: surface channel index (int), or -1 if no surface detected
    """
    nc = len(channel_labels)

    # find all channels labeled as "outside brain"
    outside_channels = np.where(channel_labels == 3)[0]

    if len(outside_channels) == 0:
        return -1

    # check if the highest channel is in the outside group
    if outside_channels[-1] != nc - 1:
        return -1

    # find the contiguous block at the top
    top_block = [outside_channels[-1]]
    for i in range(len(outside_channels) - 2, -1, -1):
        if outside_channels[i] == top_block[0] - 1:
            top_block.insert(0, outside_channels[i])
        else:
            break

    return top_block[0]


def get_shank_info(geometry):
    """
    Extract shank information from probe geometry.

    :param geometry: Geometry dictionary from Reader.geometry
    :return: Dictionary with shank info: {shank_id: [channel_indices]}
    """
    if geometry is None or geometry.get("shank") is None:
        return None

    shank_ids = np.unique(geometry["shank"])
    shank_channels = {}

    for shank_id in shank_ids:
        channels = np.where(geometry["shank"] == shank_id)[0]
        shank_channels[int(shank_id)] = channels

    return shank_channels


def filter_data_by_shank(raw, channel_labels, xfeats, shank_channels):
    """
    Filter raw data, labels, and features to only include channels from a specific shank.

    :param raw: Raw data array [nc, ns]
    :param channel_labels: Channel labels array [nc]
    :param xfeats: Dictionary of features
    :param shank_channels: Array of channel indices for this shank
    :return: Filtered raw, channel_labels, xfeats
    """
    raw_shank = raw[shank_channels, :]
    labels_shank = channel_labels[shank_channels]

    xfeats_shank = {}
    for key, val in xfeats.items():
        if key == "ind":
            xfeats_shank[key] = val[shank_channels]
        else:
            xfeats_shank[key] = val[shank_channels]

    return raw_shank, labels_shank, xfeats_shank
