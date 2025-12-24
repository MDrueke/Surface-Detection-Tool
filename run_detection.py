import numpy as np
import scipy.signal
import scipy.stats

from reader import Reader


def rms(x, axis=-1):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x**2, axis=axis))


def apply_cmr(raw):
    """
    Apply Common Median Referencing to raw data.

    For each time sample, subtract the median across all channels.

    :param raw: Raw data array [nc, ns]
    :return: CMR-applied data [nc, ns]
    """
    median_trace = np.median(
        raw, axis=0
    )  # [ns] - median across channels at each time point
    return raw - median_trace[np.newaxis, :]  # subtract from all channels


def detect_bad_channels(
    raw,
    fs,
    similarity_threshold=(-0.5, 1),
    psd_hf_threshold=None,
    display=False,
    apply_cmr_flag=False,
    debug=False,
    verbose=True,
):
    """
    Bad channels detection for Neuropixel probes
    Labels channels
     0: all clear
     1: dead low coherence / amplitude
     2: noisy
     3: outside of the brain
    :param raw: [nc, ns]
    :param fs: sampling frequency
    :param similarity_threshold:
    :param psd_hf_threshold:
    :param display: optinal (False) will show a plot of features alongside a raw data snippet
    :return: labels (numpy vector [nc]), xfeats: dictionary of features [nc]
    """

    # def detect_and_fix_flipped_lf_coherence(xcor_lf, nc):
    #     """
    #     Detect if LF coherence is flipped (due to tip reference instead of external reference).
    #
    #     Normal: xcor_lf is HIGH at bottom (brain), LOW at top (outside brain)
    #     Flipped: xcor_lf is LOW at bottom, HIGH at top
    #
    #     If flipped, return -1 * xcor_lf to correct it.
    #
    #     :param xcor_lf: LF coherence feature array [nc]
    #     :param nc: number of channels
    #     :return: corrected xcor_lf
    #     """
    #     # Calculate medians of bottom 20% and top 20% of channels
    #     bottom_20_idx = int(nc * 0.2)
    #     top_20_idx = int(nc * 0.8)
    #
    #     median_bottom = np.median(xcor_lf[:bottom_20_idx])
    #     median_top = np.median(xcor_lf[top_20_idx:])
    #
    #     # If top > bottom, the signal is flipped
    #     if median_top > median_bottom:
    #         print("Detected flipped LF coherence (likely tip reference). Correcting...")
    #         return -1 * xcor_lf
    #
    #     return xcor_lf

    def detrend(x, nmed):
        """
        Subtract the trend from a vector
        The trend is a median filtered version of the said vector with tapering
        :param x: input vector
        :param nmed: number of points of the median filter
        :return: np.array
        """
        ntap = int(np.ceil(nmed / 2))
        xf = np.r_[np.zeros(ntap) + x[0], x, np.zeros(ntap) + x[-1]]
        # assert np.all(xcorf[ntap:-ntap] == xcor)
        xf = scipy.signal.medfilt(xf, nmed)[ntap:-ntap]
        return x - xf

    def channels_similarity(raw, nmed=0):
        """
        Computes the similarity based on zero-lag crosscorrelation of each channel with the median
        trace referencing
        :param raw: [nc, ns]
        :param nmed:
        :return:
        """

        def fxcor(x, y):
            return scipy.fft.irfft(
                scipy.fft.rfft(x) * np.conj(scipy.fft.rfft(y)), n=raw.shape[-1]
            )

        def nxcor(x, ref):
            ref = ref - np.mean(ref)
            apeak = fxcor(ref, ref)[0]
            x = x - np.mean(x, axis=-1)[:, np.newaxis]  # remove DC component
            return fxcor(x, ref)[:, 0] / apeak

        ref = np.median(raw, axis=0)
        xcor = nxcor(raw, ref)

        if nmed > 0:
            xcor = detrend(xcor, nmed) + 1
        return xcor

    nc, _ = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    # Apply CMR if requested
    if apply_cmr_flag:
        raw = apply_cmr(raw)
    xcor = channels_similarity(raw)
    # Compute PSD using Welch's method
    # Use nperseg equal to the number of samples (or a large enough value) 
    # to maintain sufficient frequency resolution. Default nperseg=256 gives ~117Hz res at 30kHz!
    n_samples = raw.shape[-1]
    fscale, psd = scipy.signal.welch(raw * 1e6, fs=fs, nperseg=n_samples)  # units; uV ** 2 / Hz
    # auto-detection of the band with which we are working
    band = "ap" if fs > 2600 else "lf"
    # the LFP band data is obviously much stronger so auto-adjust the default threshold
    if band == "ap":
        psd_hf_threshold = 0.02 if psd_hf_threshold is None else psd_hf_threshold
        filter_kwargs = {"N": 3, "Wn": 300 / fs * 2, "btype": "highpass"}
    elif band == "lf":
        psd_hf_threshold = 1.4 if psd_hf_threshold is None else psd_hf_threshold
        psd_hf_threshold = 1.4 if psd_hf_threshold is None else psd_hf_threshold
        filter_kwargs = {"N": 3, "Wn": 1 / fs * 2, "btype": "highpass"}
    sos_hp = scipy.signal.butter(**filter_kwargs, output="sos")
    hf = scipy.signal.sosfiltfilt(sos_hp, raw)
    xcorf = channels_similarity(hf)

    # LF Power (0-100 Hz)
    sos_lp = scipy.signal.butter(N=3, Wn=100 / fs * 2, btype="lowpass", output="sos")
    lf = scipy.signal.sosfiltfilt(sos_lp, raw)
    rms_lf = rms(lf)

    # Gamma / HF Power (40-100 Hz)
    # psd is [nc, n_freqs], fscale is [n_freqs]
    # mask for 40-100 Hz
    mask_gamma = (fscale >= 40) & (fscale <= 100)
    # Mean power in gamma band
    if np.sum(mask_gamma) > 0:
        power_gamma = np.mean(psd[:, mask_gamma], axis=-1)
    else:
        power_gamma = np.zeros(nc)

    xfeats = {
        "ind": np.arange(nc),
        "rms_raw": rms(raw),  # very similar to the rms avfter butterworth filter
        "rms_lf": rms_lf,
        "power_gamma": power_gamma,
        "xcor_hf": detrend(xcor, 11),
        "xcor_lf": xcorf - detrend(xcorf, 11) - 1,
        "psd_hf": np.mean(psd[:, fscale > (fs / 2 * 0.8)], axis=-1),  # 80% nyquists
    }

    # Detect and correct flipped LF coherence (due to tip reference)
    # NOTE: This detection doesn't work as expected - commented out for now
    # xfeats["xcor_lf"] = detect_and_fix_flipped_lf_coherence(xfeats["xcor_lf"], nc)

    # make recommendation
    ichannels = np.zeros(nc)
    idead = np.where(similarity_threshold[0] > xfeats["xcor_hf"])[0]
    inoisy = np.where(
        np.logical_or(
            xfeats["psd_hf"] > psd_hf_threshold,
            xfeats["xcor_hf"] > similarity_threshold[1],
        )
    )[0]
    # the channels outside of the brains are the contiguous channels below the threshold on the trend coherency

    signal_noisy = xfeats["xcor_lf"]
    # Filter signal
    window_size = 25  # Choose based on desired smoothing (e.g., 25 samples)
    kernel = np.ones(window_size) / window_size
    # Apply convolution
    signal_filtered = np.convolve(signal_noisy, kernel, mode="same")

    diff_x = np.diff(signal_filtered)
    indx = np.where(diff_x < -0.02)[0]  # hardcoded threshold
    if indx.size > 0:
        indx_threshold = np.floor(np.median(indx)).astype(int)
        threshold = signal_noisy[indx_threshold]
        ioutside = np.where(signal_noisy < threshold)[0]
        if verbose:
            print(
                f"LF coherence threshold computed: {threshold:.4f} at channel {indx_threshold}"
            )
        if debug:
            print(
                f"  Found {len(indx)} channels with sharp gradient drops (diff < -0.02)"
            )
            print(f"  Gradient drop channels: {indx}")
            print(f"  Using median index: {indx_threshold}")
    else:
        ioutside = np.array([])
        threshold = None
        if verbose:
            print(
                "LF coherence threshold: No sharp gradient drops detected (no surface found)"
            )
        if debug:
            print(f"  Min gradient value: {np.min(diff_x):.4f}")
            print(f"  Gradient threshold used: -0.02")

    if ioutside.size > 0 and ioutside[-1] == (nc - 1):
        a = np.cumsum(np.r_[0, np.diff(ioutside) - 1])
        ioutside = ioutside[a == np.max(a)]
        ichannels[ioutside] = 3

    # indices
    ichannels[idead] = 1
    ichannels[inoisy] = 2

    # Store debug information if requested
    if debug:
        xfeats["debug"] = {
            "signal_noisy": signal_noisy,
            "signal_filtered": signal_filtered,
            "diff_x": diff_x,
            "gradient_threshold": -0.02,
            "lf_threshold": threshold,
            "lf_threshold_channel": indx_threshold if indx.size > 0 else None,
            "gradient_drop_indices": indx,
        }

    return ichannels, xfeats


def compute_firing_rates(sr, n_batches=20, batch_duration=0.4, threshold=-5.0):
    """
    Compute firing rates for all channels across multiple time chunks.
    Always applies highpass filtering and CMR for spike detection.

    :param sr: Reader object
    :param n_batches: number of time chunks to analyze
    :param batch_duration: duration of each chunk in seconds
    :param threshold: spike detection threshold in multiples of MAD (default: -5.0)
    :return: firing_rates: array of firing rates (spikes/sec) per channel [nc]
             spike_amplitudes: array of median spike amplitudes (uV/mad-units) per channel [nc]
    """
    nc = sr.nc - sr.nsync
    total_spikes = np.zeros(nc)
    total_duration = 0
    
    # Store all amplitudes to compute median later
    # List of lists? or just accumulate and take median at the end?
    # Since we process in batches, we can't easily take global median without storing all spikes.
    # Storing all spikes might be heavy if very long recording.
    # Compromise: Store list of amplitudes per channel
    all_amplitudes = [[] for _ in range(nc)]

    # Highpass filter for spike detection (300 Hz for AP band)
    fs = sr.fs
    sos_hp = scipy.signal.butter(N=3, Wn=300 / fs * 2, btype="highpass", output="sos")

    for i, t0 in enumerate(np.linspace(0, sr.rl - batch_duration, n_batches)):
        sl = slice(int(t0 * fs), int((t0 + batch_duration) * fs))
        raw = sr[sl, :nc].T  # [nc, ns]

        # Remove DC offset
        raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]

        # Apply CMR
        raw = apply_cmr(raw)

        # Apply highpass filter
        raw_hp = scipy.signal.sosfiltfilt(sos_hp, raw, axis=1)

        # Compute threshold per channel: threshold * MAD (Median Absolute Deviation)
        # MAD is robust to outliers
        mad = np.median(np.abs(raw_hp), axis=1)
        thresholds = threshold * mad

        # Count threshold crossings (spikes)
        for ch in range(nc):
            # Find negative threshold crossings
            crossings = np.where(raw_hp[ch, :] < thresholds[ch])[0]
            if len(crossings) > 0:
                # Remove duplicates within refractory period (1 ms = fs/1000 samples)
                refractory_samples = int(fs / 1000)
                diff = np.diff(crossings)
                valid = np.r_[True, diff > refractory_samples]
                total_spikes[ch] += np.sum(valid)
                
                # Collect amplitudes of valid spikes
                # crossings[valid] gives indices
                # We want the amplitude at the crossing (or better, the peak near crossing?)
                # For simplicity and speed, let's take the value at the crossing (which is < threshold)
                # Or find the minimum in a small window.
                # Let's just take the value at crossing or the minimum in next few samples.
                # Simple approach: Minimum value in 1ms window after crossing
                for idx in crossings[valid]:
                     # Check bounds
                     end_search = min(idx + int(fs/1000), raw_hp.shape[1])
                     if end_search > idx:
                         # Spikes are negative
                         amp = np.min(raw_hp[ch, idx:end_search])
                         all_amplitudes[ch].append(amp)
                     else:
                         all_amplitudes[ch].append(raw_hp[ch, idx])

        total_duration += batch_duration

    # Compute firing rates (spikes per second)
    firing_rates = total_spikes / total_duration
    
    # Compute median spike amplitude
    spike_amplitudes = np.zeros(nc)
    for ch in range(nc):
        if len(all_amplitudes[ch]) > 0:
            spike_amplitudes[ch] = np.median(all_amplitudes[ch])
        else:
            spike_amplitudes[ch] = 0 # No spikes

    return firing_rates, spike_amplitudes


def detect_bad_channels_cbin(
    bin_file,
    n_batches=20,
    batch_duration=0.4,
    display=False,
    apply_cmr_flag=False,
    debug=False,
):
    """
    Runs a ap-binary file scan to automatically detect faulty channels
    :param bin_file: full file path to the binary or compressed binary file from spikeglx
    :param n_batches: number of batches throughout the file (defaults to 20)
    :param batch_duration: batch length in seconds, defaults to 0.3
    :param display: if True will return a figure with features and an excerpt of the raw data
    :param apply_cmr_flag: if True, apply Common Median Referencing to the data
    :return: channel_labels: nc int array with 0:ok, 1:dead, 2:high noise, 3:outside of the brain
    """
    sr = bin_file if isinstance(bin_file, Reader) else Reader(bin_file)
    nc = sr.nc - sr.nsync
    channel_labels = np.zeros((nc, n_batches))
    # loop over the file and take the mode of detections
    for i, t0 in enumerate(np.linspace(0, sr.rl - batch_duration, n_batches)):
        sl = slice(int(t0 * sr.fs), int((t0 + batch_duration) * sr.fs))
        raw = sr[sl, :nc].T
        # Only pass debug=True and verbose=True on the last batch to avoid spamming console
        channel_labels[:, i], _xfeats = detect_bad_channels(
            raw,
            fs=sr.fs,
            apply_cmr_flag=apply_cmr_flag,
            debug=(debug and i == n_batches - 1),
            verbose=(i == n_batches - 1),  # Only print threshold on last batch
        )

    # Aggregate the labels for robust detection
    channel_flags, _ = scipy.stats.mode(channel_labels, axis=1)

    # Flatten to 1D array (mode returns shape (nc, 1))
    channel_flags = channel_flags.flatten()

    # Print summary of aggregated detection
    num_outside = np.sum(channel_flags == 3)
    if num_outside > 0:
        print(
            f"Final aggregated detection: {num_outside} channels marked as outside brain"
        )

    # Apply CMR to the raw data for plotting if requested
    # (The raw variable here is from the last chunk iteration)
    if apply_cmr_flag:
        # Remove DC offset first (same as in detect_bad_channels)
        raw_plot = raw - np.mean(raw, axis=-1)[:, np.newaxis]
        # Then apply CMR
        raw_plot = apply_cmr(raw_plot)
    else:
        raw_plot = raw

    # Return the robust labels, but the features and raw data from the LAST chunk for plotting
    return channel_flags, _xfeats, raw_plot, sr.fs


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

    # Find all channels labeled as "outside brain"
    outside_channels = np.where(channel_labels == 3)[0]

    if len(outside_channels) == 0:
        # No surface detected - return -1
        return -1

    # Check if the highest channel is in the outside group
    # (ensures we're looking at the top block, not a disconnected block lower down)
    if outside_channels[-1] != nc - 1:
        # The outside channels don't extend to the top - might be a disconnected block
        # Return -1 as fallback
        return -1

    # Find the contiguous block at the top by checking for gaps
    # Start from the highest channel and work down
    top_block = [outside_channels[-1]]
    for i in range(len(outside_channels) - 2, -1, -1):
        if outside_channels[i] == top_block[0] - 1:
            top_block.insert(0, outside_channels[i])
        else:
            # Found a gap - stop here
            break

    # Return the lowest channel in the top block (the brain surface)
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
        # Get indices of channels belonging to this shank
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
    # Filter raw data
    raw_shank = raw[shank_channels, :]

    # Filter channel labels
    labels_shank = channel_labels[shank_channels]

    # Filter features
    xfeats_shank = {}
    for key, val in xfeats.items():
        if key == "ind":
            # Keep original indices for reference
            xfeats_shank[key] = val[shank_channels]
        else:
            xfeats_shank[key] = val[shank_channels]

    return raw_shank, labels_shank, xfeats_shank
