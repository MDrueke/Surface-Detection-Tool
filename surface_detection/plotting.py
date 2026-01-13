import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d

from .run_detection import apply_cmr

AP_RANGE_UV = 40
LF_RANGE_UV = 250
SURFACE_COLOR = "#b6d56b"
SMOOTHING_SIGMA = 2.0


def apply_highpass_filter(raw, fs, cutoff_hz):
    """
    Apply a Butterworth highpass filter to raw data.

    :param raw: Raw data array [nc, ns]
    :param fs: Sampling frequency in Hz
    :param cutoff_hz: Highpass cutoff frequency in Hz
    :return: Filtered data [nc, ns]
    """
    # validate cutoff is below nyquist frequency
    nyquist = fs / 2
    if cutoff_hz >= nyquist:
        print(
            f"\033[91mWarning: Highpass cutoff ({cutoff_hz} Hz) >= Nyquist frequency ({nyquist} Hz). Filter disabled.\033[0m"
        )
        return raw

    sos = scipy.signal.butter(
        N=3, Wn=cutoff_hz / nyquist, btype="highpass", output="sos"
    )

    filtered = scipy.signal.sosfiltfilt(sos, raw, axis=1)

    return filtered


nature_style = {
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.facecolor": "#545454",
    "axes.grid": False,
    "axes.labelcolor": "white",
    "axes.labelsize": 12,
    "axes.linewidth": 1,
    "axes.titlecolor": "white",
    "axes.titlesize": 13,
    "legend.labelcolor": "white",
    "figure.facecolor": "#323232",
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "grid.color": "#505050",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.color": "white",
    "xtick.direction": "in",
    "xtick.labelsize": 12,
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "xtick.minor.size": 3,
    "xtick.minor.width": 0.5,
    "ytick.color": "white",
    "ytick.direction": "in",
    "ytick.labelsize": 12,
    "ytick.major.size": 5,
    "ytick.major.width": 1,
    "ytick.minor.size": 3,
    "ytick.minor.width": 0.5,
}

plt.rcParams.update(nature_style)


def show_channels_labels(
    raw,
    fs,
    channel_labels,
    xfeats,
    similarity_threshold=(-0.5, 1),
    psd_hf_threshold=0.02,
):
    """
    Shows the features side by side a snippet of raw data
    """
    nc, ns = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes dc offset
    ns_plot = np.minimum(ns, 3000)
    fig, ax = plt.subplots(
        1, 5, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 8, 0.2]}
    )
    ax[0].plot(xfeats["xcor_hf"], np.arange(nc))
    ax[0].plot(
        xfeats["xcor_hf"][(iko := channel_labels == 1)], np.arange(nc)[iko], "k*"
    )
    ax[0].plot(
        xfeats["xcor_hf"][
            (iko := np.where(xfeats["xcor_hf"] > similarity_threshold[1]))
        ],
        np.arange(nc)[iko],
        "r*",
    )
    ax[0].plot(similarity_threshold[0] * np.ones(2), [0, nc], "--", color="gray")
    ax[0].plot(similarity_threshold[1] * np.ones(2), [0, nc], "--", color="gray")
    ax[0].set(
        ylabel="Channel #",
        xlabel="HF\ncoherence",
        ylim=[0, nc],
        title="dead\nchannels",
    )
    ax[1].plot(xfeats["psd_hf"], np.arange(nc), "w-")
    ax[1].plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        np.arange(nc)[iko],
        "*",
        color="#d5806b",
        markersize=8,
    )
    ax[1].plot(psd_hf_threshold * np.array([1, 1]), [0, nc], "--", color="gray")
    ax[1].set(ylabel="", xlabel="HF\npower", ylim=[0, nc], title="noisy\nchannels")
    ax[1].tick_params(labelleft=False)
    ax[1].sharey(ax[0])
    ax[2].plot(xfeats["xcor_lf"], np.arange(nc), "w-")
    ax[2].plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)],
        np.arange(nc)[iko],
        "*",
        color="#b6d56b",
        markersize=8,
    )
    ax[2].set(ylabel="", xlabel="LF\ncoherence", ylim=[0, nc], title="outside")
    ax[2].tick_params(labelleft=False)
    ax[2].sharey(ax[0])
    voltageshow(raw[:, :ns_plot], fs, ax=ax[3], cax=ax[4])
    ax[3].sharey(ax[0])
    fig.tight_layout(pad=0)
    return fig, ax


def voltageshow(
    raw,
    fs,
    cmap="PuOr",
    ax=None,
    cax=None,
    cbar_label="Voltage (uV)",
    scaling=1e6,
    vrange=None,
    extent=None,
    **axis_kwargs,
):
    """
    Visualizes electrophysiological voltage data as a heatmap.

    This function displays raw voltage data as a color-coded image with appropriate
    scaling based on the sampling frequency. It automatically selects voltage range
    based on whether the data is low-frequency (LF) or action potential (AP) data.
    """
    if ax is None:
        fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
        ax, cax = axs
    nc, ns = raw.shape
    default_vrange = LF_RANGE_UV if fs < 2600 else AP_RANGE_UV
    vrange = vrange if vrange is not None else default_vrange

    if extent is None:
        extent = [0, ns / fs, 0, nc]

    im = ax.imshow(
        raw * scaling,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=-vrange,
        vmax=vrange,
        extent=extent,
    )
    # set the axis properties: we use defaults values that can be overridden by user-provided ones
    axis_kwargs = dict(ylim=[0, nc], xlabel="Time (s)", ylabel="Channel") | axis_kwargs
    ax.set(**axis_kwargs)
    ax.grid(False)
    if cax is not None:
        plt.colorbar(im, cax=cax, shrink=0.8).ax.set(ylabel=cbar_label)

    return im


def show_channels_labels_interactive(
    raw,
    fs,
    channel_labels,
    xfeats,
    auto_surface_channel,
    bin_path,
    firing_rates=None,
    spike_amplitudes=None,
    shank_id=None,
    total_shanks=None,
    shank_channels=None,
    similarity_threshold=(-0.5, 1),
    psd_hf_threshold=0.02,
    hf_cutoff=None,
    channel_range_suffix="",
):
    """
    Interactive version of show_channels_labels that allows user to manually select
    the surface channel by clicking on any subplot.

    :param raw: Raw data [nc, ns]
    :param fs: Sampling frequency
    :param channel_labels: Channel labels array
    :param xfeats: Dictionary of features. Must contain 'ind' with real channel indices.
    :param auto_surface_channel: Auto-detected surface channel (absolute index)
    :param bin_path: Path to the bin file (for window title)
    :param firing_rates: Optional array of firing rates [nc]
    :param spike_amplitudes: Optional array of spike amplitudes [nc]
    :param shank_id: Optional shank ID for multi-shank probes
    :param total_shanks: Optional total number of shanks
    :param shank_channels: Optional array of absolute channel indices for this shank (for multi-shank)
    :param similarity_threshold: Tuple of thresholds for dead/noisy detection
    :param psd_hf_threshold: Threshold for high-frequency PSD
    :param hf_cutoff: Optional highpass cutoff frequency (Hz) for heatmap visualization
    :param channel_range_suffix: Suffix to append to saved filename if filtered
    :return: Final surface channel selected (absolute index), or None if cancelled
    """
    nc, ns = raw.shape

    # Retrieve actual channel indices from xfeats, or default to 0..nc
    channel_indices = xfeats.get("ind", np.arange(nc))

    raw_original = raw.copy()
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes dc offset

    ns_plot = np.minimum(ns, 3000)

    # pre-compute all 4 permutations of the data for interactive switching
    raw_variants = {}
    raw_base = raw[:, :ns_plot]

    # 1. no processing
    raw_variants["none"] = raw_base.copy()

    # 2. cmr only
    raw_cmr = apply_cmr(raw_base)
    raw_variants["cmr"] = raw_cmr

    # 3. highpass only (300 hz)
    raw_hp = apply_highpass_filter(raw_base, fs, 300)
    raw_variants["hp"] = raw_hp

    # 4. cmr + highpass
    raw_cmr_hp = apply_cmr(raw_base)
    raw_cmr_hp = apply_highpass_filter(raw_cmr_hp, fs, 300)
    raw_variants["cmr_hp"] = raw_cmr_hp

    # start with cmr + highpass as defaults
    current_raw = raw_variants["cmr_hp"]

    # convert absolute surface channel to relative position for plotting
    # We need the relative index within the `channel_indices` array
    # If using subset, `auto_surface_channel` (absolute) needs to be mapped to row index.

    auto_surface_rel = -1
    if auto_surface_channel != -1:
        # Check if the auto surface channel is even in our current view
        matches = np.where(channel_indices == auto_surface_channel)[0]
        if len(matches) > 0:
            auto_surface_rel = matches[0]

    fig = plt.figure(figsize=(13, 9))

    # layout strategy using nested gridspec
    # outer grid: 2 rows (checkboxes, main content)
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[0.05, 1.9], hspace=0.05)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.02)

    # inner grid: 3 rows (heatmap, features, buttons)
    gs_inner = gs_outer[1].subgridspec(3, 1, height_ratios=[0.8, 1.0, 0.08], hspace=0.5)

    # --- row 1: heatmap ---
    gs_heatmap = gs_inner[0].subgridspec(1, 2, width_ratios=[1, 0.02], wspace=0.05)
    ax_heatmap = fig.add_subplot(gs_heatmap[0, 0])
    ax_cbar = fig.add_subplot(gs_heatmap[0, 1])

    # --- row 2: features ---
    gs_features = gs_inner[1].subgridspec(1, 7, wspace=0.1)

    ax_feats = [fig.add_subplot(gs_features[0, i]) for i in range(7)]

    ax_hf_coh = ax_feats[0]
    ax_psd_hf = ax_feats[1]
    ax_lf_coh = ax_feats[2]
    ax_fr = ax_feats[3]
    ax_lf_pow = ax_feats[4]
    ax_gamma_pow = ax_feats[5]
    ax_sp_amp = ax_feats[6]

    all_plot_axes = [ax_heatmap] + ax_feats
    all_feature_axes = ax_feats

    # --- row 0: checkboxes ---
    gs_checkbox = gs_outer[0].subgridspec(1, 4, width_ratios=[1, 0.1, 2, 1])
    gs_check_inner = gs_outer[0].subgridspec(1, 2, wspace=0.1)

    checkbox_ax1 = fig.add_subplot(gs_check_inner[0, 0])
    checkbox_ax2 = fig.add_subplot(gs_check_inner[0, 1])
    checkbox_ax1.axis("off")
    checkbox_ax2.axis("off")

    checkbox_labels = ["  Common Median Reference", "  Highpass (300 Hz)"]
    checkbox_states = [True, True]
    checkbox_axes = [checkbox_ax1, checkbox_ax2]

    checkbox_rects = []
    checkbox_texts = []
    checkbox_x_marks = []

    for i, (label, ax_check) in enumerate(zip(checkbox_labels, checkbox_axes)):
        # align checkboxes
        if i == 0:
            x_pos = 0.5
        else:
            x_pos = 0.1

        y_pos = 0.5
        box_size = 0.8

        x_pos = 0.1
        y_pos = 0.15

        rect = Rectangle(
            (x_pos, y_pos),
            box_size,
            box_size,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
        )
        ax_check.add_patch(rect)
        checkbox_rects.append(rect)

        text = ax_check.text(
            x_pos + box_size + 0.05,
            y_pos + box_size / 2,
            label,
            verticalalignment="center",
            fontsize=12,
            color="white",
        )
        checkbox_texts.append(text)

        # visible by default
        x_mark_1 = ax_check.plot(
            [x_pos, x_pos + box_size],
            [y_pos, y_pos + box_size],
            "w-",
            linewidth=2,
            visible=True,
        )[0]
        x_mark_2 = ax_check.plot(
            [x_pos + box_size, x_pos],
            [y_pos, y_pos + box_size],
            "w-",
            linewidth=2,
            visible=True,
        )[0]
        checkbox_x_marks.append((x_mark_1, x_mark_2))

        ax_check.set_xlim(0, 4)
        ax_check.set_ylim(0, 1)
        ax_check.set_aspect("equal")

    # set window title
    if shank_id is not None and total_shanks is not None:
        window_title = (
            f"Surface Channel Detection - Shank {shank_id + 1}/{total_shanks}"
        )
    else:
        window_title = "Surface Channel Detection"
    fig.canvas.manager.set_window_title(window_title)

    # --- plot features ---

    # hf coherence (dead)
    ax_hf_coh.plot(xfeats["xcor_hf"], channel_indices, "w-")
    ax_hf_coh.plot(
        xfeats["xcor_hf"][(iko := channel_labels == 1)],
        channel_indices[iko],
        "*",
        color="#75a1d2",
        markersize=8,
    )
    # The thresholds are vertical lines, y-axis is channel indices
    ax_hf_coh.plot(
        similarity_threshold[0] * np.ones(2),
        [channel_indices[0], channel_indices[-1]],
        "--",
        color="gray",
    )
    ax_hf_coh.plot(
        similarity_threshold[1] * np.ones(2),
        [channel_indices[0], channel_indices[-1]],
        "--",
        color="gray",
    )

    ax_hf_coh.set(
        xlabel="HF\ncoherence",
        ylim=[channel_indices[0], channel_indices[-1]],
        title="Dead Channels",
    )
    theme_color = "#75a1d2"
    ax_hf_coh.tick_params(axis="x", colors=theme_color, labelsize=8.4)
    ax_hf_coh.xaxis.label.set_color(theme_color)
    ax_hf_coh.title.set_color(theme_color)

    # psd hf (noisy)
    ax_psd_hf.plot(xfeats["psd_hf"], channel_indices, "w-")
    ax_psd_hf.plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        channel_indices[iko],
        "*",
        color="#d5806b",
        markersize=8,
    )
    ax_psd_hf.plot(
        psd_hf_threshold * np.array([1, 1]),
        [channel_indices[0], channel_indices[-1]],
        "--",
        color="gray",
    )

    ax_psd_hf.set(
        xlabel="HF\npower",
        ylim=[channel_indices[0], channel_indices[-1]],
        title="Noisy Channels",
    )
    theme_color = "#d5806b"
    ax_psd_hf.tick_params(axis="x", colors=theme_color, labelsize=8.4)
    ax_psd_hf.xaxis.label.set_color(theme_color)
    ax_psd_hf.title.set_color(theme_color)

    ax_psd_hf.sharey(ax_hf_coh)
    ax_psd_hf.tick_params(labelleft=False)

    # lf coherence (outside)
    ax_lf_coh.plot(xfeats["xcor_lf"], channel_indices, "w-")
    ax_lf_coh.plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)],
        channel_indices[iko],
        "*",
        color="#b6d56b",
        markersize=8,
    )

    ax_lf_coh.set(
        xlabel="LF\ncoherence",
        ylim=[channel_indices[0], channel_indices[-1]],
        title="Outside Brain",
    )
    theme_color = "#b6d56b"
    ax_lf_coh.tick_params(axis="x", colors=theme_color, labelsize=8.4)
    ax_lf_coh.xaxis.label.set_color(theme_color)
    ax_lf_coh.title.set_color(theme_color)

    ax_lf_coh.sharey(ax_hf_coh)
    ax_lf_coh.tick_params(labelleft=False)

    # mav (mean absolute voltage) - replaces lf power
    if "mean_abs_volt" in xfeats:
        y_vals = channel_indices
        x_vals = xfeats["mean_abs_volt"] * 1e6
        # smoothing
        x_vals = gaussian_filter1d(x_vals, sigma=SMOOTHING_SIGMA)

        ax_lf_pow.plot(x_vals, y_vals, "w-")

        # fill from min and set limits tight to min/max to remove padding gap
        xmin, xmax = np.min(x_vals), np.max(x_vals)
        ax_lf_pow.fill_betweenx(y_vals, xmin, x_vals, color="white", alpha=0.3)
        ax_lf_pow.set_xlim(xmin, xmax)

        ax_lf_pow.set(
            xlabel="Mean Absolute\nVoltage (uV)",
            ylim=[channel_indices[0], channel_indices[-1]],
        )
    elif "rms_lf" in xfeats:
        y_vals = channel_indices
        x_vals = xfeats["rms_lf"] * 1e6
        # smoothing
        x_vals = gaussian_filter1d(x_vals, sigma=SMOOTHING_SIGMA)

        ax_lf_pow.plot(x_vals, y_vals, "w-")
        ax_lf_pow.set_xlim(left=0)
        ax_lf_pow.fill_betweenx(y_vals, 0, x_vals, color="white", alpha=0.3)
        ax_lf_pow.set(
            xlabel="LF Power\n(uV)", ylim=[channel_indices[0], channel_indices[-1]]
        )
    else:
        ax_lf_pow.text(0.5, 0.5, "N/A", color="white", ha="center")
    ax_lf_pow.tick_params(axis="x", labelsize=8.4, colors="white")
    ax_lf_pow.sharey(ax_hf_coh)
    ax_lf_pow.tick_params(labelleft=False)

    # gamma power
    if "power_gamma" in xfeats:
        y_vals = channel_indices
        x_vals = xfeats["power_gamma"]
        # smoothing
        x_vals = gaussian_filter1d(x_vals, sigma=SMOOTHING_SIGMA)

        ax_gamma_pow.plot(x_vals, y_vals, "w-")

        # fill from min and set limits tight to min/max
        xmin, xmax = np.min(x_vals), np.max(x_vals)
        ax_gamma_pow.fill_betweenx(y_vals, xmin, x_vals, color="white", alpha=0.3)
        ax_gamma_pow.set_xlim(xmin, xmax)

        ax_gamma_pow.set(
            xlabel="Gamma Power\n(uV²/Hz)",
            ylim=[channel_indices[0], channel_indices[-1]],
        )
    else:
        ax_gamma_pow.text(0.5, 0.5, "N/A", color="white", ha="center")
    ax_gamma_pow.tick_params(axis="x", labelsize=8.4, colors="white")
    ax_gamma_pow.sharey(ax_hf_coh)
    ax_gamma_pow.tick_params(labelleft=False)

    # spike amplitude
    if spike_amplitudes is not None:
        y_vals = channel_indices
        # flip sign (extracellular is negative, we want positive magnitude)
        x_vals = -1 * spike_amplitudes * 1e6
        # smoothing
        x_vals = gaussian_filter1d(x_vals, sigma=SMOOTHING_SIGMA)

        ax_sp_amp.plot(x_vals, y_vals, "w-")

        # fill from min and set limits tight to min/max
        xmin, xmax = np.min(x_vals), np.max(x_vals)
        ax_sp_amp.fill_betweenx(y_vals, xmin, x_vals, color="white", alpha=0.3)
        ax_sp_amp.set_xlim(xmin, xmax)

        ax_sp_amp.set(
            xlabel="Spike Amp\n(uV)", ylim=[channel_indices[0], channel_indices[-1]]
        )
    else:
        ax_sp_amp.text(0.5, 0.5, "N/A", color="white", ha="center")
    ax_sp_amp.tick_params(axis="x", labelsize=8.4, colors="white")
    ax_sp_amp.sharey(ax_hf_coh)
    ax_sp_amp.tick_params(labelleft=False)

    # firing rate
    if firing_rates is not None:
        y_vals = channel_indices
        x_vals = firing_rates
        # smoothing
        x_vals = gaussian_filter1d(x_vals, sigma=SMOOTHING_SIGMA)

        ax_fr.plot(x_vals, y_vals, "w-", linewidth=1.5)
        # also remove limits for consistency
        xmin, xmax = np.min(x_vals), np.max(x_vals)
        ax_fr.fill_betweenx(y_vals, xmin, x_vals, color="white", alpha=0.3)
        ax_fr.set_xlim(xmin, xmax)

        ax_fr.set(
            xlabel="Firing Rate\n(Hz)",
            ylim=[channel_indices[0], channel_indices[-1]],
            title="Activity",
        )
    else:
        ax_fr.text(0.5, 0.5, "N/A", color="white", ha="center")

    display_color = "#e2b962"
    ax_fr.tick_params(axis="x", labelsize=8.4, colors=display_color)
    ax_fr.xaxis.label.set_color(display_color)
    ax_fr.title.set_color(display_color)

    ax_fr.sharey(ax_hf_coh)
    ax_fr.tick_params(labelleft=False)
    ax_fr.grid(True, alpha=0.3)

    # ensure "dead channels" has y-ticks
    ax_hf_coh.set(ylabel="Channel")

    # --- plot heatmap ---
    # Update heatmap to use proper extent matching real channel indices
    # extent = [x_min, x_max, y_min, y_max]
    # We want y_min = channel_indices[0], y_max = channel_indices[-1]
    # NOTE: imshow with origin='lower' puts index 0 at bottom.
    # If we supply a subset of data (say 50 rows), but we want it to cover y=100 to y=150:
    heatmap_extent = [
        0,
        ns_plot / fs,
        channel_indices[0],
        channel_indices[-1] + 1,
    ]  # +1 to cover the last channel width

    heatmap_image = voltageshow(
        current_raw,
        fs,
        ax=ax_heatmap,
        cax=ax_cbar,
        extent=heatmap_extent,
        ylim=[channel_indices[0], channel_indices[-1] + 1],
    )
    ax_heatmap.set(ylabel="Channel")
    ax_heatmap.tick_params(axis="x", colors="white")

    # --- checkbox logic ---
    def on_checkbox_click(event):
        if event.inaxes not in checkbox_axes:
            return
        checkbox_idx = checkbox_axes.index(event.inaxes)
        rect = checkbox_rects[checkbox_idx]
        if (
            rect.get_x() <= event.xdata <= rect.get_x() + rect.get_width()
            and rect.get_y() <= event.ydata <= rect.get_y() + rect.get_height()
        ):
            checkbox_states[checkbox_idx] = not checkbox_states[checkbox_idx]
            checkbox_x_marks[checkbox_idx][0].set_visible(checkbox_states[checkbox_idx])
            checkbox_x_marks[checkbox_idx][1].set_visible(checkbox_states[checkbox_idx])

            cmr, hp = checkbox_states
            if cmr and hp:
                new_data = raw_variants["cmr_hp"]
            elif cmr:
                new_data = raw_variants["cmr"]
            elif hp:
                new_data = raw_variants["hp"]
            else:
                new_data = raw_variants["none"]

            heatmap_image.set_data(new_data * 1e6)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_checkbox_click)

    # --- buttons ---
    gs_buttons = gs_inner[2].subgridspec(1, 3, width_ratios=[1, 1.5, 1])
    gs_buttons_inner = gs_buttons[0, 1].subgridspec(1, 2, wspace=0.1)

    button_ax_save = fig.add_subplot(gs_buttons_inner[0, 0])
    button_ax_nosurface = fig.add_subplot(gs_buttons_inner[0, 1])

    save_button = Button(
        button_ax_save, "Save Surface Channel", color="#d1d8a1", hovercolor="#e9f0b3"
    )
    nosurface_button = Button(
        button_ax_nosurface,
        "No Surface (Save -1)",
        color="#c9c9c9",
        hovercolor="#ececec",
    )

    save_button.label.set_fontsize(10)
    save_button.label.set_fontweight("bold")
    nosurface_button.label.set_fontsize(10)
    nosurface_button.label.set_fontweight("bold")

    # --- interaction logic ---
    state = {
        "user_surface_channel": None,
        "final_channel": auto_surface_channel,
        "auto_line": [],
        "user_line": [],
        "saved": False,
    }

    # draw auto line
    if auto_surface_rel != -1:
        # We need the relative index for finding where to draw, but since we updated
        # the axes to use absolute coordinates (channel_indices), we can plot the
        # absolute channel number directly!

        # Plot at auto_surface_channel + 0.5 (center of the channel bin)
        line_y = auto_surface_channel + 0.5

        for ax_i in all_plot_axes:
            line = ax_i.axhline(
                line_y,
                color=SURFACE_COLOR,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Auto surface" if ax_i == ax_heatmap else "",
            )
            state["auto_line"].append(line)
        ax_heatmap.legend(loc="lower right", fontsize=10)

    # title
    title_text = f"File: {bin_path.name}"
    if shank_id is not None:
        title_text += f" | Shank {shank_id + 1}"
    title_text += f"\nAuto-detected surface channel: {auto_surface_channel}"
    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98, color="#ababab")

    def on_click(event):
        if event.inaxes in all_plot_axes and event.ydata is not None:
            # ydata is now in real channel indices
            clicked_ch_abs = int(np.round(event.ydata))

            # constrain to valid range within the current view
            # find nearest channel in channel_indices
            idx = (np.abs(channel_indices - clicked_ch_abs)).argmin()
            clicked_ch_abs = channel_indices[idx]

            state["user_surface_channel"] = clicked_ch_abs
            state["final_channel"] = clicked_ch_abs

            # remove old lines
            for line in state["user_line"]:
                line.remove()
            state["user_line"] = []

            # draw new lines
            line_y = clicked_ch_abs + 0.5
            for ax_i in all_plot_axes:
                line = ax_i.axhline(
                    line_y,
                    color=SURFACE_COLOR,
                    linestyle="-",
                    linewidth=3,
                    alpha=0.9,
                    label="User surface" if ax_i == ax_heatmap else "",
                )
                state["user_line"].append(line)

            ax_heatmap.legend(loc="lower right", fontsize=10)

            # update title
            new_title = f"File: {bin_path.name}"
            if shank_id is not None:
                new_title += f" | Shank {shank_id + 1}"
            new_title += f"\nAuto: {auto_surface_channel}  |  User: {clicked_ch_abs}"
            fig.suptitle(
                new_title, fontsize=14, fontweight="bold", y=0.98, color="#ababab"
            )
            fig.canvas.draw_idle()

    # hover logic needs to handle the 2 rows
    hover_text = fig.text(
        0.01,
        0.01,
        "",
        fontsize=10,
        family="monospace",
        verticalalignment="bottom",
        color="white",
        bbox=dict(boxstyle="round", facecolor="#404040", alpha=0.9, edgecolor="white"),
    )

    def on_hover(event):
        if event.inaxes is None:
            hover_text.set_text("")
            fig.canvas.draw_idle()
            return

        # ydata is absolute channel index
        ch_abs = int(np.round(event.ydata))

        # check if this channel is in our current subset
        if ch_abs not in channel_indices:
            return

        # Get relative index for looking up data arrays
        # (Assuming channel_indices is sorted, which it should be)
        # using searchsorted or just where
        matches = np.where(channel_indices == ch_abs)[0]
        if len(matches) == 0:
            return
        idx_rel = matches[0]

        txt = ""
        if event.inaxes == ax_hf_coh:
            txt = f"Ch {ch_abs}: HF coherence = {xfeats['xcor_hf'][idx_rel]:.4f}"
        elif event.inaxes == ax_psd_hf:
            txt = f"Ch {ch_abs}: HF Noise (PSD) = {xfeats['psd_hf'][idx_rel]:.4f}"
        elif event.inaxes == ax_lf_coh:
            txt = f"Ch {ch_abs}: LF coherence = {xfeats['xcor_lf'][idx_rel]:.4f}"
        elif event.inaxes == ax_lf_pow and "rms_lf" in xfeats:
            txt = f"Ch {ch_abs}: LF Power = {xfeats['rms_lf'][idx_rel] * 1e6:.1f} uV"
        elif event.inaxes == ax_gamma_pow and "power_gamma" in xfeats:
            txt = f"Ch {ch_abs}: Gamma Power = {xfeats['power_gamma'][idx_rel]:.2f} uV²/Hz"
        elif event.inaxes == ax_sp_amp and spike_amplitudes is not None:
            txt = f"Ch {ch_abs}: Spike Amp = {spike_amplitudes[idx_rel] * 1e6:.1f} uV"
        elif event.inaxes == ax_fr and firing_rates is not None:
            txt = f"Ch {ch_abs}: Firing Rate = {firing_rates[idx_rel]:.1f} Hz"
        elif event.inaxes == ax_heatmap:
            x_time = event.xdata
            if 0 <= x_time <= ns_plot / fs:
                t_idx = int(x_time * fs)
                if t_idx < ns_plot:
                    val = raw[idx_rel, t_idx] * 1e6
                    txt = f"Ch {ch_abs}, t={x_time:.3f}s: {val:.1f} uV"

        hover_text.set_text(txt)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    def save_plot():
        shank_num = shank_id if shank_id is not None else 0
        # Naming convention: surface_<input_file_name>_<shank_number>.png
        fname = (
            bin_path.parent
            / f"surface_{bin_path.name}_{shank_num}{channel_range_suffix}.png"
        )
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {fname}")

    def on_save(event):
        state["saved"] = True
        print(f"Saving surface channel: {state['final_channel']}")
        save_plot()
        plt.close(fig)

    def on_no_surface(event):
        state["final_channel"] = -1
        state["saved"] = True
        print("No surface detected by user, saving -1")
        save_plot()
        plt.close(fig)

    save_button.on_clicked(on_save)
    nosurface_button.on_clicked(on_no_surface)

    plt.show()

    if not state["saved"]:
        print("Window closed without saving. Using auto-detected channel.")
        save_plot()

    return state["final_channel"]
