# Import CMR function from run_detection
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from matplotlib.widgets import Button

sys.path.append(str(Path(__file__).parent))
from run_detection import apply_cmr

AP_RANGE_UV = 75
LF_RANGE_UV = 250
SURFACE_COLOR = "#b6d56b"  # Dirty yellow-green for surface channel line


def apply_highpass_filter(raw, fs, cutoff_hz):
    """
    Apply a Butterworth highpass filter to raw data.

    :param raw: Raw data array [nc, ns]
    :param fs: Sampling frequency in Hz
    :param cutoff_hz: Highpass cutoff frequency in Hz
    :return: Filtered data [nc, ns]
    """
    # Validate cutoff is below Nyquist frequency
    nyquist = fs / 2
    if cutoff_hz >= nyquist:
        print(
            f"\033[91mWarning: Highpass cutoff ({cutoff_hz} Hz) >= Nyquist frequency ({nyquist} Hz). Filter disabled.\033[0m"
        )
        return raw

    # Design Butterworth highpass filter (3rd order, same as in detect_bad_channels)
    sos = scipy.signal.butter(
        N=3, Wn=cutoff_hz / nyquist, btype="highpass", output="sos"
    )

    # Apply filter to each channel
    filtered = scipy.signal.sosfiltfilt(sos, raw, axis=1)

    return filtered


nature_style = {
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.facecolor": "#545454",  # "#2b2b2b",  # Dark gray background
    "axes.grid": False,
    "axes.labelcolor": "white",  # White labels
    "axes.labelsize": 14,
    "axes.linewidth": 1,
    "axes.titlecolor": "white",  # White titles
    "axes.titlesize": 16,
    "legend.labelcolor": "white",  # White legend labels
    "figure.facecolor": "#323232",  # 2b2b2b",  # Dark gray background
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "grid.color": "#505050",  # Darker gray grid
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.color": "white",  # White tick marks and labels
    "xtick.direction": "in",
    "xtick.labelsize": 12,
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "xtick.minor.size": 3,
    "xtick.minor.width": 0.5,
    "ytick.color": "white",  # White tick marks and labels
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
    :param sr:
    :return:
    """
    nc, ns = raw.shape
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset
    ns_plot = np.minimum(ns, 3000)
    fig, ax = plt.subplots(
        1, 5, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 8, 0.2]}
    )
    ax[0].plot(xfeats["xcor_hf"], np.arange(nc))
    ax[0].plot(  # plot channel below the similarity threshold as dead in black
        xfeats["xcor_hf"][(iko := channel_labels == 1)], np.arange(nc)[iko], "k*"
    )
    ax[0].plot(  # plot the values above the similarity threshold as noisy in red
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
    ax[1].plot(xfeats["psd_hf"], np.arange(nc), "w-")  # White line
    ax[1].plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        np.arange(nc)[iko],
        "*",
        color="#d5806b",
        markersize=8,
    )
    ax[1].plot(psd_hf_threshold * np.array([1, 1]), [0, nc], "--", color="gray")
    ax[1].set(ylabel="", xlabel="HF\npower", ylim=[0, nc], title="noisy\nchannels")
    ax[1].tick_params(labelleft=False)  # Hide tick labels
    ax[1].sharey(ax[0])
    ax[2].plot(xfeats["xcor_lf"], np.arange(nc), "w-")  # White line
    ax[2].plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)],
        np.arange(nc)[iko],
        "*",
        color="#b6d56b",
        markersize=8,
    )
    ax[2].set(ylabel="", xlabel="LF\ncoherence", ylim=[0, nc], title="outside")
    ax[2].tick_params(labelleft=False)  # Hide tick labels
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
    **axis_kwargs,
):
    """
    Visualizes electrophysiological voltage data as a heatmap.

    This function displays raw voltage data as a color-coded image with appropriate
    scaling based on the sampling frequency. It automatically selects voltage range
    based on whether the data is low-frequency (LF) or action potential (AP) data.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw voltage data array with shape (channels, samples), in Volts
    fs : float
        Sampling frequency in Hz, used to determine time axis scaling and voltage range.
    cmap : str, optional
        Matplotlib colormap name for the heatmap. Default is 'PuOr'.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    cax : matplotlib.axes.Axes, optional
        Axes object for the colorbar. If None and ax is None, a new colorbar axes is created.
    cbar_label : str, optional
        Label for the colorbar. Default is 'Voltage (uV)'.
    vrange: float, optional
        Voltage range for the colorbar. Defaults to +/- 75 uV for AP and +/- 250 uV for LF.
    scaling: float, optional
        Unit transform: default is 1e6: we expect Volts but plot uV.
    **axis_kwargs: optional
        Additional keyword arguments for the axis properties, fed to the ax.set() method.
    Returns
    -------
    matplotlib.image.AxesImage
        The image object created by imshow, which can be used for further customization.
    """
    if ax is None:
        fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.05]})
        ax, cax = axs
    nc, ns = raw.shape
    default_vrange = LF_RANGE_UV if fs < 2600 else AP_RANGE_UV
    vrange = vrange if vrange is not None else default_vrange
    im = ax.imshow(
        raw * scaling,
        origin="lower",
        cmap=cmap,
        aspect="auto",
        vmin=-vrange,
        vmax=vrange,
        extent=[0, ns / fs, 0, nc],
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
    shank_id=None,
    total_shanks=None,
    shank_channels=None,
    similarity_threshold=(-0.5, 1),
    psd_hf_threshold=0.02,
    hf_cutoff=None,
):
    """
    Interactive version of show_channels_labels that allows user to manually select
    the surface channel by clicking on any subplot.

    :param raw: Raw data [nc, ns]
    :param fs: Sampling frequency
    :param channel_labels: Channel labels array
    :param xfeats: Dictionary of features
    :param auto_surface_channel: Auto-detected surface channel (absolute index)
    :param bin_path: Path to the bin file (for window title)
    :param shank_id: Optional shank ID for multi-shank probes
    :param total_shanks: Optional total number of shanks
    :param shank_channels: Optional array of absolute channel indices for this shank (for multi-shank)
    :param similarity_threshold: Tuple of thresholds for dead/noisy detection
    :param psd_hf_threshold: Threshold for high-frequency PSD
    :param hf_cutoff: Optional highpass cutoff frequency (Hz) for heatmap visualization
    :return: Final surface channel selected (absolute index), or None if cancelled
    """
    nc, ns = raw.shape
    raw_original = raw.copy()
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset

    ns_plot = np.minimum(ns, 3000)

    # Pre-compute all 4 permutations of the data for interactive switching
    # Store only the plot window
    raw_variants = {}
    raw_base = raw[:, :ns_plot]

    # 1. No processing
    raw_variants["none"] = raw_base.copy()

    # 2. CMR only
    raw_cmr = apply_cmr(raw_base)
    raw_variants["cmr"] = raw_cmr

    # 3. Highpass only (300 Hz)
    raw_hp = apply_highpass_filter(raw_base, fs, 300)
    raw_variants["hp"] = raw_hp

    # 4. CMR + Highpass
    raw_cmr_hp = apply_cmr(raw_base)
    raw_cmr_hp = apply_highpass_filter(raw_cmr_hp, fs, 300)
    raw_variants["cmr_hp"] = raw_cmr_hp

    # Start with the variant based on hf_cutoff parameter (for backwards compatibility)
    if hf_cutoff is not None:
        current_raw = apply_highpass_filter(raw_base, fs, hf_cutoff)
    else:
        current_raw = raw_base.copy()

    # Convert absolute surface channel to relative position for plotting (if multi-shank)
    if shank_channels is not None and auto_surface_channel != -1:
        # Find the position of auto_surface_channel within shank_channels
        auto_surface_rel = np.where(shank_channels == auto_surface_channel)[0]
        if len(auto_surface_rel) > 0:
            auto_surface_rel = auto_surface_rel[0]
        else:
            auto_surface_rel = -1
    else:
        # Single-shank or no surface: use absolute as-is
        auto_surface_rel = auto_surface_channel

    # Create figure with space for checkboxes, button and firing rate plot
    fig = plt.figure(figsize=(16, 7))  # Reduced width by 20% (from 20 to 16)
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[
            0.05,
            1,
            0.08,  # 0.052
        ],  # Top row for checkboxes, bottom row for buttons (increased by 30% from 0.04)
        width_ratios=[1, 1, 1, 8, 1, 0.2],
        wspace=0.05,
        hspace=0.5,  # Doubled from 0.25 to push buttons twice as far down
    )

    # Create the main plot axes (now on row 1 instead of row 0)
    ax = [
        fig.add_subplot(gs[1, 0]),  # HF coherence
        fig.add_subplot(gs[1, 1]),  # PSD
        fig.add_subplot(gs[1, 2]),  # LF coherence
        fig.add_subplot(gs[1, 3]),  # Voltage heatmap
        fig.add_subplot(gs[1, 4]),  # Firing rate
        fig.add_subplot(gs[1, 5]),  # Colorbar
    ]

    # Create checkboxes above the heatmap - each in its own axis for proper spacing
    from matplotlib.patches import Rectangle

    # Create subgridspec for 2 checkbox columns
    checkbox_gs = gs[0, 3].subgridspec(1, 2, wspace=0.3)
    checkbox_ax1 = fig.add_subplot(checkbox_gs[0, 0])
    checkbox_ax2 = fig.add_subplot(checkbox_gs[0, 1])
    checkbox_ax1.axis("off")
    checkbox_ax2.axis("off")

    checkbox_labels = [" Common Median Reference", " Highpass (300 Hz)"]
    checkbox_states = [False, False]  # Track checkbox states
    checkbox_axes = [checkbox_ax1, checkbox_ax2]

    # Manually draw checkbox rectangles and labels
    checkbox_rects = []
    checkbox_texts = []
    checkbox_x_marks = []

    for i, (label, ax_check) in enumerate(zip(checkbox_labels, checkbox_axes)):
        # Position checkbox in center of its axis
        x_pos = 0.1
        y_pos = 0.15
        box_size = 0.8  # Doubled from 0.4

        # Draw checkbox rectangle (square)
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

        # Add label text (positioned to the right of checkbox)
        text = ax_check.text(
            x_pos + box_size + 0.05,
            y_pos + box_size / 2,
            label,
            verticalalignment="center",
            fontsize=15,
            color="white",
        )
        checkbox_texts.append(text)

        # Create X mark lines (initially invisible)
        x_mark_1 = ax_check.plot(
            [x_pos, x_pos + box_size],
            [y_pos, y_pos + box_size],
            "w-",
            linewidth=2,
            visible=False,
        )[0]
        x_mark_2 = ax_check.plot(
            [x_pos + box_size, x_pos],
            [y_pos, y_pos + box_size],
            "w-",
            linewidth=2,
            visible=False,
        )[0]
        checkbox_x_marks.append((x_mark_1, x_mark_2))

        ax_check.set_xlim(0, 2)
        ax_check.set_ylim(0, 1)
        ax_check.set_aspect("equal")
    # Set window title with shank info if multi-shank
    if shank_id is not None and total_shanks is not None:
        window_title = (
            f"Surface Channel Detection - Shank {shank_id + 1}/{total_shanks}"
        )
    else:
        window_title = "Surface Channel Detection"
    fig.canvas.manager.set_window_title(window_title)

    # Plot the features
    ax[0].plot(xfeats["xcor_hf"], np.arange(nc), "w-")  # White line
    ax[0].plot(
        xfeats["xcor_hf"][(iko := channel_labels == 1)],
        np.arange(nc)[iko],
        "*",
        color="#75a1d2",
        markersize=8,
    )
    ax[0].plot(similarity_threshold[0] * np.ones(2), [0, nc], "--", color="gray")
    ax[0].plot(similarity_threshold[1] * np.ones(2), [0, nc], "--", color="gray")
    ax[0].set(
        ylabel="Channel #",
        xlabel="HF\ncoherence",
        ylim=[0, nc],
        title="dead\nchannels",
    )
    ax[0].tick_params(
        axis="x", colors="#75a1d2", labelsize=8.4
    )  # Blue x-ticks, 30% smaller (12 * 0.7)
    ax[0].xaxis.label.set_color("#75a1d2")  # Blue x-axis label
    ax[0].title.set_color("#75a1d2")  # Blue title

    ax[1].plot(xfeats["psd_hf"], np.arange(nc), "w-")  # White line
    ax[1].plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        np.arange(nc)[iko],
        "*",
        color="#d5806b",
        markersize=8,
    )
    ax[1].plot(psd_hf_threshold * np.array([1, 1]), [0, nc], "--", color="gray")
    ax[1].set(ylabel="", xlabel="HF\npower", ylim=[0, nc], title="noisy\nchannels")
    ax[1].tick_params(labelleft=False)  # Hide tick labels
    ax[1].tick_params(
        axis="x", colors="#d5806b", labelsize=8.4
    )  # Red/orange x-ticks, 30% smaller
    ax[1].xaxis.label.set_color("#d5806b")  # Red/orange x-axis label
    ax[1].title.set_color("#d5806b")  # Red/orange title
    ax[1].sharey(ax[0])

    ax[2].plot(xfeats["xcor_lf"], np.arange(nc), "w-")  # White line
    ax[2].plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)],
        np.arange(nc)[iko],
        "*",
        color="#b6d56b",
        markersize=8,
    )
    ax[2].set(
        ylabel="", xlabel="LF\ncoherence", ylim=[0, nc], title=f"outside\nchannels"
    )
    ax[2].tick_params(labelleft=False)  # Hide tick labels
    ax[2].tick_params(
        axis="x", colors="#b6d56b", labelsize=8.4
    )  # Yellow/green x-ticks, 30% smaller
    ax[2].xaxis.label.set_color("#b6d56b")  # Yellow/green x-axis label
    ax[2].title.set_color("#b6d56b")  # Yellow/green title
    ax[2].sharey(ax[0])

    # Initial plot with no processing
    heatmap_image = voltageshow(current_raw, fs, ax=ax[3], cax=ax[5])
    ax[3].set(ylabel="")  # Remove y-axis labels from heatmap
    ax[3].tick_params(labelleft=False)  # Hide tick labels
    ax[3].tick_params(axis="x", colors="white")  # White x-ticks
    ax[3].sharey(ax[0])

    # Checkbox click handler
    def on_checkbox_click(event):
        """Handle checkbox clicks"""
        # Check if click is in any of the checkbox axes
        if event.inaxes not in checkbox_axes:
            return

        # Find which checkbox axis was clicked
        checkbox_idx = checkbox_axes.index(event.inaxes)

        # Get the rect for this checkbox
        rect = checkbox_rects[checkbox_idx]
        x_min = rect.get_x()
        x_max = x_min + rect.get_width()
        y_min = rect.get_y()
        y_max = y_min + rect.get_height()

        # Check if click is inside this checkbox
        if x_min <= event.xdata <= x_max and y_min <= event.ydata <= y_max:
            # Toggle checkbox state
            checkbox_states[checkbox_idx] = not checkbox_states[checkbox_idx]

            # Update X mark visibility
            checkbox_x_marks[checkbox_idx][0].set_visible(checkbox_states[checkbox_idx])
            checkbox_x_marks[checkbox_idx][1].set_visible(checkbox_states[checkbox_idx])

            # Update heatmap based on current states
            cmr_checked = checkbox_states[0]
            hp_checked = checkbox_states[1]

            if cmr_checked and hp_checked:
                new_data = raw_variants["cmr_hp"]
            elif cmr_checked:
                new_data = raw_variants["cmr"]
            elif hp_checked:
                new_data = raw_variants["hp"]
            else:
                new_data = raw_variants["none"]

            # Update the image data
            heatmap_image.set_data(new_data * 1e6)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_checkbox_click)

    # Plot firing rates if provided
    if firing_rates is not None:
        ax[4].plot(firing_rates, np.arange(nc), "w-", linewidth=1.5)  # White line
        ax[4].set(
            ylabel="",
            xlabel="Firing rate (Hz)",
            ylim=[0, nc],
            title="activity",
        )
        ax[4].tick_params(labelleft=False)  # Hide tick labels
        ax[4].tick_params(
            axis="x", colors="white", labelsize=8.4
        )  # White x-ticks, 30% smaller
        ax[4].sharey(ax[0])
        ax[4].grid(True, alpha=0.3)
    else:
        # Hide the firing rate axis if no data
        ax[4].set_visible(False)

    # Create text annotations for surface channels at the top
    if shank_id is not None:
        title_text = f"File: {bin_path.name} | Shank {shank_id}\nAuto-detected surface channel: {auto_surface_channel}"
    else:
        title_text = f"File: {bin_path.name}\nAuto-detected surface channel: {auto_surface_channel}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.98, color="#ababab")

    # State to track user selection
    state = {
        "user_surface_channel": None,
        "final_channel": auto_surface_channel,
        "auto_line": None,
        "user_line": None,
        "user_text": None,
        "saved": False,
    }

    # Draw the auto-detected surface line (use relative position for plotting)
    auto_lines = []
    if auto_surface_rel != -1:
        # Draw on all subplots except colorbar (first 5: coherence, PSD, LF, heatmap, firing rate)
        for i in range(5):
            if ax[i].get_visible():  # Only draw if subplot is visible
                line = ax[i].axhline(
                    auto_surface_rel + 0.5,
                    color=SURFACE_COLOR,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label="Auto surface"
                    if i == 3
                    else "",  # Only label line on imshow (ax[3]) for legend
                )
                auto_lines.append(line)
    state["auto_line"] = auto_lines

    # Add legend to imshow subplot (ax[3]) only if there are labeled artists
    if auto_surface_rel != -1:
        ax[3].legend(loc="lower right", fontsize=10)

    # Create Save and No Surface buttons under the heatmap (narrower buttons with padding)
    button_gs = gs[2, 3].subgridspec(
        1, 6, width_ratios=[1, 1, 0.2, 1, 1, 0]
    )  # [1, 2, 0.2, 2, 1, 0]
    button_ax_save = fig.add_subplot(button_gs[0, 1])  # Use column 1 (middle-left)
    button_ax_nosurface = fig.add_subplot(
        button_gs[0, 3]
    )  # Use column 3 (middle-right)
    save_button = Button(
        button_ax_save, "Save Surface Channel", color="#d1d8a1", hovercolor="#e9f0b3"
    )
    nosurface_button = Button(
        button_ax_nosurface,
        "No Surface (Save -1)",
        color="#c9c9c9",
        hovercolor="#ececec",
    )

    # Reduce button text font size by half
    save_button.label.set_fontsize(10)
    save_button.label.set_fontweight("bold")
    nosurface_button.label.set_fontsize(10)
    nosurface_button.label.set_fontweight("bold")

    def on_click(event):
        """Handle click events on any of the plot axes"""
        if (
            event.inaxes in ax[:5] and event.ydata is not None
        ):  # Include firing rate plot
            # Get the relative channel number from the y-coordinate
            clicked_channel_rel = int(np.round(event.ydata))

            # Clamp to valid range
            clicked_channel_rel = max(0, min(nc - 1, clicked_channel_rel))

            # Convert to absolute channel if multi-shank
            if shank_channels is not None:
                clicked_channel_abs = shank_channels[clicked_channel_rel]
            else:
                clicked_channel_abs = clicked_channel_rel

            state["user_surface_channel"] = clicked_channel_abs
            state["final_channel"] = clicked_channel_abs

            # Remove old user line if it exists
            if state["user_line"] is not None:
                for line in state["user_line"]:
                    line.remove()

            # Draw new user line (use relative position for plotting)
            user_lines = []
            for i in range(5):  # Draw on all subplots except colorbar
                if ax[i].get_visible():  # Only draw if subplot is visible
                    line = ax[i].axhline(
                        clicked_channel_rel + 0.5,
                        color=SURFACE_COLOR,
                        linestyle="-",
                        linewidth=3,
                        alpha=0.9,
                        label="User surface"
                        if i == 3
                        else "",  # Only label line on imshow (ax[3]) for legend
                    )
                    user_lines.append(line)
            state["user_line"] = user_lines

            # Update legend (always show after user clicks)
            ax[3].legend(loc="lower right", fontsize=10)

            # Update title to show user selection (show absolute channel numbers)
            if shank_id is not None:
                title_text = f"File: {bin_path.name} | Shank {shank_id}\nAuto: {auto_surface_channel}  |  User: {clicked_channel_abs}"
            else:
                title_text = f"File: {bin_path.name}\nAuto: {auto_surface_channel}  |  User: {clicked_channel_abs}"

            fig.suptitle(
                title_text, fontsize=14, fontweight="bold", y=0.98, color="#ababab"
            )

            # Redraw the canvas
            fig.canvas.draw_idle()

    def save_plot():
        """Save the current plot to file"""
        # Generate filename
        if shank_id is not None:
            plot_filename = bin_path.with_suffix("").with_suffix(
                f".SURFACE_SHANK{shank_id}.png"
            )
        else:
            plot_filename = bin_path.with_suffix("").with_suffix(".SURFACE.png")

        fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {plot_filename}")

    def on_save(event):
        """Handle save button click"""
        state["saved"] = True
        print(f"Saving surface channel: {state['final_channel']}")
        save_plot()
        plt.close(fig)

    def on_no_surface(event):
        """Handle no surface button click - set to -1 and save"""
        state["final_channel"] = -1
        state["saved"] = True
        print("No surface detected by user, saving -1")
        save_plot()
        plt.close(fig)

    # Add text annotation for hover values at bottom left
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
        """Display values at cursor position"""
        if event.inaxes is None:
            hover_text.set_text("")
            fig.canvas.draw_idle()
            return

        # Determine which subplot we're hovering over
        if event.inaxes == ax[0]:  # HF coherence
            ch = int(np.round(event.ydata))
            if 0 <= ch < nc:
                val = xfeats["xcor_hf"][ch]
                hover_text.set_text(f"Channel {ch}: HF coherence = {val:.4f}")
        elif event.inaxes == ax[1]:  # PSD
            ch = int(np.round(event.ydata))
            if 0 <= ch < nc:
                val = xfeats["psd_hf"][ch]
                hover_text.set_text(f"Channel {ch}: PSD = {val:.4f}")
        elif event.inaxes == ax[2]:  # LF coherence
            ch = int(np.round(event.ydata))
            if 0 <= ch < nc:
                val = xfeats["xcor_lf"][ch]
                hover_text.set_text(f"Channel {ch}: LF coherence = {val:.4f}")
        elif event.inaxes == ax[3]:  # Voltage heatmap
            # Get time and channel from cursor position
            x_time = event.xdata
            ch = int(np.round(event.ydata))
            if 0 <= ch < nc and 0 <= x_time <= ns_plot / fs:
                # Get sample index
                sample_idx = int(x_time * fs)
                if 0 <= sample_idx < ns_plot:
                    val_uv = raw[ch, sample_idx] * 1e6  # Convert to uV
                    hover_text.set_text(f"Ch {ch}, t={x_time:.3f}s: {val_uv:.1f} μV")
        elif event.inaxes == ax[4] and firing_rates is not None:  # Firing rate
            ch = int(np.round(event.ydata))
            if 0 <= ch < nc:
                val = firing_rates[ch]
                hover_text.set_text(f"Channel {ch}: Firing rate = {val:.1f} sp/s")
        else:
            hover_text.set_text("")

        fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    save_button.on_clicked(on_save)
    nosurface_button.on_clicked(on_no_surface)

    fig.tight_layout(pad=0)
    plt.show()

    # Save plot if user closed without clicking a button
    if not state["saved"]:
        print("Window closed without saving. Using auto-detected channel.")
        save_plot()

    # Return the final channel
    return state["final_channel"]
