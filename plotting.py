import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from matplotlib.widgets import Button

AP_RANGE_UV = 75
LF_RANGE_UV = 250
SURFACE_COLOR = 'yellowgreen'  # Dirty yellow-green for surface channel line


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
        print(f"\033[91mWarning: Highpass cutoff ({cutoff_hz} Hz) >= Nyquist frequency ({nyquist} Hz). Filter disabled.\033[0m")
        return raw

    # Design Butterworth highpass filter (3rd order, same as in detect_bad_channels)
    sos = scipy.signal.butter(N=3, Wn=cutoff_hz / nyquist, btype='highpass', output='sos')

    # Apply filter to each channel
    filtered = scipy.signal.sosfiltfilt(sos, raw, axis=1)

    return filtered

nature_style = {
    "axes.axisbelow": True,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.labelcolor": "black",
    "axes.labelsize": 14,
    "axes.linewidth": 1,
    "axes.titlecolor": "black",
    "axes.titlesize": 16,
    "figure.facecolor": "white",
    "figure.figsize": (10, 6),
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "xtick.color": "black",
    "xtick.direction": "in",
    "xtick.labelsize": 12,
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "xtick.minor.size": 3,
    "xtick.minor.width": 0.5,
    "ytick.color": "black",
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
    ax[0].plot(similarity_threshold[0] * np.ones(2), [0, nc], "k--")
    ax[0].plot(similarity_threshold[1] * np.ones(2), [0, nc], "r--")
    ax[0].set(
        ylabel="channel #",
        xlabel="high coherence",
        ylim=[0, nc],
        title="dead channel",
    )
    ax[1].plot(xfeats["psd_hf"], np.arange(nc))
    ax[1].plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        np.arange(nc)[iko],
        "r*",
    )
    ax[1].plot(psd_hf_threshold * np.array([1, 1]), [0, nc], "r--")
    ax[1].set(yticklabels=[], xlabel="PSD", ylim=[0, nc], title="noisy channel")
    ax[1].sharey(ax[0])
    ax[2].plot(xfeats["xcor_lf"], np.arange(nc))
    ax[2].plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)], np.arange(nc)[iko], "y*"
    )
    ax[2].plot([-0.75, -0.75], [0, nc], "y--")
    ax[2].set(yticklabels=[], xlabel="LF coherence", ylim=[0, nc], title="outside")
    ax[2].sharey(ax[0])
    voltageshow(raw[:, :ns_plot], fs, ax=ax[3], cax=ax[4])
    ax[3].sharey(ax[0])
    fig.tight_layout()
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
    axis_kwargs = (
        dict(ylim=[0, nc], xlabel="Time (s)", ylabel="Depth (μm)") | axis_kwargs
    )
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
    raw = raw - np.mean(raw, axis=-1)[:, np.newaxis]  # removes DC offset

    # Apply highpass filter to heatmap if requested
    if hf_cutoff is not None:
        raw = apply_highpass_filter(raw, fs, hf_cutoff)

    ns_plot = np.minimum(ns, 3000)

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

    # Create figure with space for button
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.08], width_ratios=[1, 1, 1, 8, 0.2])

    # Create the main plot axes
    ax = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
    ]

    # Set window title with shank info if multi-shank
    if shank_id is not None and total_shanks is not None:
        window_title = f"Surface Channel Detection - Shank {shank_id + 1}/{total_shanks}"
    else:
        window_title = "Surface Channel Detection"
    fig.canvas.manager.set_window_title(window_title)

    # Plot the features
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
    ax[0].plot(similarity_threshold[0] * np.ones(2), [0, nc], "k--")
    ax[0].plot(similarity_threshold[1] * np.ones(2), [0, nc], "r--")
    ax[0].set(
        ylabel="channel #",
        xlabel="high coherence",
        ylim=[0, nc],
        title="dead channel",
    )

    ax[1].plot(xfeats["psd_hf"], np.arange(nc))
    ax[1].plot(
        xfeats["psd_hf"][(iko := xfeats["psd_hf"] > psd_hf_threshold)],
        np.arange(nc)[iko],
        "r*",
    )
    ax[1].plot(psd_hf_threshold * np.array([1, 1]), [0, nc], "r--")
    ax[1].set(yticklabels=[], xlabel="PSD", ylim=[0, nc], title="noisy channel")
    ax[1].sharey(ax[0])

    ax[2].plot(xfeats["xcor_lf"], np.arange(nc))
    ax[2].plot(
        xfeats["xcor_lf"][(iko := channel_labels == 3)], np.arange(nc)[iko], "y*"
    )
    ax[2].plot([-0.75, -0.75], [0, nc], "y--")
    ax[2].set(yticklabels=[], xlabel="LF coherence", ylim=[0, nc], title="outside")
    ax[2].sharey(ax[0])

    voltageshow(raw[:, :ns_plot], fs, ax=ax[3], cax=ax[4])
    ax[3].sharey(ax[0])

    # Create text annotations for surface channels at the top
    if shank_id is not None:
        title_text = f"File: {bin_path.name} | Shank {shank_id}\nAuto-detected surface channel: {auto_surface_channel}"
    else:
        title_text = f"File: {bin_path.name}\nAuto-detected surface channel: {auto_surface_channel}"

    fig.suptitle(
        title_text,
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    # State to track user selection
    state = {
        'user_surface_channel': None,
        'final_channel': auto_surface_channel,
        'auto_line': None,
        'user_line': None,
        'user_text': None,
        'saved': False,
    }

    # Draw the auto-detected surface line (use relative position for plotting)
    auto_lines = []
    if auto_surface_rel != -1:
        for i in range(4):  # Draw on first 4 subplots
            line = ax[i].axhline(
                auto_surface_rel + 0.5,
                color=SURFACE_COLOR,
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label='Auto surface' if i == 0 else ''  # Only label first line for legend
            )
            auto_lines.append(line)
    state['auto_line'] = auto_lines

    # Add legend to first subplot only if there are labeled artists
    if auto_surface_rel != -1:
        ax[0].legend(loc='lower left', fontsize=10)

    # Create Save and No Surface buttons under the heatmap (split 50/50)
    button_gs = gs[1, 3].subgridspec(1, 2)
    button_ax_save = fig.add_subplot(button_gs[0, 0])
    button_ax_nosurface = fig.add_subplot(button_gs[0, 1])
    save_button = Button(button_ax_save, 'Save Surface Channel', color='lightgreen', hovercolor='green')
    nosurface_button = Button(button_ax_nosurface, 'No Surface (Save -1)', color='lightyellow', hovercolor='yellow')

    def on_click(event):
        """Handle click events on any of the plot axes"""
        if event.inaxes in ax[:4] and event.ydata is not None:
            # Get the relative channel number from the y-coordinate
            clicked_channel_rel = int(np.round(event.ydata))

            # Clamp to valid range
            clicked_channel_rel = max(0, min(nc - 1, clicked_channel_rel))

            # Convert to absolute channel if multi-shank
            if shank_channels is not None:
                clicked_channel_abs = shank_channels[clicked_channel_rel]
            else:
                clicked_channel_abs = clicked_channel_rel

            state['user_surface_channel'] = clicked_channel_abs
            state['final_channel'] = clicked_channel_abs

            # Remove old user line if it exists
            if state['user_line'] is not None:
                for line in state['user_line']:
                    line.remove()

            # Draw new user line (use relative position for plotting)
            user_lines = []
            for i in range(4):
                line = ax[i].axhline(
                    clicked_channel_rel + 0.5,
                    color=SURFACE_COLOR,
                    linestyle='-',
                    linewidth=3,
                    alpha=0.9,
                    label='User surface' if i == 0 else ''  # Only label first line for legend
                )
                user_lines.append(line)
            state['user_line'] = user_lines

            # Update legend (always show after user clicks)
            ax[0].legend(loc='lower left', fontsize=10)

            # Update title to show user selection (show absolute channel numbers)
            if shank_id is not None:
                title_text = f"File: {bin_path.name} | Shank {shank_id}\nAuto: {auto_surface_channel}  |  User: {clicked_channel_abs}"
            else:
                title_text = f"File: {bin_path.name}\nAuto: {auto_surface_channel}  |  User: {clicked_channel_abs}"

            fig.suptitle(
                title_text,
                fontsize=14,
                fontweight='bold',
                y=0.98
            )

            # Redraw the canvas
            fig.canvas.draw_idle()

    def save_plot():
        """Save the current plot to file"""
        # Generate filename
        if shank_id is not None:
            plot_filename = bin_path.with_suffix('').with_suffix(f'.SURFACE_SHANK{shank_id}.png')
        else:
            plot_filename = bin_path.with_suffix('').with_suffix('.SURFACE.png')

        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")

    def on_save(event):
        """Handle save button click"""
        state['saved'] = True
        print(f"Saving surface channel: {state['final_channel']}")
        save_plot()
        plt.close(fig)

    def on_no_surface(event):
        """Handle no surface button click - set to -1 and save"""
        state['final_channel'] = -1
        state['saved'] = True
        print("No surface detected by user, saving -1")
        save_plot()
        plt.close(fig)

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    save_button.on_clicked(on_save)
    nosurface_button.on_clicked(on_no_surface)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    # Save plot if user closed without clicking a button
    if not state['saved']:
        print("Window closed without saving. Using auto-detected channel.")
        save_plot()

    # Return the final channel
    return state['final_channel']
