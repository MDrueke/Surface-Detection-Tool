import sys
from pathlib import Path
import numpy as np
import scipy.signal
import argparse
import tkinter as tk
from tkinter import filedialog

sys.path.append(str(Path(__file__).parent))

from plotting import show_channels_labels_interactive
from run_detection import (
    detect_bad_channels_cbin,
    find_surface_channel,
    get_shank_info,
    filter_data_by_shank,
)
from reader import Reader


def get_options_dialog(cmr_default=False, hf_default=None):
    """
    Show an options dialog to get processing options.

    :param cmr_default: Default state of CMR checkbox
    :param hf_default: Default highpass cutoff frequency (None = disabled)
    :return: Dictionary with options {'cmr': bool, 'hf_cutoff': float or None}
    """
    options_window = tk.Tk()
    options_window.title("Options")
    options_window.geometry("350x180")

    # Center the window
    options_window.eval('tk::PlaceWindow . center')

    result = {'cmr': cmr_default, 'hf_cutoff': hf_default, 'confirmed': False}

    # CMR checkbox
    cmr_var = tk.BooleanVar(value=cmr_default)
    cmr_checkbox = tk.Checkbutton(
        options_window,
        text="Apply Common Median Referencing (CMR)",
        variable=cmr_var,
        font=("Arial", 10)
    )
    cmr_checkbox.pack(pady=(20, 10))

    # Highpass filter entry
    hf_frame = tk.Frame(options_window)
    hf_frame.pack(pady=10)

    hf_label = tk.Label(
        hf_frame,
        text="Heatmap Highpass Cutoff (Hz):\n(empty = disabled)",
        font=("Arial", 10),
        justify=tk.LEFT
    )
    hf_label.pack(side=tk.LEFT, padx=(0, 10))

    hf_entry = tk.Entry(hf_frame, width=10, font=("Arial", 10))
    hf_entry.pack(side=tk.LEFT)
    if hf_default is not None:
        hf_entry.insert(0, str(hf_default))

    def on_ok():
        result['cmr'] = cmr_var.get()

        # Validate and parse highpass cutoff
        hf_text = hf_entry.get().strip()
        if hf_text == "":
            result['hf_cutoff'] = None
        else:
            try:
                hf_value = float(hf_text)
                if hf_value <= 0:
                    print("\033[91mWarning: Highpass cutoff must be positive. Highpass filter disabled.\033[0m")
                    result['hf_cutoff'] = None
                else:
                    result['hf_cutoff'] = hf_value
            except ValueError:
                print(f"\033[91mWarning: Invalid highpass cutoff '{hf_text}'. Must be a number. Highpass filter disabled.\033[0m")
                result['hf_cutoff'] = None

        result['confirmed'] = True
        options_window.destroy()

    def on_cancel():
        result['confirmed'] = False
        options_window.destroy()

    # Buttons frame
    button_frame = tk.Frame(options_window)
    button_frame.pack(pady=10)

    ok_button = tk.Button(button_frame, text="OK", command=on_ok, width=10)
    ok_button.pack(side=tk.LEFT, padx=5)

    cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, width=10)
    cancel_button.pack(side=tk.LEFT, padx=5)

    options_window.mainloop()

    return result


def get_bin_file_path_and_options():
    """
    Get the path to a .bin file and processing options either from command line or via dialogs.

    :return: Tuple of (Path object to the .bin file, options dict)
    """
    parser = argparse.ArgumentParser(
        description="Interactive surface channel detection for Neuropixel recordings"
    )
    parser.add_argument(
        "bin_file",
        nargs="?",
        type=str,
        help="Path to the .ap.bin or .ap.cbin file",
    )
    parser.add_argument(
        "--cmr",
        action="store_true",
        help="Apply Common Median Referencing",
    )
    parser.add_argument(
        "--hf",
        type=str,
        default=None,
        help="Highpass filter cutoff frequency (Hz) for heatmap visualization",
    )
    args = parser.parse_args()

    # Parse and validate highpass cutoff
    hf_cutoff = None
    if args.hf is not None:
        try:
            hf_cutoff = float(args.hf)
            if hf_cutoff <= 0:
                print("\033[91mWarning: Highpass cutoff must be positive. Highpass filter disabled.\033[0m")
                hf_cutoff = None
        except ValueError:
            print(f"\033[91mWarning: Invalid highpass cutoff '{args.hf}'. Must be a number. Highpass filter disabled.\033[0m")
            hf_cutoff = None

    if args.bin_file:
        bin_path = Path(args.bin_file)
        if not bin_path.exists():
            raise FileNotFoundError(f"File not found: {bin_path}")
        return bin_path, {'cmr': args.cmr, 'hf_cutoff': hf_cutoff}
    else:
        # No command-line argument provided - open file dialog
        print("No file specified, opening file dialog...")
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        file_path = filedialog.askopenfilename(
            title="Select Neuropixel recording file",
            filetypes=[
                ("SpikeGLX files", "*.ap.bin *.ap.cbin"),
                ("Binary files", "*.bin"),
                ("Compressed files", "*.cbin"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            print("No file selected. Exiting.")
            sys.exit(0)

        root.destroy()  # Clean up the hidden window

        # Show options dialog
        options = get_options_dialog(cmr_default=args.cmr, hf_default=hf_cutoff)
        if not options['confirmed']:
            print("Options dialog cancelled. Exiting.")
            sys.exit(0)

        return Path(file_path), options


def save_surface_channel(bin_path, surface_results, is_multi_shank=False):
    """
    Save the surface channel(s) to a text file next to the .bin file.

    :param bin_path: Path to the original .bin file
    :param surface_results: For single-shank: int (surface channel)
                           For multi-shank: list of dicts with keys 'shank_id', 'abs_channel', 'rel_channel'
    :param is_multi_shank: Boolean indicating if this is a multi-shank probe
    """
    # Create output filename
    output_file = bin_path.with_suffix("").with_suffix(".surface_channel.txt")

    with open(output_file, "w") as f:
        if is_multi_shank:
            # Multi-shank format: shank0: 245 (on-shank: 50)
            for result in surface_results:
                shank_id = result['shank_id']
                abs_ch = result['abs_channel']
                rel_ch = result['rel_channel']
                f.write(f"shank{shank_id}: {abs_ch} (on-shank: {rel_ch})\n")
            print(f"Multi-shank surface channels saved to: {output_file}")
        else:
            # Single-shank format: just the channel number
            f.write(f"{surface_results}\n")
            print(f"Surface channel {surface_results} saved to: {output_file}")

    return output_file


def main():
    # Get the bin file path and options from command line or dialogs
    bin_path, options = get_bin_file_path_and_options()
    print(f"Processing: {bin_path}")

    # Print CMR status if enabled
    if options['cmr']:
        print("Applying Common Median Referencing (CMR)")

    with Reader(bin_path) as sr:
        # Get robust channel labels by aggregating over chunks, but get the
        # features and raw data from the LAST chunk for plotting.
        channel_labels, xfeats, raw, fs = detect_bad_channels_cbin(sr, apply_cmr_flag=options['cmr'])

        # Apply the median filter to the LF coherence feature for plotting,
        # exactly as the IBL QC pipeline does.
        xfeats['xcor_lf'] = scipy.signal.medfilt(xfeats['xcor_lf'], 11)

        # Get shank information
        shank_info = get_shank_info(sr.geometry)

        # Print probe information
        probe_version = sr.version if hasattr(sr, 'version') else 'Unknown'
        num_active_shanks = len(shank_info) if shank_info is not None else 1
        print(f"Probe version: {probe_version}, Active shanks: {num_active_shanks}")

        if shank_info is None or len(shank_info) == 1:
            # Single active shank - use original workflow
            # Find the auto-detected surface channel
            auto_surface_channel = find_surface_channel(channel_labels)
            if auto_surface_channel == -1:
                print("No surface channel detected")
            else:
                print(f"Auto-detected surface channel: {auto_surface_channel}")

            # Show interactive plot and get user-selected surface channel (if any)
            final_surface_channel = show_channels_labels_interactive(
                raw, fs, channel_labels, xfeats, auto_surface_channel, bin_path,
                hf_cutoff=options['hf_cutoff']
            )

            # Save the final surface channel
            if final_surface_channel is not None:
                save_surface_channel(bin_path, final_surface_channel, is_multi_shank=False)

        else:
            # Multi-shank probe - process each shank sequentially
            num_shanks = len(shank_info)
            print(f"Multi-shank probe detected: {num_shanks} shanks")

            surface_results = []

            for shank_id, shank_channels in sorted(shank_info.items()):
                print(f"\n=== Processing Shank {shank_id} ({len(shank_channels)} channels) ===")

                # Filter data for this shank
                raw_shank, labels_shank, xfeats_shank = filter_data_by_shank(
                    raw, channel_labels, xfeats, shank_channels
                )

                # Find auto-detected surface for this shank
                auto_surface_local = find_surface_channel(labels_shank)

                # Convert local channel index to absolute channel index
                if auto_surface_local == -1:
                    auto_surface_abs = -1
                else:
                    auto_surface_abs = shank_channels[auto_surface_local]

                if auto_surface_abs == -1:
                    print("No surface channel detected")
                else:
                    print(f"Auto-detected surface channel (abs): {auto_surface_abs}")

                # Show interactive plot for this shank
                final_surface_abs = show_channels_labels_interactive(
                    raw_shank,
                    fs,
                    labels_shank,
                    xfeats_shank,
                    auto_surface_abs,
                    bin_path,
                    shank_id=shank_id,
                    total_shanks=num_shanks,
                    shank_channels=shank_channels,
                    hf_cutoff=options['hf_cutoff'],
                )

                # Calculate relative channel (position within shank)
                if final_surface_abs == -1:
                    final_surface_rel = -1
                else:
                    # Find the position of this channel within the shank's channels
                    final_surface_rel = np.where(shank_channels == final_surface_abs)[0][0]

                # Store results
                surface_results.append({
                    'shank_id': shank_id,
                    'abs_channel': final_surface_abs,
                    'rel_channel': final_surface_rel,
                })

                print(f"Shank {shank_id}: Surface channel = {final_surface_abs} (on-shank: {final_surface_rel})")

            # Save all results
            save_surface_channel(bin_path, surface_results, is_multi_shank=True)


if __name__ == "__main__":
    main()
