import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import scipy.signal

sys.path.append(str(Path(__file__).parent))

from plotting import show_channels_labels_interactive
from reader import Reader
from run_detection import (
    analyze_recording,
    filter_data_by_shank,
    find_surface_channel,
    get_shank_info,
)


def get_options_dialog(cmr_default=False, hf_default=None):
    """
    Show an options dialog to get processing options.
    """
    options_window = tk.Tk()
    options_window.title("Options")
    options_window.geometry("350x180")

    bg_color = "#323232"
    fg_color = "white"

    options_window.configure(bg=bg_color)
    options_window.eval("tk::PlaceWindow . center")

    result = {"cmr": cmr_default, "hf_cutoff": hf_default, "confirmed": False}

    cmr_var = tk.BooleanVar(value=cmr_default)
    cmr_checkbox = tk.Checkbutton(
        options_window,
        text="Apply Common Median Referencing (CMR)",
        variable=cmr_var,
        font=("Arial", 10),
        bg=bg_color,
        fg=fg_color,
        selectcolor="black",
        activebackground=bg_color,
        activeforeground=fg_color,
    )
    cmr_checkbox.pack(pady=(20, 10))

    hf_frame = tk.Frame(options_window, bg=bg_color)
    hf_frame.pack(pady=10)

    hf_label = tk.Label(
        hf_frame,
        text="Heatmap Highpass Cutoff (Hz):\n(empty = disabled)",
        font=("Arial", 10),
        justify=tk.LEFT,
        bg=bg_color,
        fg=fg_color,
    )
    hf_label.pack(side=tk.LEFT, padx=(0, 10))

    hf_entry = tk.Entry(
        hf_frame,
        width=10,
        font=("Arial", 10),
        bg="#505050",
        fg="white",
        insertbackground="white",
    )
    hf_entry.pack(side=tk.LEFT)
    if hf_default is not None:
        hf_entry.insert(0, str(hf_default))

    def on_ok():
        result["cmr"] = cmr_var.get()

        hf_text = hf_entry.get().strip()
        if hf_text == "":
            result["hf_cutoff"] = None
        else:
            try:
                hf_value = float(hf_text)
                if hf_value <= 0:
                    print(
                        "\033[91mWarning: Highpass cutoff must be positive. Highpass filter disabled.\033[0m"
                    )
                    result["hf_cutoff"] = None
                else:
                    result["hf_cutoff"] = hf_value
            except ValueError:
                print(
                    f"\033[91mWarning: Invalid highpass cutoff '{hf_text}'. Must be a number. Highpass filter disabled.\033[0m"
                )
                result["hf_cutoff"] = None

        result["confirmed"] = True
        options_window.destroy()

    def on_cancel():
        result["confirmed"] = False
        options_window.destroy()

    button_frame = tk.Frame(options_window, bg=bg_color)
    button_frame.pack(pady=10)

    ok_button = tk.Button(
        button_frame, text="OK", command=on_ok, width=10, highlightbackground=bg_color
    )
    ok_button.pack(side=tk.LEFT, padx=5)

    cancel_button = tk.Button(
        button_frame,
        text="Cancel",
        command=on_cancel,
        width=10,
        highlightbackground=bg_color,
    )
    cancel_button.pack(side=tk.LEFT, padx=5)

    options_window.mainloop()

    return result


def get_bin_file_path_and_options():
    """
    Get the path to a .bin file and processing options either from command line or via dialogs.
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: print detailed detection info and save debug data to .npy files",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=40,
        help="Number of time chunks to analyze for detection (default: 40)",
    )
    parser.add_argument(
        "--spike_threshold",
        type=float,
        default=-6.0,
        help="Spike detection threshold in multiples of MAD (default: -6.0)",
    )
    parser.add_argument(
        "-t",
        "--time_slice",
        nargs=2,
        type=float,
        help="Time window to analyze. Either as proportions (0.0-1.0) or seconds (0-total). E.g. '-t 0.8 1' or '-t 4500 7000'",
    )
    args = parser.parse_args()

    hf_cutoff = None
    if args.hf is not None:
        try:
            hf_cutoff = float(args.hf)
            if hf_cutoff <= 0:
                print(
                    "\033[91mWarning: Highpass cutoff must be positive. Highpass filter disabled.\033[0m"
                )
                hf_cutoff = None
        except ValueError:
            print(
                f"\033[91mWarning: Invalid highpass cutoff '{args.hf}'. Must be a number. Highpass filter disabled.\033[0m"
            )
            hf_cutoff = None

    if args.bin_file:
        bin_path = Path(args.bin_file)
        if not bin_path.exists():
            raise FileNotFoundError(f"File not found: {bin_path}")

        time_slice = None
        if args.time_slice:
            t1, t2 = args.time_slice
            if t1 < 0 or t2 < 0:
                print("\033[91mError: Time values must be positive.\033[0m")
                sys.exit(1)
            if t1 >= t2:
                print("\033[91mError: Start time must be less than end time.\033[0m")
                sys.exit(1)

            # heuristic: if both <= 1.0, treat as proportions. else seconds.
            if t2 <= 1.0:
                time_slice = ("proportion", t1, t2)
            else:
                time_slice = ("seconds", t1, t2)

        return bin_path, {
            "cmr": args.cmr,
            "hf_cutoff": hf_cutoff,
            "debug": args.debug,
            "n_chunks": args.n_chunks,
            "spike_threshold": args.spike_threshold,
            "time_slice": time_slice,
        }
    else:
        print("No file specified, opening file dialog...")
        root = tk.Tk()
        root.withdraw()

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

        root.destroy()

        options = get_options_dialog(cmr_default=args.cmr, hf_default=hf_cutoff)
        if not options["confirmed"]:
            print("Options dialog cancelled. Exiting.")
            sys.exit(0)

        options["debug"] = False
        options["n_chunks"] = 20
        # spike_threshold will be whatever default is passed to analyze_recording or hardcoded here
        options["spike_threshold"] = -6
        options["time_slice"] = None

        return Path(file_path), options


def save_surface_channel(bin_path, surface_results, is_multi_shank=False):
    """
    Save the surface channel(s) to a text file next to the .bin file.
    """
    output_file = bin_path.with_suffix("").with_suffix(".surface_channel.txt")

    with open(output_file, "w") as f:
        if is_multi_shank:
            for result in surface_results:
                shank_id = result["shank_id"]
                abs_ch = result["abs_channel"]
                rel_ch = result["rel_channel"]
                f.write(f"shank{shank_id}: {abs_ch} (on-shank: {rel_ch})\n")
            print(f"Multi-shank surface channels saved to: {output_file}")
        else:
            f.write(f"{surface_results}\n")
            print(f"Surface channel {surface_results} saved to: {output_file}")

    return output_file


def main():
    bin_path, options = get_bin_file_path_and_options()
    print(f"Processing: {bin_path}")

    if options["cmr"]:
        print("Applying Common Median Referencing (CMR)")

    with Reader(bin_path) as sr:
        channel_labels, xfeats, raw, fs, firing_rates, spike_amplitudes = (
            analyze_recording(
                sr,
                n_batches=options["n_chunks"],
                spike_threshold=options["spike_threshold"],
                apply_cmr_flag=options["cmr"],
                debug=options["debug"],
                time_slice=options["time_slice"],
            )
        )

        print(
            f"Firing rate range: {np.min(firing_rates):.1f} - {np.max(firing_rates):.1f} spikes/s"
        )

        # apply the median filter to the lf coherence feature for plotting,
        # exactly as the ibl pipeline does.
        xfeats["xcor_lf"] = scipy.signal.medfilt(xfeats["xcor_lf"], 11)

        if options["debug"] and "debug" in xfeats:
            debug_file = bin_path.with_suffix("").with_suffix(".debug.npy")
            np.save(debug_file, xfeats["debug"])
            print(f"Debug data saved to: {debug_file}")
            print(f"  Available keys: {list(xfeats['debug'].keys())}")

        shank_info = get_shank_info(sr.geometry)

        probe_version = sr.version if hasattr(sr, "version") else "Unknown"
        num_active_shanks = len(shank_info) if shank_info is not None else 1
        print(f"Probe version: {probe_version}, Active shanks: {num_active_shanks}")

        if shank_info is None or len(shank_info) == 1:
            auto_surface_channel = find_surface_channel(channel_labels)
            if auto_surface_channel == -1:
                print("No surface channel detected")
            else:
                print(f"Auto-detected surface channel: {auto_surface_channel}")

            final_surface_channel = show_channels_labels_interactive(
                raw,
                fs,
                channel_labels,
                xfeats,
                auto_surface_channel,
                bin_path,
                firing_rates=firing_rates,
                spike_amplitudes=spike_amplitudes,
                hf_cutoff=options["hf_cutoff"],
            )

            if final_surface_channel is not None:
                save_surface_channel(
                    bin_path, final_surface_channel, is_multi_shank=False
                )

        else:
            num_shanks = len(shank_info)
            print(f"Multi-shank probe detected: {num_shanks} shanks")

            surface_results = []

            for shank_id, shank_channels in sorted(shank_info.items()):
                print(
                    f"\n=== Processing Shank {shank_id} ({len(shank_channels)} channels) ==="
                )

                raw_shank, labels_shank, xfeats_shank = filter_data_by_shank(
                    raw, channel_labels, xfeats, shank_channels
                )

                firing_rates_shank = firing_rates[shank_channels]
                spike_amplitudes_shank = spike_amplitudes[shank_channels]

                auto_surface_local = find_surface_channel(labels_shank)

                if auto_surface_local == -1:
                    auto_surface_abs = -1
                else:
                    auto_surface_abs = shank_channels[auto_surface_local]

                if auto_surface_abs == -1:
                    print("No surface channel detected")
                else:
                    print(f"Auto-detected surface channel (abs): {auto_surface_abs}")

                final_surface_abs = show_channels_labels_interactive(
                    raw_shank,
                    fs,
                    labels_shank,
                    xfeats_shank,
                    auto_surface_abs,
                    bin_path,
                    firing_rates=firing_rates_shank,
                    spike_amplitudes=spike_amplitudes_shank,
                    shank_id=shank_id,
                    total_shanks=num_shanks,
                    shank_channels=shank_channels,
                    hf_cutoff=options["hf_cutoff"],
                )

                if final_surface_abs == -1:
                    final_surface_rel = -1
                else:
                    final_surface_rel = np.where(shank_channels == final_surface_abs)[
                        0
                    ][0]

                surface_results.append(
                    {
                        "shank_id": shank_id,
                        "abs_channel": final_surface_abs,
                        "rel_channel": final_surface_rel,
                    }
                )

                print(
                    f"Shank {shank_id}: Surface channel = {final_surface_abs} (on-shank: {final_surface_rel})"
                )

            save_surface_channel(bin_path, surface_results, is_multi_shank=True)


if __name__ == "__main__":
    main()
