# Surface Detection for Neuropixel Recordings

## Overview

This tool detects the brain surface and bad channels in Neuropixel recordings (SpikeGLX .bin/.cbin). The surface, noisy channel and dead channel detection is based on code that's part of the ibl sorter repository found here: https://github.com/int-brain-lab/ibl-sorter. It provides an automated estimate alongside an interactive visualization, allowing users to verify and adjust the surface channel based on signal features. 

The tool analyzes recordings to classify channels as:
- **Dead channels**: Low signal correlation
- **Noisy channels**: High-frequency (>80% Nyquist) power or excessive correlation
- **Outside channels**: Channels above the brain surface (in air/saline)

The tool is currently limited to data acquired with SpikeGLX. Contact me if you would like me to adapt it to data acquired with OpenEphys.

![Surface Detection Example](example.png)

## Installation

To install the tool locally:

```bash
pip install .
```

## Usage

### GUI file picker

Run without arguments to launch the file picker and options dialog:

```bash
surface-detection
```

### Command Line Mode

```bash
surface-detection /path/to/recording.ap.bin [OPTIONS]
```

### Command Line Options

**File input:**
- `PATH` or `--bin_file PATH` - Path to the .ap.bin or .ap.cbin file

**Detection preprocessing:**
- `--cmr` - Apply Common Median Referencing to data before running detection algorithms
- `--hf FREQUENCY` - Apply highpass filter (Hz) to data before detection (e.g., `--hf 300`)
- `--n_chunks N` - Number of time chunks to analyze (default: 40)
- `--spike_threshold THRESH` - Spike detection threshold in multiples of MAD (default: -6.0)
- `-t` or `--time_slice START END` - Time window to analyze:
    - **Proportions**: Use values <= 1.0 (e.g., `-t 0.0 0.1` for first 10%)
    - **Seconds**: Use values > 1.0 (e.g., `-t 0 600` for first 10 minutes)
- `-cr` or `--channel_range START END [START END ...]` - Restrict analysis to specific channel ranges (integers).
    - **Single-shank:** Provide 2 numbers (e.g., `-cr 100 200` to analyze absolute channels 100 to 200).
    - **Multi-shank (Universal):** Provide 2 numbers. These apply relatively to *every* shank (e.g., `-cr 0 50` checks the bottom 50 channels of each shank).
    - **Multi-shank (Specific):** Provide 2 numbers per shank. (e.g., for a 4-shank probe, provide 8 numbers to set specific ranges for Shank 0, Shank 1, etc. in order).

**Debug options:**
- `--debug` - Enable debug mode (prints info, saves intermediate .npy)

### Multi-Shank Probes

The tool automatically detects if the recording comes from a multi-shank probe (e.g., NP 2.4).
- It iterates through each shank sequentially.
- The interactive plot shows data specific to the current shank.
- **Surface Channel Output:** For multi-shank probes, the output text file lists the surface channel for each shank ID.
- **Filtering:** Use the `-cr` flag to restrict the search range if you already have a rough idea of the surface location (e.g., `-cr 0 100` to only search the bottom 100 channels of each shank).

### Examples

**Basic usage:**
```bash
surface-detection
```

**Analyze specific time window (0-600s) with common median referencing:**
```bash
surface-detection recording.ap.bin --cmr -t 0 600
```

**Debug mode:**
```bash
surface-detection recording.ap.bin --debug
```

## Interactive Plot

The visualization displays the following metrics for surface determination:

1. **Dead Channels**: Channels with very low signal correlation (black stars = dead)
2. **Noisy Channels**: Channels with excessive high-frequency noise (red stars = noisy)
3. **Outside Brain**: Channels with low low-frequency coherence (green stars = outside)
4. **Activity**: Firing rate in Hz (Gold line)
5. **Mean Absolute Voltage (MAV)**: General signal magnitude (uV)
6. **Gamma Power**: Power in 60-100 Hz band (useful for cortex identification)
7. **Spike Amplitude**: Median amplitude of detected spikes

**Interaction:**
- **Heatmap**: Shows raw voltage traces. Controls for CMR and Highpass (300Hz) are checked by default.
- **Click**: Click on any plot to manually set the surface channel.
- **Save**: "Save Surface Channel" writes the result to a text file.

## Output

The tool saves:
- `<filename>.surface_channel.txt` - The detected (or manually selected) surface channel number.
- `<filename>.debug.npy` - (If `--debug`) Dictionary of computed features.

### Performance
The tool is parallelizedm it will use <n_cpu_cores> - 1 jobs, or estimate memory usage and limit the jobs according to that. The default of 40 analyzed chunks per shank is a lot, but finishes quite quickly due to the parallelization. More chunks improve precision but slow down the processing.
