# Surface Detection for Neuropixel Recordings

## Attribution

The channel quality detection algorithms (dead, noisy, and outside channel detection) used in this tool are derived from the [IBL sorter pipeline](https://github.com/int-brain-lab/ibl-sorter). The pipeline is described in the International Brain Laboratory's technical documentation: [https://doi.org/10.6084/m9.figshare.19705522.v4](https://doi.org/10.6084/m9.figshare.19705522.v4).

## Overview

This tool detects the brain surface and bad channels in Neuropixel recordings (SpikeGLX .bin/.cbin). It provides an automated estimate alongside an interactive visualization, allowing users to verify and adjust the surface channel based on signal features. It makes some informed guesses about the surface location, but the script is mainly meant to provide the user with useful information to make an informed decision about the surface channel.

The tool analyzes recordings to classify channels as:
- **Dead channels**: Low signal correlation, likely non-functional
- **Noisy channels**: High-frequency (>80% Nyquist) power or excessive correlation, indicating noise
- **Outside channels**: Channels above the brain surface (in air/saline)

![Surface Detection Example](example.png)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- scipy
- matplotlib
- mtscomp (for reading .cbin compressed files)
- neuropixel
- iblutil

## Usage

### GUI Mode (Default)

Run without arguments to launch the file picker and options dialog:

```bash
python main.py
```

This opens:
1. A file selection dialog to choose your .bin or .cbin file
2. An options dialog to configure preprocessing settings
3. An interactive plot for reviewing and adjusting the detected surface channel

### Command Line Mode

Specify the file path and options directly:

```bash
python main.py /path/to/recording.ap.bin [OPTIONS]
```

### Command Line Options

**File input:**
- `PATH` or `--bin_file PATH` - Path to the .ap.bin or .ap.cbin file

**Detection preprocessing:**
- `--cmr` - Apply Common Median Referencing to data before running detection algorithms
- `--hf FREQUENCY` - Apply highpass filter (Hz) to data before detection (e.g., `--hf 300`)
- `--n_chunks N` - Number of time chunks to analyze (default: 20)
- `--spike_threshold THRESH` - Spike detection threshold in multiples of MAD (Median Absolute Deviation, default: -5.0)

**Debug options:**
- `--debug` - Enable debug mode: prints detailed detection info and saves intermediate data to .npy files

### Examples

**Basic usage with file dialog:**
```bash
python main.py
```

**Command line with preprocessing and custom parameters:**
```bash
python main.py recording.ap.bin --cmr --hf 300 --n_chunks 30 --spike_threshold -4.5
```

**Debug mode to save intermediate features:**
```bash
python main.py recording.ap.bin --debug
```

## Preprocessing Options

The `--cmr` and `--hf` flags control the preprocessing applied to the data **before** the detection algorithms analyze it. These options affect which channels are classified as dead, noisy, or outside.

The interactive visualization includes checkboxes for CMR and highpass filtering. These checkboxes **only change the displayed raw voltage snippet** and do not affect the channel detection results.

## Interactive Plot

The visualization shows:
- **Dead channels**: Low signal correlation (blue)
- **Noisy channels**: High high-frequency power (red)
- **Outside channels**: Low low-frequency coherence (green)
- **LF Power**: Power < 100 Hz
- **Gamma Power**: Power 40-100 Hz (can be useful to identify cortex)
- **Spike Amplitude**: Median amplitude of multi-unit activity (useful to identify cortical layers)
- **Firing rate**: Spike activity across channels
- **Voltage heatmap**: Raw voltage snippet with optional filtering

**Interaction:**
- Click on any subplot to manually select a different surface channel
- Hover over the plots to see channel numbers and values in the bottom-left corner

## Multi-Shank Support

For multi-shank probes (e.g., Neuropixels 2.0), the tool automatically detects shanks and processes each independently, sequentially. Each shank gets its own interactive plot and surface channel output.

## Output

The tool saves:
- `<filename>.surface_channel.txt` - The detected (or manually selected) surface channel number
- `<filename>.debug.npy` - Debug data with detection features (if `--debug` flag is used)
