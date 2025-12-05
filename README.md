# Surface Detection for Neuropixel Recordings

## Attribution

The channel quality detection algorithms (dead, noisy, and outside channel detection) used in this tool are derived from the [IBL sorter pipeline](https://github.com/int-brain-lab/ibl-sorter). The pipeline is described in the International Brain Laboratory's technical documentation: [https://doi.org/10.6084/m9.figshare.19705522.v4](https://doi.org/10.6084/m9.figshare.19705522.v4).

## Overview

This tool detects the brain surface and identifies bad channels in Neuropixel electrophysiology recordings. It processes SpikeGLX binary files (.bin or .cbin) and provides an interactive visualization for manual verification and adjustment of the detected surface channel.

The tool analyzes recordings to classify channels as:
- **Dead channels**: Low signal correlation, likely non-functional
- **Noisy channels**: High-frequency power or excessive correlation, indicating noise
- **Outside channels**: Channels above the brain surface (in air/saline)

![Surface Detection Example](example_image.png)

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

Or use the `--bin_file` flag:

```bash
python main.py --bin_file /path/to/recording.ap.bin [OPTIONS]
```

### Command Line Options

**File input:**
- `PATH` or `--bin_file PATH` - Path to the .ap.bin or .ap.cbin file

**Detection preprocessing:**
- `--cmr` - Apply Common Median Referencing to data before running detection algorithms
- `--hf FREQUENCY` - Apply highpass filter (Hz) to data before detection (e.g., `--hf 300`)
- `--n_chunks N` - Number of time chunks to analyze (default: 20)
- `--spike_threshold THRESH` - Spike detection threshold in multiples of MAD (default: -5.0)

**Debug options:**
- `--debug` - Enable debug mode: prints detailed detection info and saves intermediate data to .npy files

### Examples

**Basic usage with GUI:**
```bash
python main.py
```

**Command line with Common Median Referencing:**
```bash
python main.py recording.ap.bin --cmr
```

**With preprocessing and custom parameters:**
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
- **Dead channels**: Channels with low high-frequency coherence (blue)
- **Noisy channels**: Channels with high power or correlation (red/orange)  
- **Outside channels**: Channels detected above the brain surface (yellow/green)
- **Voltage heatmap**: Raw voltage snippet with optional filtering
- **Firing rate**: Spike activity across channels

**Interaction:**
- Click on any subplot to manually select a different surface channel
- Hover over the plots to see channel numbers and values in the bottom-left corner

The tool saves the final selection to a `.surface_channel.txt` file next to the input file.

## Multi-Shank Support

For multi-shank probes (e.g., Neuropixels 2.0), the tool automatically detects shanks and processes each independently. Each shank gets its own interactive plot and surface channel output.

## Output

The tool saves:
- `<filename>.surface_channel.txt` - The detected (or manually selected) surface channel number
- `<filename>.debug.npy` - Debug data with detection features (if `--debug` flag is used)
