# Surface Detection Script Documentation

This document outlines the functionality of the standalone surface detection script and clarifies its relationship with the main IBL pipeline.

## Goal

The primary goal of this script is to extract the bad channel detection algorithm, which includes surface detection (i.e., identifying channels outside the brain), from the `ibl-sorter` pipeline into a self-contained, runnable script.

## Current Status

The surface detection script has been successfully extracted from the IBL pipeline and is now fully functional. The script correctly replicates the bad channel detection algorithm, matching the IBL pipeline output to within floating point precision (max difference: 5.5e-05).

## Debugging Journey (Resolved)

During extraction, we encountered discrepancies between our output and the reference data. After investigation, we identified:

1.  **Typo in xcor_lf calculation:** The initial copy had `detrend(xcor, 11)` instead of `detrend(xcorf, 11)` on line 97 of `run_detection.py`. Fixed.

2.  **System-wide vs. local ibldsp package:** The iblsorter package installs ibldsp system-wide. When the IBL pipeline runs, it imports from `/home/mdrueke/anaconda3/lib/python3.13/site-packages/ibldsp/`, NOT the local copy. The `run_detection.py` file replicates this code for standalone use.

3.  **Single chunk vs. aggregation:** The IBL pipeline's `get_good_channels` function:
    *   Loops over 25 time chunks (0.4s each) throughout the recording
    *   Calls `detect_bad_channels` on each chunk
    *   Takes the MODE of channel labels across chunks for robust classification
    *   Saves features (xfeats) from ONLY THE LAST chunk for plotting/QC
    
    The standalone script now mimics this by calling `detect_bad_channels` on a single chunk (the last time point at t=3116s), matching the IBL reference data.

4.  **Alternative robust approach:** The `detect_bad_channels_cbin` function in `run_detection.py` provides a more robust alternative that aggregates features across multiple chunks. Use this if you want detection on the full recording rather than a single snapshot.

## How the IBL Pipeline Works

The bad channel detection logic is orchestrated by the following functions:

1.  **`iblsorter.preprocess.get_good_channels`**:
    *   This is the high-level function that manages the QC process.
    *   It calls `detect_bad_channels` over several chunks of the data file to get robust channel labels and features.
    *   Crucially, for plotting, it applies a `scipy.signal.medfilt` to the `xcor_lf` feature before passing the data to the plotting function.
    *   It saves the final plot as `_iblqc_channel_detection.png`.

2.  **`ibldsp.voltage.detect_bad_channels`**:
    *   This is the core function that computes metrics to classify channels as good, dead, noisy, or outside the brain.
    *   It calculates several features, including `xcor_lf` (LF coherence), which is used to find the brain surface.
    *   The `xcor_lf` feature is calculated by cross-correlating each channel with a median reference trace and then applying a `detrend` operation. The specific `detrend` logic is what creates the characteristic shape of the curve at the probe edges.

## Code Location in This Directory (`surfaceDetection/`)

The IBL pipeline logic has been copied into the following files within this directory:

1.  **`run_detection.py`**:
    *   Contains `detect_bad_channels_cbin`: A wrapper function that reads data in chunks from a raw data file and passes them to `detect_bad_channels`. This mimics the chunking behavior of the IBL pipeline.
    *   Contains `detect_bad_channels`: A direct copy of the core logic from `ibldsp/voltage.py`.

2.  **`plotting.py`**:
    *   Contains `show_channels_labels`: A copy of the plotting function from `ibldsp/plots.py` used to visualize the detection results.

3.  **`reader.py`**:
    *   Contains the `Reader` class, a copy of the `spikeglx.Reader` from the IBL pipeline, which handles reading both `.bin` and compressed `.cbin` files.

4.  **`main.py`**:
    *   The main entry point that uses the functions and classes above to run the detection and generate a plot.
