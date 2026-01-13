
import copy
import hashlib
import numbers
import logging
from pathlib import Path
import numpy as np

# Try importing tqdm, but fallback if not present to avoid hard dependency if possible
try:
    from tqdm import tqdm
except ImportError:
    # Minimal mock of tqdm if strictly necessary, but better to just not fail on import
    # For exact copy, we'd add tqdm to requirements. 
    # Let's assume we can add it or make it optional.
    def tqdm(*args, **kwargs):
        class MockPbar:
            def update(self, *args): pass
            def close(self): pass
        return MockPbar()

# --- Neuropixel Geometry Utils (from ibl-neuropixel) ---

# Sample to volt conversion factors
S2V_AP = 2.34375e-06
S2V_LFP = 4.6875e-06
NC = 384
CHANNEL_GRID = {
    1: dict(DX=16, X0=11, DY=20, Y0=20),
    2: dict(DX=32, X0=27, DY=15, Y0=20),
    "NPultra": dict(DX=6, X0=0, DY=6, Y0=0),
}


def xy2rc(x, y, version=1):
    """
    converts the um indices to row/col coordinates.
    :param y: row coordinate on the probe
    :param x: col coordinate on the probe
    :param version: neuropixel major version 1 or 2
    :return: dictionary with keys x and y
    """
    version = np.floor(version) if isinstance(version, numbers.Number) else version
    grid = CHANNEL_GRID[version]
    col = (x - grid["X0"]) / grid["DX"]
    row = (y - grid["Y0"]) / grid["DY"]
    return {"col": col, "row": row}


def rc2xy(row, col, version=1):
    """
    converts the row/col indices to um coordinates.
    :param row: row index on the probe
    :param col: col index on the probe
    :param version: neuropixel major version 1 or 2
    :return: dictionary with keys x and y
    """
    version = np.floor(version) if isinstance(version, numbers.Number) else version
    grid = CHANNEL_GRID[version]
    x = col * grid["DX"] + grid["X0"]
    y = row * grid["DY"] + grid["Y0"]
    return {"x": x, "y": y}


def dense_layout(version=1, nshank=1):
    """
    Returns a dense layout indices map for neuropixel, as used at IBL
    :param version: major version number: 1 or 2 or 2.4
    :return: dictionary with keys 'ind', 'col', 'row', 'x', 'y'
    """
    ch = {
        "ind": np.arange(NC),
        "row": np.floor(np.arange(NC) / 2),
        "shank": np.zeros(NC),
    }

    if version == 1:  # version 1 has a dense layout, checkerboard pattern
        ch.update({"col": np.tile(np.array([2, 0, 3, 1]), int(NC / 4))})
    elif version == "NPultra":  # NPultra has 8 columns with square grid spacing
        ch.update({"row": np.floor(np.arange(NC) / 8)})
        ch.update({"col": np.tile(np.arange(8), int(NC / 8))})
    elif (
        np.floor(version) == 2 and nshank == 1
    ):  # single shank NP1 has 2 columns in a dense patter
        ch.update({"col": np.tile(np.array([0, 1]), int(NC / 2))})
    elif (
        np.floor(version) == 2 and nshank == 4
    ):  # the 4 shank version default is rather complicated
        shank_row = np.tile(np.arange(NC / 16), (2, 1)).T[:, np.newaxis].flatten()
        shank_row = np.tile(shank_row, 8)
        shank_row += (
            np.tile(
                np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, np.newaxis], (1, int(NC / 8))
            ).flatten()
            * 24
        )
        ch.update(
            {
                "col": np.tile(np.array([0, 1]), int(NC / 2)),
                "shank": np.tile(
                    np.array([0, 1, 0, 1, 2, 3, 2, 3])[:, np.newaxis], (1, int(NC / 8))
                ).flatten(),
                "row": shank_row,
            }
        )
    # for all, get coordinates
    ch.update(rc2xy(ch["row"], ch["col"], version=version))
    return ch


def adc_shifts(version=1, nc=NC):
    """
    Neuropixel NP1
    The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
    :param version: neuropixel major version 1 or 2
    :param nc: number of channels
    """
    if version == 1 or version == "NPultra":
        adc_channels = 12
        n_cycles = 13
        # version 1 uses 32 ADC that sample 12 channels each
    elif np.floor(version) == 2:
        # version 2 uses 24 ADC that sample 16 channels each
        adc_channels = n_cycles = 16
    adc = np.floor(np.arange(NC) / (adc_channels * 2)) * 2 + np.mod(np.arange(NC), 2)
    sample_shift = np.zeros_like(adc)
    for a in adc:
        sample_shift[adc == a] = np.arange(adc_channels) / n_cycles
    return sample_shift[:nc], adc[:nc]


def trace_header(version=1, nshank=1):
    """
    Returns the channel map for the dense layouts used at IBL.
    :param version: major version number: 1 or 2
    :param nshank: (defaults 1) number of shanks for NP2
    :return: dictionary with keys x, y, row, col, ind, adc and sampleshift vectors
    """
    h = dense_layout(version=version, nshank=nshank)
    h["sample_shift"], h["adc"] = adc_shifts(version=version)
    return h


# --- Bunch Class (from iblutil.util) ---

class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self, deep=False):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return copy.deepcopy(self) if deep else Bunch(super(Bunch, self).copy())

    def save(self, npz_file, compress=False):
        """
        Saves a npz file containing the arrays of the bunch.
        """
        if compress:
            np.savez_compressed(npz_file, **self)
        else:
            np.savez(npz_file, **self)

    @staticmethod
    def load(npz_file):
        """
        Loads a npz file containing the arrays of the bunch.
        """
        if not Path(npz_file).exists():
            raise FileNotFoundError(f'{npz_file}')
        return Bunch(np.load(npz_file))


# --- Hashing Utils (from iblutil.io.hashfile) ---

BUF_SIZE = 2**28  # 256 megs

def sha1(file_path, *args, **kwargs):
    """
    Computes sha1 hash in a memory reasoned way
    sha1_hash = hashfile.sha1(file_path)
    """
    return _hash_file(file_path, hashlib.sha1(), *args, **kwargs)


def _hash_file(file_path, hash_obj, progress_bar=None):
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    # by default prints a progress bar only for files above 512 Mb
    if progress_bar is None:
        progress_bar = file_size > (512 * 1024 * 1024)
    b = bytearray(BUF_SIZE)
    mv = memoryview(b)
    pbar = tqdm(total=np.ceil(file_size / BUF_SIZE), disable=not progress_bar)
    with open(file_path, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            hash_obj.update(mv[:n])
            pbar.update(1)
    pbar.close()
    return hash_obj.hexdigest()
