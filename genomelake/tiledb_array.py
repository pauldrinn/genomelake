from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
import tiledb

GENOME_DOMAIN_NAME = "genome_coord"
SECONDARY_DOMAIN_NAME = "signal_coord"
GENOME_VALUE_NAME = "v"

DEFAULT_GENOME_TILE_EXTENT = 9000


def write_tiledb(arr, path, overwrite=True):
    """Write a tiledb to disk.
    """
    if os.path.exists(path) and os.path.isdir(path) and overwrite:
        shutil.rmtree(path)

    if os.path.exists(path):
        raise FileExistsError("Output path {} already exists".format(path))

    
    n = arr.shape[0]
    if arr.ndim == 2:
        m = arr.shape[1]
    n_tile_extent = min(DEFAULT_GENOME_TILE_EXTENT, n)
    
    if arr.ndim > 2:
        raise ValueError("tiledb backend only supports 1D or 2D arrays")

    dom = tiledb.Domain(
        tiledb.Dim(GENOME_DOMAIN_NAME, domain=(0, n - 1), tile=n_tile_extent, dtype="uint32"),
        *([tiledb.Dim(SECONDARY_DOMAIN_NAME, domain=(0, m - 1), tile=m, dtype="uint32")]
            if arr.ndim == 2 else [])
    )

    schema = tiledb.ArraySchema(
        domain=dom,
        attrs=[
            tiledb.Attr(
                GENOME_VALUE_NAME,
                dtype="float32",
                filters=tiledb.FilterList([tiledb.ByteShuffleFilter(), tiledb.LZ4Filter()])
            )
        ],
        cell_order="row-major",
        tile_order="row-major",
    )

    tiledb.Array.create(path, schema)
    with tiledb.open(path, mode="w") as A:
        A[:] = {GENOME_VALUE_NAME: arr.astype(np.float32)}


def load_tiledb(path):
    """Load a TileDB array."""
    #with tiledb.open(path, mode="r") as A:
    return TDBDenseArray(tiledb.open(path, mode="r")) #TDBDenseArray(A)


class TDBDenseArray:
    """A read-only wrapper of tiledb.DenseArray"""

    def __init__(self, array):
        self._arr = array

    def __getitem__(self, key):
        return self._arr[key][GENOME_VALUE_NAME]

    def __setitem__(self, key, item):
        raise NotImplemented("TDBDenseArray is read-only")

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self):
        return self._arr.ndim
