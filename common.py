import json
import os
from glob import glob

import numpy as np
import zarr
from tqdm import tqdm

ROOT = "/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff"
# CHANNEL_TO_NAME = {
#    0: "membrane-marker1", 1: "membrane-marker2", 2: "membrane-marker3", 3: "nucleus-marker"
# }
CHANNEL_TO_NAME = {0: "membrane-marker", 3: "nucleus-marker"}

# NOTE: we only add the cell segmentation,
# since the nuclei are just given as binary mask and don't provide much information
# SEG_NAMES = ["cells", "nuclei"]
SEG_NAMES = ["cells"]

# TODO determine a good spot radius
SPOT_RADIUS = 0.5


def compute_clims(ds_name):
    ds_folder = os.path.join(ROOT, ds_name)
    clim_file = "./clims.json"

    if os.path.exists(clim_file):
        with open(clim_file) as f:
            clims = json.load(f)
        return clims

    clims = {}
    image_files = glob(os.path.join(ds_folder, "*.ome.zarr"))
    for file_path in tqdm(image_files, desc="Compute contrast limits"):
        fname = os.path.basename(file_path)[:-len(".ome.zarr")]
        with zarr.open(file_path, "r") as f:
            ds = f["0"]
            assert ds.shape[0] == len(CHANNEL_TO_NAME)
            data = ds[:]
            clims[fname] = {
                c: [float(np.percentile(data[c], 1.0)), float(np.percentile(data[c], 99.0))]
                for c in range(ds.shape[0])
            }

    with open(clim_file, "w") as f:
        json.dump(clims, f)

    return clims
