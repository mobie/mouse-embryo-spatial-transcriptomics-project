import os
from glob import glob

import numpy as np
import vigra
import z5py
from common import ROOT
from mobie.tables import compute_default_table
from skimage.transform import resize


def fix_segmentation_issues(seg_path, table_folder):
    with z5py.File(seg_path, "a", dimension_separator="/") as f:
        ds = f["labels/cells"]
        resolution = ds.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        seg = ds["0"][:]

        # bigger 4000 pix: this is background
        bg_threshold = 4000
        new_seg = np.zeros_like(seg).astype(seg.dtype)

        offset = 0
        for z in range(seg.shape[0]):
            seg_z = seg[z]
            ids, counts = np.unique(seg_z, return_counts=True)

            bg_ids = ids[counts > bg_threshold]
            seg_z[np.isin(seg_z, bg_ids)] = 0
            seg_z, max_id, _ = vigra.analysis.relabelConsecutive(seg_z)

            seg_z[seg_z != 0] += offset

            new_seg[z] = seg_z
            offset += max_id

        ds["0"][:] = new_seg
        ds["0"].attrs["maxId"] = int(new_seg[-1].max())
        for i in range(1, len(ds)):
            dset = ds[str(i)]
            dset[:] = resize(
                new_seg, dset.shape, order=0, anti_aliasing=False, preserve_range=True
            ).astype(new_seg.dtype)

    table_path = os.path.join(table_folder, "default.tsv")
    seg_key = "labels/cells/0"
    compute_default_table(seg_path, seg_key, table_path, resolution,
                          tmp_folder=f"tmps/tab_{os.path.basename(seg_path)}",
                          target="local", max_jobs=8)


def main():
    ds_name = "embryo3"
    ds_folder = os.path.join(ROOT, ds_name)

    seg_paths = glob(os.path.join(ds_folder, "*.ome.zarr"))
    seg_paths.sort()
    assert len(seg_paths) > 0

    table_folders = glob(os.path.join(ds_folder, "tables", "*_cells"))
    table_folders.sort()
    assert len(seg_paths) == len(table_folders)

    for seg_path, table_folder in zip(seg_paths, table_folders):
        print(seg_path, table_folder)
        fix_segmentation_issues(seg_path, table_folder)


def undo():
    import imageio
    from tqdm import tqdm

    paths = glob("/g/kreshuk/data/marioni/shila/mouse-atlas-2020/segmentation/embryo3/cells/MMStack_Pos*.ome.tif")
    paths.sort()

    for path in tqdm(paths):
        seg = imageio.volread(path)
        fname = os.path.basename(path).replace(".tif", ".zarr")
        out_path = f"/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff/embryo3/{fname}"
        out_key = "labels/cells"
        print(seg.shape)

        with z5py.File(out_path, "a", dimension_separator="/") as f:
            attrs = {k: v for k, v in f.attrs.items()}
            attrs["multiscales"][0]["datasets"] = [
                {"path": "0", "coordinateTransformations": [{'scale': [4.0, 0.68, 0.68], 'type': 'scale'}]},
                {"path": "1", "coordinateTransformations": [{'scale': [4.0, 1.36, 1.36], 'type': 'scale'}]},
                {"path": "2", "coordinateTransformations": [{'scale': [4.0, 2.72, 2.72], 'type': 'scale'}]},
            ]
            attrs["multiscales"][0]["axes"] = [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ]
            attrs["multiscales"][0]["name"] = "cells"

            del f[out_key]
            g = f.create_group(out_key)
            g.attrs["multiscales"] = attrs["multiscales"]
            g.attrs["image-label"] = {}

            ds = g.create_dataset("0", shape=seg.shape, chunks=(1, 512, 512), dtype=seg.dtype)
            ds[:] = seg
            g.create_dataset("1", shape=(6, 256, 256), chunks=(1, 256, 256), dtype=seg.dtype)
            g.create_dataset("2", shape=(6, 128, 128), chunks=(1, 128, 128), dtype=seg.dtype)


if __name__ == "__main__":
    undo()
    main()
