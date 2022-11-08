import os
from glob import glob

import numpy as np
import pandas as pd
import vigra
import z5py
from common import ROOT
from skimage.measure import regionprops
from skimage.transform import resize
from tqdm import tqdm


def fix_segmentation(seg_path):
    with z5py.File(seg_path, "a", dimension_separator="/") as f:
        ds = f["labels/cells"]
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


def _compute_table(seg_path, seg_key, table_path):
    ndim = 3
    with z5py.File(seg_path, "r", dimension_separator="/") as f:
        seg = f[seg_key]
        resolution = seg.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        seg = seg["0"][:]

    # centers = vigra.filters.eccentricityCenters(seg.astype("uint32"))
    tab = []
    for z in range(seg.shape[0]):
        props = regionprops(seg[z:z+1])
        ztab = np.array([
            [int(p.label)]
            + [ce / res for ce, res in zip(p.centroid, resolution)]
            + [float(bb) / res for bb, res in zip(p.bbox[:ndim], resolution)]
            + [float(bb) / res for bb, res in zip(p.bbox[ndim:], resolution)]
            + [p.area]
            for p in props
        ])
        ztab[:, 1] += resolution[0] * z
        ztab[:, 4] += resolution[0] * z
        ztab[:, 7] += resolution[0] * z
        tab.append(ztab)

    tab = np.concatenate(tab, axis=0)
    col_names = ["label_id",
                 "anchor_z", "anchor_y", "anchor_x",
                 "bb_min_z", "bb_min_y", "bb_min_x",
                 "bb_max_z", "bb_max_y", "bb_max_x", "n_pixels"]
    assert tab.shape[1] == len(col_names), f"{tab.shape}, {len(col_names)}"
    tab = pd.DataFrame(tab, columns=col_names)
    tab.to_csv(table_path, sep="\t", index=False)
    print(table_path, ":", len(tab))


def fix_table(seg_path, table_folder):
    table_path = os.path.join(table_folder, "default.tsv")
    seg_key = "labels/cells"
    _compute_table(seg_path, seg_key, table_path)


def main():
    ds_name = "embryo3"
    ds_folder = os.path.join(ROOT, ds_name)

    seg_paths = glob(os.path.join(ds_folder, "*.ome.zarr"))
    seg_paths.sort()
    assert len(seg_paths) > 0

    # for seg_path in tqdm(seg_paths):
    #     fix_segmentation(seg_path)

    # seg_paths = ["/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff/embryo3/MMStack_Pos23.ome.zarr"]
    # table_folders = ["/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff/embryo3/tables/MMStack_Pos23_cells"]
    for seg_path in seg_paths:
        pos = os.path.basename(seg_path).split("_")[1].split(".")[0]
        table_folder = os.path.join(ds_folder, "tables", f"MMStack_{pos}_cells")
        assert os.path.exists(table_folder), table_folder
        fix_table(seg_path, table_folder)


def undo():
    import imageio

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
    # undo()
    main()
