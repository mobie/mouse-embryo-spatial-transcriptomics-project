import json
import os
from glob import glob

import mobie
import numpy as np
import pandas as pd
import zarr

from skimage.measure import regionprops
from tqdm import tqdm

ROOT = "/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff"
CHANNEL_TO_NAME = {0: "membrane-marker1", 1: "membrane-marker2", 2: "membrane-marker3", 3: "nucleus-marker"}
# NOTE: we only add the cell segmentation,
# since the nuclei are just given as binary mask and don't provide much information
# SEG_NAMES = ["cells", "nuclei"]
SEG_NAMES = ["cells"]


#
# Functionality for adding source metadata
#

def add_image_data(ds_name):
    ds_folder = os.path.join(ROOT, ds_name)
    if not mobie.metadata.dataset_exists(ROOT, ds_name):
        mobie.metadata.create_dataset_metadata(ds_folder)
        mobie.metadata.add_dataset(ROOT, ds_name, is_default=False)

    image_files = glob(os.path.join(ds_folder, "*.ome.zarr"))
    n_images = len(image_files)
    file_paths = [
        os.path.join(ds_folder, f"MMStack_Pos{i}.ome.zarr") for i in range(n_images)
    ]

    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]
    n_channels = len(CHANNEL_TO_NAME)
    for file_path in file_paths:
        fname = os.path.basename(file_path)[:-len(".ome.zarr")]
        with zarr.open(file_path, "r") as f:
            this_channels = f["0"].shape[0]
            assert this_channels == n_channels

        for channel_id in range(n_channels):
            name = f"{fname}_{CHANNEL_TO_NAME[channel_id]}"
            if name in sources:
                continue
            mobie.metadata.add_source_to_dataset(
                ds_folder, "image", name, file_format="ome.zarr",
                image_metadata_path=file_path,
                view={}, channel=channel_id,
            )


def read_resolution(seg_path):
    attrs_file = os.path.join(seg_path, ".zattrs")
    assert os.path.exists(attrs_file), attrs_file
    with open(attrs_file) as f:
        attrs = json.load(f)
    resolution = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert len(resolution) == 3
    return resolution


# NOTE: computing the default table via mobie-python does not work for ome.zarr
# (my guess because reading zarr with / separator does currently not work in c++)
# so we compute the default tables manually
# this should be refactored somewhere in mobie-python
def require_default_table(file_path, seg_key, source_name, table_folder):
    table_path = os.path.join(table_folder, "default.tsv")
    if os.path.exists(table_path):
        return
    os.makedirs(table_folder, exist_ok=True)

    resolution = read_resolution(os.path.join(file_path, seg_key[:-2]))
    with zarr.open(file_path, "r") as f:
        seg = f[seg_key][:]
    props = regionprops(seg)
    table = {
        "label_id": [prop.label for prop in props],
        "anchor_x": [prop.centroid[2] * resolution[2] for prop in props],
        "anchor_y": [prop.centroid[1] * resolution[1] for prop in props],
        "anchor_z": [prop.centroid[0] * resolution[0] for prop in props],
        "bb_min_x": [prop.bbox[2] * resolution[2] for prop in props],
        "bb_min_y": [prop.bbox[1] * resolution[1] for prop in props],
        "bb_min_z": [prop.bbox[0] * resolution[0] for prop in props],
        "bb_max_x": [prop.bbox[5] * resolution[2] for prop in props],
        "bb_max_y": [prop.bbox[4] * resolution[1] for prop in props],
        "bb_max_z": [prop.bbox[3] * resolution[0] for prop in props],
        "n_pixels": [prop.area for prop in props],
    }
    table = pd.DataFrame.from_dict(table)
    table.to_csv(table_path, sep="\t", na_rep="nan", index=False, float_format="%.2f")


def add_segmentation_data(ds_name):
    ds_folder = os.path.join(ROOT, ds_name)

    image_files = glob(os.path.join(ds_folder, "*.ome.zarr"))
    n_images = len(image_files)
    file_paths = [
        os.path.join(ds_folder, f"MMStack_Pos{i}.ome.zarr") for i in range(n_images)
    ]
    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]

    for file_path in tqdm(file_paths, desc="Add segmentations"):
        fname = os.path.basename(file_path)[:-len(".ome.zarr")]
        for seg_name in SEG_NAMES:
            name = f"{fname}_{seg_name}"
            if name in sources:
                continue

            seg_path = os.path.join(file_path, "labels", seg_name)
            assert os.path.exists(seg_path)

            table_folder = os.path.join(ds_folder, "tables", name)
            require_default_table(file_path, f"labels/{seg_name}/0", name, table_folder)
            mobie.metadata.add_source_to_dataset(
                ds_folder, "segmentation", name,
                file_format="ome.zarr",
                image_metadata_path=seg_path, view={},
                table_folder=table_folder
            )


#
# Functionality for adding view metadata
#

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


def add_position_views(ds_name, clims):
    ds_folder = os.path.join(ROOT, ds_name)

    n_positions = len(glob(os.path.join(ds_folder, "*.ome.zarr")))
    position_names = [f"MMStack_Pos{i}" for i in range(n_positions)]

    default_view = None
    for name in position_names:

        # add the image sources for this position
        sources = [[f"{name}_{channel_name}"] for channel_name in CHANNEL_TO_NAME.values()]
        display_group_names = list(CHANNEL_TO_NAME.values())
        source_types = len(sources) * ["image"]
        display_settings = [
            {
                "contrastLimits": clims[name][str(channel)],
                "visible": channel == 0,  # only show the first channel by default
            }
            for channel in CHANNEL_TO_NAME.keys()
        ]

        # add the segmentation sources for this view
        seg_sources = [[f"{name}_{seg_name}"] for seg_name in SEG_NAMES]
        sources.extend(seg_sources)
        display_group_names.extend(SEG_NAMES)
        source_types.extend(len(seg_sources) * ["segmentation"])
        display_settings.extend([
            {"lut": "glasbey", "opacity": 0.5, "visible": False, "showTable": False} for seg_name in SEG_NAMES
        ])

        # TODO add the spot sources for this view

        position_view = mobie.metadata.get_view(display_group_names, source_types, sources, display_settings,
                                                is_exclusive=True, menu_name="positions")
        mobie.metadata.add_view_to_dataset(ds_folder, name, position_view)
        if default_view is None:
            default_view = position_view
            default_view["uiSelectionGroup"] = "bookmarks"

    mobie.metadata.add_view_to_dataset(ds_folder, "default", default_view)


def add_grid_view(ds_name, clims):
    ds_folder = os.path.join(ROOT, ds_name)

    n_positions = len(glob(os.path.join(ds_folder, "*.ome.zarr")))
    position_names = [f"MMStack_Pos{i}" for i in range(n_positions)]

    # add image sources
    sources = [
        [f"{name}_{channel_name}" for channel_name in CHANNEL_TO_NAME.values()]
        for name in position_names
    ]
    display_groups = {
        f"{name}_{channel_name}": channel_name
        for name in position_names for channel_name in CHANNEL_TO_NAME.values()
    }
    channel_clims = {
        cid: [
            np.min([clim[str(cid)][0] for clim in clims.values()]),
            np.max([clim[str(cid)][1] for clim in clims.values()]),
        ] for cid in CHANNEL_TO_NAME
    }
    display_group_settings = {
        channel_name: {
            "contrastLimits": channel_clims[cid],
            "visible": cid == 0,
        }
        for cid, channel_name in CHANNEL_TO_NAME.items()
    }

    # add segmentation sources
    sources = [
        pos_sources + [f"{name}_{seg_name}" for seg_name in SEG_NAMES]
        for pos_sources, name in zip(sources, position_names)
    ]
    display_groups.update({
        f"{name}_{seg_name}": seg_name
        for name in position_names for seg_name in SEG_NAMES
    })
    display_group_settings.update({
        seg_name: {"lut": "glasbey", "opacity": 0.5, "visible": False, "showTable": False}
        for seg_name in SEG_NAMES
    })

    # TODO add spot sources

    # create a table source and table for the grid view
    table_source = "all_positions"
    table = pd.DataFrame.from_dict({
        "region_id": range(n_positions),
        "position_name": position_names,
    })
    mobie.metadata.add_regions_to_dataset(ds_folder, table_source, table)

    view_name = "all_positions"
    grid_view = mobie.metadata.get_grid_view(
        ds_folder, view_name, sources,
        menu_name="bookmarks", table_source=table_source,
        display_groups=display_groups,
        display_group_settings=display_group_settings,
    )
    mobie.metadata.add_view_to_dataset(ds_folder, view_name, grid_view)


def create_dataset(ds_name):
    add_image_data(ds_name)
    add_segmentation_data(ds_name)
    # TODO
    # add_spot_table_data()

    clims = compute_clims(ds_name)
    add_position_views(ds_name, clims)
    add_grid_view(ds_name, clims)

    # TODO
    # add_idr_data_links()


def create_project():
    if not mobie.metadata.project_exists(ROOT):
        mobie.metadata.create_project_metadata(ROOT)
    dataset_names = ["embryo3"]
    for ds_name in dataset_names:
        create_dataset(ds_name)
    mobie.validation.validate_project(ROOT)


if __name__ == "__main__":
    create_project()
