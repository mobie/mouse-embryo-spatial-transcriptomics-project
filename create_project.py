import json
import os
from glob import glob

import mobie
import numpy as np
import pandas as pd
import zarr

from skimage.measure import regionprops
from tqdm import tqdm, trange

from common import compute_clims, CHANNEL_TO_NAME, ROOT, SEG_NAMES, SPOT_RADIUS


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
    for file_path in file_paths:
        fname = os.path.basename(file_path)[:-len(".ome.zarr")]
        with zarr.open(file_path, "r") as f:
            this_channels = range(f["0"].shape[0])

        for channel_id, channel_name in CHANNEL_TO_NAME.items():
            assert channel_id in this_channels
            name = f"{fname}_{channel_name}"
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


def scale_table_and_compute_bounding_box(table, ds_folder, pos):
    # read the image metadata
    image_path = os.path.join(ds_folder, f"MMStack_Pos{pos}.ome.zarr")
    with zarr.open(image_path, "r") as f:
        mscales = f.attrs["multiscales"][0]
        shape = f["0"].shape

    # get the scale information
    scale = mscales["datasets"][0]["coordinateTransformations"][0]["scale"]

    # get rid of the channel axis
    assert len(scale) == 4
    scale = scale[1:]
    assert len(shape) == 4
    shape = shape[1:]

    # scale the table coordinate columns
    table["x"] *= scale[2]
    table["y"] *= scale[1]
    table["z"] *= scale[0]

    # bounding box along the z axis: min max of the z coordinates
    min_z, max_z = table["z"].min(), table["z"].max()
    # the other bounding boxes correspond to the image shape
    min_y, max_y = 0, shape[1] * scale[1]
    min_x, max_x = 0, shape[2] * scale[2]

    return table, [min_z, min_y, min_x], [max_z, max_y, max_x]


def add_spot_data(ds_name):
    ds_folder = os.path.join(ROOT, ds_name)
    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]
    n_pos = len(glob(os.path.join(ds_folder, "*.ome.zarr")))

    # temporary loc for the spot table data that was downloaded from dropbox
    table_root = "./spot_table_data"
    for pos in trange(n_pos, desc="Add spot tables"):
        name = f"MMStack_Pos{pos}_genes"
        if name in sources:
            continue

        table_path = os.path.join(
            table_root,
            f"segmentedData-Tim-120919-Pos{pos}-1error-sqrt6-2020-02-12.csv"
        )
        assert os.path.exists(table_path)
        table = pd.read_csv(table_path)
        table.insert(
            loc=0, column="spot_id", value=np.arange(1, len(table) + 1).astype("uint64")
        )
        table, bb_min, bb_max = scale_table_and_compute_bounding_box(table, ds_folder, pos)
        mobie.add_spots(table, ROOT, ds_name, name, unit="micrometer",
                        bounding_box_min=bb_min, bounding_box_max=bb_max,
                        view={})


#
# Functionality for adding view metadata
#


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
            {"lut": "glasbey", "opacity": 0.5, "visible": False, "showTable": False}
            for seg_name in SEG_NAMES
        ])

        # add the spot sources for this view
        spot_sources = [[f"{name}_genes"]]
        sources.extend(spot_sources)
        display_group_names.extend(["genes"])
        source_types.extend(["spots"])
        display_settings.extend([
            {"lut": "glasbey", "opacity": 0.5,
             "visible": False, "showTable": False, "spotRadius": SPOT_RADIUS}
        ])

        position_view = mobie.metadata.get_view(
            display_group_names, source_types, sources, display_settings,
            is_exclusive=True, menu_name="positions"
        )
        mobie.metadata.add_view_to_dataset(ds_folder, name, position_view)
        if default_view is None:
            default_view = position_view
            default_view["uiSelectionGroup"] = "bookmarks"

    mobie.metadata.add_view_to_dataset(ds_folder, "default", default_view)


def create_dataset(ds_name):
    add_image_data(ds_name)
    add_segmentation_data(ds_name)
    add_spot_data(ds_name)

    clims = compute_clims(ds_name)
    add_position_views(ds_name, clims)


def create_project():
    if not mobie.metadata.project_exists(ROOT):
        mobie.metadata.create_project_metadata(ROOT)
    dataset_names = ["embryo3"]
    for ds_name in dataset_names:
        create_dataset(ds_name)
    mobie.validation.validate_project(ROOT)


if __name__ == "__main__":
    create_project()
