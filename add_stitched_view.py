import os

import mobie
import numpy as np
import pandas as pd

from elf.transformation import affine_matrix_3d, native_to_bdv
from common import compute_clims, CHANNEL_TO_NAME, ROOT, SEG_NAMES, SPOT_RADIUS


def get_source_transforms(pos_to_sources):
    table_path = "./spot_table_data/embryo3_fovs_scaled.csv"
    fov_table = pd.read_csv(table_path)
    positions = fov_table["fov"] - 1
    x0, y0 = fov_table["bound_x_1"], fov_table["bound_y_1"]
    x0 = (x0 - x0.min()) * 0.17
    y0 = (y0 - y0.min()) * 0.17

    trafos = {}
    for pos, trans_x, trans_y in zip(positions, x0, y0):
        trafo = affine_matrix_3d(translation=(0, trans_y, trans_x))
        trafo = native_to_bdv(trafo, invert=False)
        trafos[pos] = trafo

    source_transforms = [
        mobie.metadata.get_affine_source_transform(
            sources=pos_to_sources[pos], parameters=trafos[pos])
        for pos in positions
    ]

    return source_transforms


def add_stitched_view(ds_name, clims, view_name, add_spots, add_segmentation):
    ds_folder = os.path.join(ROOT, ds_name)
    sources = mobie.metadata.read_dataset_metadata(ds_folder)["sources"]

    # TODO don't hardcode
    positions = list(range(40))

    # create a table source and table for the positions in the view
    table_source = "positions"
    if table_source not in sources:
        table = pd.DataFrame.from_dict({
            "region_id": positions,
            "position_name": [f"MMStack_Pos{i}" for i in positions],
        })
        mobie.metadata.add_regions_to_dataset(ds_folder, table_source, table)

    image_suffixes = list(CHANNEL_TO_NAME.values())
    spot_suffixes = ["genes"] if add_spots else []
    seg_suffixes = SEG_NAMES if add_segmentation else []

    suffixes = image_suffixes + spot_suffixes + seg_suffixes

    source_names = [name for name in sources if any(name.endswith(suffix) for suffix in suffixes)]
    pos_to_sources = {
        pos: [source for source in source_names if source.split("_")[1] == f"Pos{pos}"]
        for pos in positions
    }
    source_transforms = get_source_transforms(pos_to_sources)

    sources = [[source for source in source_names if suffix in source] for suffix in suffixes]
    display_group_names = suffixes

    clims = compute_clims(ds_name)
    chan_to_name_rev = {v: str(k) for k, v in CHANNEL_TO_NAME.items()}
    channel_clims = {
        suffix: [
            np.min([clim[chan_to_name_rev[suffix]][0] for clim in clims.values()]),
            np.max([clim[chan_to_name_rev[suffix]][1] for clim in clims.values()]),
        ] for suffix in image_suffixes
    }

    display_settings = [
        {"contrastLimits": channel_clims[suffix], "visible": i == 0} for i, suffix in enumerate(image_suffixes)
    ]
    display_settings += [
        {"lut": "glasbey", "opacity": 0.5, "visible": False, "showTable": False, "spotRadius": SPOT_RADIUS}
    ] * len(spot_suffixes)
    display_settings += [
        {"lut": "glasbey", "opacity": 0.5, "visible": False, "showTable": False}
    ] * len(seg_suffixes)

    # the region display
    region_displays = {
        "positions": {"sources": pos_to_sources, "tableSource": table_source, "showTable": True}
    }

    mobie.create_view(
        ds_folder, view_name, sources,
        display_settings=display_settings,
        source_transforms=source_transforms,
        display_group_names=display_group_names,
        region_displays=region_displays,
        overwrite=True
    )


# TODO add all other sources
# TODO add the region display
def main():
    ds_name = "embryo3"
    clims = compute_clims(ds_name)
    # add_stitched_view(ds_name, clims, "stitched", add_spots=False, add_segmentation=False)
    add_stitched_view(ds_name, clims, "stitched-full", add_segmentation=True, add_spots=True)


if __name__ == "__main__":
    main()
