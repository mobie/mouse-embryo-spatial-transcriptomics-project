import os
from glob import glob

import mobie
import numpy as np
import pandas as pd

from common import compute_clims, CHANNEL_TO_NAME, ROOT, SEG_NAMES, SPOT_RADIUS


def add_grid_view(ds_name, clims, n_positions=None, view_name=None):
    ds_folder = os.path.join(ROOT, ds_name)

    if n_positions is None:
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

    # add spot sources
    sources = [
        pos_sources + [f"{name}_genes"] for pos_sources, name in zip(sources, position_names)
    ]
    display_groups.update({f"{name}_genes": "genes" for name in position_names})
    display_group_settings.update({
        "genes": {"lut": "glasbey", "opacity": 0.5, "visible": False, "showTable": False, "spotRadius": SPOT_RADIUS}
    })

    # create a table source and table for the grid view
    table_source = "all_positions"
    table = pd.DataFrame.from_dict({
        "region_id": range(n_positions),
        "position_name": position_names,
    })
    mobie.metadata.add_regions_to_dataset(ds_folder, table_source, table)

    view_name = "all_positions" if view_name is None else view_name
    grid_view = mobie.metadata.get_grid_view(
        ds_folder, view_name, sources,
        menu_name="bookmarks", table_source=table_source,
        display_groups=display_groups,
        display_group_settings=display_group_settings,
    )
    mobie.metadata.add_view_to_dataset(ds_folder, view_name, grid_view)


# NOTE: we use the stitched view ('add_stitched_view') instead
def main():
    ds_name = "embryo3"
    clims = compute_clims(ds_name)
    add_grid_view(ds_name, clims)
    add_grid_view(ds_name, clims, n_positions=2, view_name="small-grid")


if __name__ == "__main__":
    main()
