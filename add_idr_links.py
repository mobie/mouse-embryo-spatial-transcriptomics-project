import os
import mobie
from common import ROOT


def add_idr_links():
    ds_folder = os.path.join(ROOT, "embryo3")
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)

    root_url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0138A/TimEmbryos-120919/HybCycle_29"

    sources = metadata["sources"]
    new_sources = {}
    for name, source in sources.items():
        source_type, source_data = next(iter(source.items()))
        if source_type not in ("regions", "spots"):
            image_data = source_data["imageData"]
            file_name = image_data["ome.zarr"]["relativePath"]
            channel = image_data["ome.zarr"].get("channel")

            url = f"{root_url}/{file_name}"
            image_data["ome.zarr.s3"] = {"s3Address": url}
            if channel is not None:
                image_data["ome.zarr.s3"]["channel"] = channel

            source_data["imageData"] = image_data
            source = {source_type: source_data}

        new_sources[name] = source

    mobie.metadata.write_dataset_metadata(ds_folder, metadata)
    mobie.validation.validate_project(ROOT, require_remote_data=True)


if __name__ == "__main__":
    add_idr_links()
