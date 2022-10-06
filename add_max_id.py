from glob import glob

import zarr
from tqdm import tqdm


def add_max_id(path):
    with zarr.open(path, "a") as f:
        for name in ("nuclei", "cells"):
            ds = f[f"labels/{name}/0"]
            data = ds[:]
            max_id = int(data.max())
            ds.attrs["maxId"] = max_id


def main():
    paths = glob("/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff/embryo3/*.ome.zarr")
    for path in tqdm(paths):
        add_max_id(path)


if __name__ == "__main__":
    main()
