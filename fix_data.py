from glob import glob

import zarr
from tqdm import tqdm


def fix_data(path):
    data_list = []
    chunk_list = []
    with zarr.open(path, "a") as f:
        for ds_name in range(1, 5):
            ds_name = str(ds_name)
            data = f[ds_name][:].astype("uint16")
            chunks = f[ds_name].chunks
            del f[ds_name]
            data_list.append(data)
            chunk_list.append(chunks)

    store = zarr.DirectoryStore(path, dimension_separator="/")
    with zarr.open(store, mode="a") as f:
        for ii, ds_name in enumerate(range(1, 5)):
            ds_name = str(ds_name)
            data = data_list[ii]
            chunks = chunk_list[ii]
            f.create_dataset(ds_name, data=data, chunks=chunks)


def main():
    paths = glob("/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff/embryo3/*.ome.zarr")
    for path in tqdm(paths):
        fix_data(path)


if __name__ == "__main__":
    main()
