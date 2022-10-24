import os
from copy import deepcopy

import mobie
import pandas as pd
import numpy as np
import zarr
from sklearn.cluster import KMeans

from tqdm import tqdm


def compute_gene_profiles(data_path, gene_table_path, table_folder):
    out_table_path = os.path.join(table_folder, "gene_table.tsv")
    print(out_table_path)
    if os.path.exists(out_table_path):
        return pd.read_csv(out_table_path, sep="\t")

    with zarr.open(data_path, "r") as f:
        ds = f["labels/cells"]
        mscales = ds.attrs["multiscales"][0]
        resolution = mscales["datasets"][0]["coordinateTransformations"][0]["scale"]
        labels = ds["0"][:]
    gene_table = pd.read_csv(gene_table_path, sep="\t")

    label_ids = np.unique(labels)[1:]
    gene_ids = np.unique(gene_table["geneID"])
    gene_profiles = {label_id: [] for label_id in label_ids}

    for _, row in tqdm(gene_table.iterrows(), total=len(gene_table)):
        z, y, x = row.z, row.y, row.x
        z, y, x = int(z / resolution[0]), int(y / resolution[1]), int(x / resolution[2])
        label_id = labels[z, y, x]
        if label_id == 0:
            continue
        gene_profiles[label_id].append(row.geneID)

    gene_profiles = {label_id: np.unique(v, return_counts=True) for label_id, v in gene_profiles.items()}
    gene_profiles = {
        label_id: {gene_id: count for gene_id, count in zip(*v)}
        for label_id, v in gene_profiles.items()
    }
    out_table = {"label_id": label_ids}
    for gene_id in gene_ids:
        out_table[gene_id] = [gene_profiles[label_id].get(gene_id, 0) for label_id in label_ids]

    out_table = pd.DataFrame.from_dict(out_table)
    out_table.to_csv(out_table_path, sep="\t", index=False)
    return out_table


def cluster_genes(gene_profiles, table_folder):
    table_path = os.path.join(table_folder, "gene_table.tsv")
    table = pd.read_csv(table_path, sep="\t")
    col_name = "gene_cluster"
    if col_name in table:
        return os.path.basename(table_path)
    data = table.drop(columns="label_id").values
    # perform simple k-means clustering
    clustering = KMeans(n_clusters=8)
    clustering.fit(data)
    cluster_ids = clustering.predict(data)
    table[col_name] = cluster_ids
    table.to_csv(table_path, sep="\t", index=False)
    return os.path.basename(table_path)


def compute_gene_clustering(label_path, gene_table, cell_table_folder):
    gene_profiles = compute_gene_profiles(label_path, gene_table, cell_table_folder)
    return cluster_genes(gene_profiles, cell_table_folder)


def add_example_view(ds_folder, pos_name, table_name, view_name):
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    view = deepcopy(metadata["views"][pos_name])

    displays = view["sourceDisplays"]
    new_displays = []
    for display in displays:
        display_type, display_data = next(iter(display.items()))
        if display_type == "segmentationDisplay":
            display_data["additionalTables"] = [table_name]
            display_data["visible"] = True
            display_data["showTable"] = True
            display_data["colorByColumn"] = "gene_cluster"
            display = {display_type: display_data}
        new_displays.append(display)

    view["uiSelectionGroup"] = "bookmarks"
    metadata["views"][view_name] = view

    mobie.metadata.write_dataset_metadata(ds_folder, metadata)


def main():
    root = "/g/kreshuk/data/marioni/shila/mouse-atlas-2020/ngff"
    ds_name = "embryo3"
    ds_folder = os.path.join(root, ds_name)

    pos_name = "MMStack_Pos0"

    label_path = os.path.join(ds_folder, f"{pos_name}.ome.zarr")
    assert os.path.exists(label_path), label_path
    gene_table_path = os.path.join(
        ds_folder, "tables", f"{pos_name}_genes", "default.tsv"
    )
    assert os.path.exists(gene_table_path), gene_table_path

    cell_table_folder = os.path.join(ds_folder, "tables", f"{pos_name}_cells")
    assert os.path.exists(cell_table_folder), cell_table_folder
    table_name = compute_gene_clustering(label_path, gene_table_path, cell_table_folder)
    add_example_view(ds_folder, pos_name, table_name, view_name="gene-clustering-example")


if __name__ == "__main__":
    main()
