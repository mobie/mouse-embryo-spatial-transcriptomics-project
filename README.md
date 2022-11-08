# Spatial Transcriptomics Mouse Embryo Data

This project shows data from the publication [Integration of spatial and single-cell transcriptomic data elucidates mouse organogenesis](https://www.nature.com/articles/s41587-021-01006-2).
This data contains imaging and spatial transcriptomics (seqFISH) data of a mouse embryo.

The datas is also available on IDR: TODO
And the image data is streamed from the s3 EBI-Embassy cloud, which mirrors the data from IDR.

The table data (decoded gene detections and segmentation tables) are currently stord on github, as soon as this data is available on IDR we will also stream it from there.

**Project creation:**

The project was created from a local version of the data on s3 by running the following scripts:
- `create_project.py`
- `add_stitched_view.py`
- `add_idr_links.py`
