# Tour of the Compose Volume

This project uses a shared filesystem volume as a way to share and coordinate data workloads across the containers.

The volume contains the following files and directories:

## `chunks/`

This directory contains the chunked data files (zarr format) that are used to store the underlying imagery for each job.

The files are stored in `chunks/{job_id}/` where `job_id` is the unique identifier for the job and the directory serves as a zarr array.

## `jobs.json`

The "database" for the web app. This file is a JSON file that contains the metadata for each job. The web app reads and writes to this file to keep track of the jobs.

A metadata blob looks like this:

```json
{
    "CC7651": {
        "status": "JobStatus.MESHED",
        "name": "Burrow",
        "id": "CC7651",
        "created_at": "2024-07-05T16:24:13.446237",
        "last_updated_at": "2024-07-05T16:35:11.017184",
        "current_job_progress": 0.9791666666666666,
        "shape": [100, 100, 100]
    }
}
```

## `meshed/`

This directory contains subdirectories named like this:

```
meshed/{job_id}/{model_id}.zarr/
```

This contains files like this,

```
255.combined.obj  255.combined.stl  _255_0_0_0.stl  _255_0_1024_0.stl  _255_0_512_0.stl  _255_512_0_0.stl  _255_512_1024_0.stl  _255_512_512_0.stl
```

...where the 255 is the voxel value and the underscore-prefixed files are chunks.

## `models/`

This directory contains the trained models. The models are stored in `models/{job_id}/` where `job_id` is the unique identifier for the job, and the models themselves are stored as .json/.model pairs, where the .json file contains metadata about the model and the .model file contains the model itself as a pickle. The JSON looks like this:

```json
{ "rf_kwargs": {}, "model_class": "RandomForest3DSegmenter" }
```

Thus, a model can be reconstructed with hyperparameters from this JSON.

## `segmented/`

This directory contains the segmented data files (zarr format) that are used to store the segmentation results for each job:

```
segmented/{job_id}/{model_id}.zarr/
```

## `training/`

This directory contains paired training images and masks, with files named like this:

```
training/{job_id}/img{timestamp}.png
training/{job_id}/seg{timestamp}.png
```

Because `model_id`s are also timestamps, you can tell which images were used to train which models by comparing the timestamps {`timestamp`} <= {`model_id`}.

## `uploads/`

This directory contains the uploaded raw z-slice images for each job,

```
uploads/{job_id}/{original_filename1}
uploads/{job_id}/{original_filename2}
uploads/{job_id}/{original_filename3}
```

These must sort lexicographically to be in the correct order when converted.
