"""
This file stores configuration variables that are used throughout the web
application. These variables are specific to the web tool and won't be needed
if you're running your own workflow (e.g., in a Jupyter notebook).

"""


class CONFIG:
    """
    A configuration variable container for the ml4paleo web application.

    """

    #
    job_poll_sec = 10

    # Uploads
    upload_directory = "volume/uploads"
    chunked_directory = "volume/chunks"
    chunk_size = (300, 300, 300)

    # Training and annotation
    annotation_shape = (512, 512)
    training_directory = "volume/training"
    training_img_prefix = "img"
    training_seg_prefix = "seg"
    model_directory = "volume/models"
    # The number of segment chunk jobs to run in parallel:
    segment_job_parallelism = 4
    segmentation_chunk_size = (64, 64, 64)

    segmented_directory = "volume/segmented"

    meshed_directory = "volume/meshed"
    meshing_chunk_size = (512, 512, 512)

    download_cache = "volume/download_cache"
