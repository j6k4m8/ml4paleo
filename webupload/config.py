class CONFIG:
    upload_directory = "volume/uploads"
    chunked_directory = "volume/chunks"

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
    chunk_size = (300, 300, 300)

    download_cache = "volume/download_cache"
