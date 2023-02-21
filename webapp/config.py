"""
This file stores configuration variables that are used throughout the web
application. These variables are specific to the web tool and won't be needed
if you're running your own workflow (e.g., in a Jupyter notebook).

"""


class CONFIG:
    """
    A configuration variable container for the ml4paleo web application.

    """

    # The ml4paleo web application has three job runners: A conversion job
    # that converts uploaded data to zarr; a segmentation job that segments
    # the data; and a meshing job that meshes the segmented data. The job
    # runners check for new jobs every few seconds:
    job_poll_sec = 10

    # Uploads
    #
    # The upload directory is where the uploaded slices are stored.
    upload_directory = "volume/uploads"
    # Uploaded data are converted to the zarr format by the conversion runner,
    # and then the zarr arrays are stored here.
    chunked_directory = "volume/chunks"
    # The size for zarr chunks. 300x300x300 sized chunks means ~20 MB files
    # for u8, and ~40 MB files for u16. Any larger than this, you should
    # make sure you can handle each chunk in RAM in your workflow.
    chunk_size = (300, 300, 300)

    # Training and Annotation
    #
    # The size of the image to render to the user in the annotation app.
    # Note that this should be large enough that you can tell what's going on
    # in the image, but small enough that you can interactively run inference
    # on it (for online annotation).
    annotation_shape = (512, 512)
    # The directory where the training dataset is stored. This directory will
    # be populated with the training images and segmentation masks, with the
    # prefixes specified below. For example, with the default settings, you'll
    # get files like "volume/training/img234327.png" and
    # "volume/training/seg234327.png" that will correspond to the same tile.
    training_directory = "volume/training"
    training_img_prefix = "img"
    training_seg_prefix = "seg"
    # The directory where trained models (i.e., parameters, or weights for DL)
    # are stored. The models are stored alongside a freeform JSON file that can
    # contain arbitrary metadata, per the `ml4paleo.segmentationSegmenter3D`
    # class's preferences. The model will be saved under the Job ID, with the
    # name being `[timestamp].model` and the metadata being `[timestamp].json`.
    model_directory = "volume/models"
    # The number of segment chunk jobs to run in parallel. Note that for some
    # segmenters, this can be dangerous to set too high — i.e., if you're using
    # the same GPU for all of the jobs. Also be wary of, e.g., sklearn models,
    # which have their own parallelism settings.
    segment_job_parallelism = 4
    # How large each chunk should be when segmenting. Note that this is smaller
    # than the storage chunk size, because we want to be able to segment the
    # data in parallel and therefore may need more space in RAM.
    segmentation_chunk_size = (256, 256, 256)
    # The directory where segmented arrays should be stored, as zarrs. The
    # segmentation will be stored with the name "[timestamp].zarr", where the
    # timestamp lines up with the model that was used to generate it.
    segmented_directory = "volume/segmented"

    # Meshing
    #
    # The meshing job runner will take the segmented data and generate meshes
    # from it. This is the directory where meshes should be stored. They will
    # be named after the model timestamp that generated the underlying seg.
    meshed_directory = "volume/meshed"
    # Meshing can be done in relatively large chunks, because it's not run in
    # parallel. This is the size of the chunks to use when meshing. This does
    # not need to be the same as the segmentation chunk size or the storage
    # chunk size.
    meshing_chunk_size = (512, 512, 512)

    # Downloading
    #
    # The download cache is where the application will store files that are
    # requested by the user if they require additional processing (i.e., file
    # format conversion, or zipping). This is to avoid having to reprocess
    # files that are requested multiple times. This is a good place to delete
    # files if you need space. If a download file is damaged or corrupted, it
    # should be deleted from this directory, or it will be repeatedly served to
    # the requesting user.
    download_cache = "volume/download_cache"
