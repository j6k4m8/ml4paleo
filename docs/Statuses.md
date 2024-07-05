# Job Statuses

The `Job` schema goes through a lifecycle of job statuses that indicate the current state of the job. The following statuses are available:

```python
class JobStatus(Enum):
    # Unused.
    PENDING = "pending"

    # When a user starts a job, it is created in the Uploading state.
    UPLOADING = "uploading"
    # Once the web client registers the 100% mark on the upload, we call the
    # server to mark the job as "uploaded".
    UPLOADED = "uploaded"

    # The server will periodically check for UPLOADED jobs and start the
    # conversion process by switching the status to CONVERTING.
    CONVERTING = "converting"
    # Once the conversion is complete (right now all handled by the same job),
    # the status will be set to CONVERTED.
    CONVERTED = "converted"
    # Unused.
    CONVERT_ERROR = "convert_error"

    # If a job has been annotated AT ALL and there are ANY training examples,
    # the job will be put in the ANNOTATED state. This doesn't mean there is
    # a complete annotation, just that there is at least one training example.
    # This means that the job can be trained, and we can show that option to
    # the user, in case they're low-patience. (But it also means that it's
    # pretty likely that the model will be retrained a few times with an
    # increasing number of training examples...)
    ANNOTATED = "annotated"

    # Once a user manually kicks off the training process, the job will be
    # queued for training. This is the state it will be in until the training
    # process picks it up and converts it to TRAINING.
    TRAINING_QUEUED = "training_queued"
    # The job is currently being trained.
    TRAINING = "training"
    # The job has been trained.
    TRAINED = "trained"

    SEGMENTING = "segmenting"
    SEGMENTED = "segmented"
    SEGMENT_ERROR = "segment_error"

    # Any job that is "SEGMENTED" can be manually put into the MESHING_QUEUED
    # state. This is the state it will be in until the meshing process picks
    # it up and converts it to MESHING.
    MESHING_QUEUED = "meshing_queued"
    MESHING = "meshing"
    MESHED = "meshed"
    MESH_ERROR = "mesh_error"

    DONE = "done"
    ERROR = "error"
```
