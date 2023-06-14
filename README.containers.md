# ml4paleo: Container Deploy

The `ml4paleo` web application comprises four jobs total, each of which is given its own container in a docker-compose setup:

-   `webapp`: The web application itself.
-   `conversionrunner`: A job runner that handles image conversion jobs.
-   `segmentrunner`: A job runner that handles segmentation jobs.
-   `meshrunner`: A job runner that handles mesh generation jobs.

For more information on these jobs, see the [webapp README](../webapp/README.md).

To minimize code duplication, we have one `ml4paleo` image, which has the `ml4paleo` library installed and the `webapp/` directory copied into the `/app` directory. This image is used for all four jobs. Because all jobs differ only in their entrypoint, we can use the same image for all four jobs.

To run the containers with Docker-Compose, run `docker-compose up` from the root of the repository. This will build the `Dockerfile` and start the four services. By default, the web service runs on an exposed port 5000.
