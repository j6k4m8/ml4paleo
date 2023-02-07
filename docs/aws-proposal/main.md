# Proposal for AWS Research Credit

https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research

## Should Include

-   Description of problem to be solved.
-   Proposed Amazon Web Services (AWS) solution (including specific AWS tools, timeline, key milestones).
-   Plan for sharing outcomes (tools, data, and/or resources) created during project.
-   Any potential future use of AWS beyond grant duration by individual research group or broader community.
-   Names of any AWS employees you have been in contact with (this is not a prerequisite for the application).
-   Any AWS Public Data Sets to be used in your research.
-   Keywords to facilitate proposal review.

# Proposal

Proposal for AWS Research Credit for X-Ray Tomography Segmentation Pipeline in Paleontology

Problem:
The current process of X-ray tomography segmentation in paleontology is done manually, which is time-consuming and limiting the advancement of science. There are many relatively small (i.e., 10s of GB) image stacks that need to be segmented and will yield rich new scientific data, such as skeletal morphology, gross anatomy, and ...

Proposed AWS Solution:
To automate the process, we propose building a X-ray tomography segmentation pipeline that consists of three components: an uploader service, a segmenter, and a mesh generator. The user will be able to upload their images through a website by clicking and dragging --- prioritizing user-friendliness for non technical users. The images will be stored in S3 and segmented into small subvolumes using a pre-trained U-Net. The segmentation can be done either on Sagemaker or EC2. The subvolumes will be stitched together either in Lambda or EC2 using a simple deterministic stitching process borrowed from big data neuroscience, and then the pipeline will generate meshes using standard marching-cubes libraries. The final result, the segmented stacks and meshes, will be available for the user to download from S3 through a link, which will be sent after authentication. Simple authentication will be implemented by requiring a secret token for the upload from the splash page and for downloading the results. The tokens will be stored in job metadata in a DynamoDB table, which will be created at the start of the upload.

Plan for Sharing Outcomes:
The tools, data, and resources created during the project will be shared with the research community through open-source repository and relevant conferences and journals.

Potential Future Use:
The X-ray tomography segmentation pipeline will not only benefit the individual research group but also the broader community in the field of paleontology. It has the potential to be used in various other applications in the future.

AWS Employees Contacted: [none]

AWS Public Data Sets Used: [none]

Keywords: X-ray tomography, segmentation, paleontology, S3, Lambda, EC2, Sagemaker, U-Net, DynamoDB, authentication.
