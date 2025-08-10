I have the example sub-video in s3, the path is like s3:/spoton_ml/video_s14/c09/sub_video_01.mp4

The details is like:
User: Jwizzed
Repo: spoton_ml
Full dagshub url: https://dagshub.com/Jwizzed/spoton_ml
Bucket_name: spoton_ml
Endpoint_url: https://dagshub.com/api/v1/repo-buckets/s3/Jwizzed
Public Key ID and Secret Access Key: 0023115884d84bc9b52fce4451572ccb08d9f85f
region: us-east-1

The folder in the bucket:
video_s14
|-c09
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
|-c12
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
|-c13
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
|-c16
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
video_s37
|-c01
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
|-c02
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
|-c03
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4
|-c05
||-sub_video_01.mp4
||-sub_video_02.mp4
||-sub_video_03.mp4
||-sub_video_04.mp4

You can get these file by:
```
from dagshub import get_repo_bucket_client

boto_client = get_repo_bucket_client("<user>/<repo>", flavor="boto")

# Upload file
boto_client.upload_file(
    Bucket="<repo>",       # name of the repo
    Filename="local.csv",  # local path of file to upload
    Key="remote.csv",      # remote path where to upload the file
)
# Download file
boto_client.download_file(
    Bucket="<repo>",       # name of the repo
    Key="remote.csv",      # remote path from where to download the file
    Filename="local.csv",  # local path where to download the file
)
```
