


# Getting mask from a Web Service

## Setup

Install redis in the server and configure in the file `SegmentAnythingAsService.py`


# Tools:

## Write JSON to Redis

Load the JSON file with the boxes and image name to Redis

```shell
python -m utils.WirteJsonToRedis
```

## ManageImage


### OpenCV Logger


## Raw Image Reader


## SegmentAnything as service


Sample CURL:
```shell
curl --location 'http://127.0.0.1:5000/process' --header 'Content-Type: application/json' --data '{ "flower_name": "flower_DSC09017_JPG.jpg", "boxes": ""}'
```

> NOTE: `CALIBRATED_IMAGE_DIR` should be specify to find the images

## Select Points