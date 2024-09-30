# Segment Anything with SAM2

This project uses the Segment Anything Model 2 (SAM2) to segment images of flowers based on predefined bounding boxes. The implementation leverages PyTorch for model inference and Redis for storing the results.

## Requirements

- Python 3.7+
- PyTorch (with support for CUDA or MPS)
- OpenCV
- Redis
- PIL (Pillow)
- NumPy

## Setup

SAM2 can be  directly installed from the [REPO](https://github.com/facebookresearch/segment-anything-2) or by clonning the repo.

1. **Clone the repository**:


   ```bash
   git clone https://github.com/your-repo/segment-anything.git
   cd segment-anything-2 & pip install -e .
   ```

   > Note: More information in the [REPO](https://github.com/facebookresearch/segment-anything-2)

2. **Directly install with pip**

    Follow the direction given in the official repository to install, but it is basically:

    ```shell
    pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'
    ```

    Copy the calibrated image to the remote server

    ```
    rsync -r calibrated gio@192.168.1.16:/home/gio/OpenCV/SegmentAnything/images/
    ```

3. **Install required packages:**

    You can install the remining Python packages using pip:


    ```shell
    pip install torch torchvision torchaudio
    pip install opencv-python redis Pillow numpy
    ```

3. **Download the SAM2 model weights:**

    Ensure you have the SAM2 model weights (`sam2_hiera_large.pt`) and configuration file (`sam2_hiera_l.yaml`) in your project directory. You can find these files in the SAM2 GitHub [repository](https://github.com/facebookresearch/segment-anything-2).

# Configuration

## Environment Variables

If you are using Apple MPS, you may want to fall back to CPU for unsupported operations by setting the following environment variable:


```shell
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Redis

Make sure Redis is running on your local machine. The script uses Redis to store the segmented masks and scores.

# Usage

1. Place your images in the `calibrated_image_directory`, specified in the code:

    ```shell
    calibrated_image_directory = "/path/to/your/calibrated/images"
    segmented_image_directory = "/path/to/save/segmented/images"
    ```

2. Prepare a JSON file (`flowers_with_boxes.json`) with the following structure:

    ```shell
    [
        {
            "flower_name": "flower1.jpg",
            "boxes": [
                [[x1, y1], [x2, y2]],
                ...
            ]
        },
        ...
    ]
    ```

3. Execute Segment Anything:

    ```shell
    python -m SegmentAnything.SegmentAnything
    ```

# Logging

The application uses a logging module to output debug and info messages. Check the console and log files for detailed logs about the processing of images.

# References

* [Repository](https://github.com/facebookresearch/segment-anything-2)
* Segment Anything 2 [Site](https://ai.meta.com/sam2/)
* [Examples](https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb)
