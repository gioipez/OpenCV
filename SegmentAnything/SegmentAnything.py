"""
This was based on example https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb
"""
import os
import cv2
import json
import redis

from utils.ManageImage import save_result
from utils.opencvLogger import logger

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


logger.debug(f"PyTorch version: {torch.__version__}")
logger.debug(f"CUDA is available: {torch.cuda.is_available()}")

#### Setup
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    logger.debug(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

## SAM2 Model
sam2_checkpoint = "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def get_image_masks(image, box_coords):
    predictor.set_image(image)

    try:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_coords[None, :],
            multimask_output=False,
        )
    except Exception as e:
        logger.error(f"Error predicting masks: {e}")

    return masks, scores

def read_image(image_path, color_map="RGB"):
    image = Image.open(image_path)
    image = np.array(image.convert(color_map))
    return image

def main():

    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

    calibrated_image_directory = "/home/gio/OpenCV/SegmentAnything/images/calibrated"
    segmented_image_directory = "/home/gio/OpenCV/SegmentAnything/segmented_images"
    logger.debug(f"Directory with source images: {calibrated_image_directory}")
    logger.debug(f"Directory with segmented images: {calibrated_image_directory}")

    with open('flowers_with_boxes.json', 'r') as file:
        data = json.load(file)

    # count = 0
    for flower in data:
        flower_name = flower['flower_name']
        flower_boxes = flower['boxes']
        logger.debug(f"Flower Name: {flower_name}, Boxes: {flower_boxes}")

        image_path = os.path.join(calibrated_image_directory, flower_name)
        segmented_image_path = os.path.join(segmented_image_directory, flower_name)
        logger.debug(f"Processing image: {flower_name}")

        image = read_image(image_path=image_path)

        boxes_ = np.array(flower["boxes"])
        boxes_coor = boxes_.reshape(boxes_.shape[0], 4)

        masks, scores = get_image_masks(image, boxes_coor)
        
        logger.debug(f"Saving masks and scores for {flower_name}")
        redis_client.hset(f"masks:{flower_name}", mapping={
            "status": "Processed",
            "flower_name": flower_name,
            "maks": len(masks),
            "scores": json.dumps(scores.tolist()),
        })

        finish_mask = np.zeros_like(image)

        mask_for_segmentation = masks.astype(np.uint8)

        for mask in mask_for_segmentation:
            color_mask = cv2.merge([mask, mask, mask])
            finish_mask = finish_mask + color_mask

        finish_mask = finish_mask.astype(np.uint8)

        color_mask = cv2.merge([finish_mask[0], finish_mask[0], finish_mask[0]])

        segmented_black = image * finish_mask

        # segmented_black = np.array(image.convert("RGB"))

        if len(masks.shape) > 3:
            is_save = save_result(segmented_black[0], segmented_image_path)
        else:
            is_save = save_result(segmented_black, segmented_image_path)
        
        logger.info(f"Black background image {flower_name} saved: {is_save}")


if __name__ == "__main__":
    main()