import cv2
from PIL import Image
import numpy as np
from utils.opencvLogger import logger

def save_result(out_img, output_filename) -> bool:
    return cv2.imwrite(output_filename, out_img)

def read_image(image_path, color_map="RGB"):
    image = Image.open(image_path)
    image = np.array(image.convert(color_map))
    return image

def show_image(image, title="Image"):
    """
    Display an image using OpenCV.
    :param image: The image to be displayed.
    :param title: The title of the window where the image will be displayed.
    """
    if image is None:
        logger.error(f"Error: Unable to load image.")
        return
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
