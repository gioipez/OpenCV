import cv2
from PIL import Image
import numpy as np

def save_result(out_img, output_filename) -> bool:
    return cv2.imwrite(output_filename, out_img)

def read_image(image_path, color_map="RGB"):
    image = Image.open(image_path)
    image = np.array(image.convert(color_map))
    return image
