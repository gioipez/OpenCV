import cv2
import os
import numpy as np
from collections import Counter
from utils.ManageImage import save_result, show_image
from utils.opencvLogger import logger
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Segment a flower image and analyze its colors.')
parser.add_argument('image_name', type=str, help='The name of the image file to process (e.g., flower_DSC09124.jpg)')
args = parser.parse_args()

# Directories
root_directory = '/Users/giovannilopez/Downloads/2024-08-15_Cultivos/'
calibrated_image_path = 'calibrated/'
sam_mask_directory = 'segmented_images_SAM2/'
is_mask_directory = 'segmented_images_intelligent_scissors/'

image_name = args.image_name  # Get the image name from command line arguments

image_path = os.path.join(root_directory, calibrated_image_path, image_name)
mask_image_path = os.path.join(root_directory, sam_mask_directory, image_name)

# Load the image in BGR
image = cv2.imread(image_path)
image_mask = cv2.imread(mask_image_path)

gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
_, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

# Use the mask to segment the flower from the background
flower_segmented = cv2.bitwise_and(image, image, mask=binary_mask)

# Check if the image was loaded successfully
if flower_segmented is None:
    logger.error(f"Error: Unable to load image at {image_path}")
else:
    image_hsv = cv2.cvtColor(flower_segmented, cv2.COLOR_BGR2HSV)

    # Apply MeanShift algorithm for smoothing
    spatial_radius = 30  # Adjust spatial radius to better capture details in the flower
    color_radius = 25    # Keep color radius relatively small to differentiate flower from black background
    max_level = 1        # Use only 1 level of pyramid (you can experiment with 0 or higher)

    # Perform MeanShift filtering
    mean_shift_result = cv2.pyrMeanShiftFiltering(image_hsv, spatial_radius, color_radius, max_level)

    # Convert the result back to RGB for logging and visualization
    result_image = cv2.cvtColor(mean_shift_result, cv2.COLOR_HSV2BGR)

    save_result(image, f'flower_segmented_source_{image_name}')
    save_result(result_image, f'flower_segmented_result_{image_name}')

    # Now, extract colors and log them based on their frequency
    # Reshape the image to be a list of pixels
    pixels = result_image.reshape(-1, 3)  # Reshape to Nx3 where N is the number of pixels

    # Convert the pixels array into a list of tuples (each tuple is a color in RGB)
    pixels_list = [tuple(pixel) for pixel in pixels]

    # Filter out black or near-black colors (you can adjust the threshold)
    black_threshold = 10  # You can increase this value if needed
    filtered_pixels = [pixel for pixel in pixels_list if not (pixel[0] < black_threshold and pixel[1] < black_threshold and pixel[2] < black_threshold)]

    # Count the frequency of each color (excluding black or near-black)
    color_counter = Counter(filtered_pixels)

    # Sort colors by frequency in descending order
    sorted_colors = color_counter.most_common()

    # Log the most common colors (now excluding black)
    logger.info(f"spatial_radius: {spatial_radius}. color_radius: {color_radius}")
    logger.info("Most common colors found in the segmented image (excluding black):")
    for color, count in sorted_colors[:10]:  # Log top 10 most frequent colors
        # result_image is stored in BGR, that's why we print in RGB, to directly understand
        logger.info(f"Color (RBG): {color[2]},{color[1]},{color[0]}, Count: {count}")
