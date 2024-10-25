import cv2
import numpy as np
from collections import Counter
from utils.ManageImage import save_result
from utils.opencvLogger import logger

# Load the image
image_path = '/Users/giovannilopez/Downloads/2024-08-15_Cultivos/segmented_images_SAM2/flower_DSC_4561.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    logger.error(f"Error: Unable to load image at {image_path}")
else:
    if cv2.cvtColor(image, cv2.COLOR_BGR2HSV) is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv = image

    # Apply MeanShift algorithm for smoothing
    spatial_radius = 30  # Adjust spatial radius to better capture details in the flower
    color_radius = 25    # Keep color radius relatively small to differentiate flower from black background
    max_level = 1        # Use only 1 level of pyramid (you can experiment with 0 or higher)

    # Perform MeanShift filtering
    mean_shift_result = cv2.pyrMeanShiftFiltering(image_hsv, spatial_radius, color_radius, max_level)

    # Convert the result back to RGB for logging and visualization
    result_image = cv2.cvtColor(mean_shift_result, cv2.COLOR_HSV2RGB)

    save_result(cv2.cvtColor(image, cv2.COLOR_HSV2RGB), 'flower_segmented_source_DSC_4561.jpg')
    save_result(result_image, 'flower_segmented_result_DSC_4561.jpg')

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
    logger.info("Most common colors found in the segmented image (excluding black):")
    for color, count in sorted_colors[:10]:  # Log top 20 most frequent colors
        logger.info(f"Color (RGB): {color[0]},{color[1]},{color[2]}, Count: {count}")
