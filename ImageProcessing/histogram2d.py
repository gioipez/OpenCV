"""
This script calculates the 2D histogram of an image using OpenCV's calcHist
function. It takes an image path as a command-line argument and displays
the histogram plot.

Example usage:
* python ImageProcessing/histogram2d.py ColorCalibration/calibrated_images/corrected_output.jpg
"""

import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from ColorCalibration.ColorCalibration import logger
from MeanShift.MeanShift import apply_mask

def read_image(image_path):
    """Read an image from a given path."""
    # BGR order
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"File not found: {image_path}")
        sys.exit(1)
    return img

def calculate_2d_histogram(image):
    """Calculate the 2D histogram for the Hue and Saturation channels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255  # Full mask

    # Calculate the histogram
    hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    return hist

def plot_histogram(hist, cmap='viridis'):
    """Display the histogram with a specified colormap."""
    plt.imshow(hist, interpolation='nearest', cmap=cmap, vmin=0, vmax=np.max(hist)/2000)
    plt.title('2D Histogram (Hue and Saturation)')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.colorbar(label='Pixel Count')
    plt.show()

def main(image_path, cmap='viridis', mask_path=None):
    """Main function to process the image and display the histogram."""
    img = read_image(image_path)
    segmented_image = apply_mask(img, mask_path)
    hist = calculate_2d_histogram(segmented_image)
    plot_histogram(hist, cmap)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python ImageProcessing/histogram2d.py <image_path> [<mask_path>] [<cmap>]")
        sys.exit(1)

    cmap = sys.argv[3] if len(sys.argv) > 3 else 'viridis'  # Default colormap
    mask_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(image_path=sys.argv[1], cmap=cmap, mask_path=mask_path)

"""
Sample usage
    python -m  ImageProcessing.histogram2d /Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated_all_flowers_cie/flower_DSC09172_JPG.jpg /Users/giovannilopez/Downloads/2024-08-15_Cultivos/segmented_images_intelligent_scissors/segmented_flower_DSC09172_JPG.jpg Grays
"""