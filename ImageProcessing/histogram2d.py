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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ColorCalibration.ColorCalibration import logger
from matplotlib import pyplot as plt

if len(sys.argv) < 2:
    logger.error("Usage: python ImageProcessing/histogram2d.py <image_path>")
    sys.exit(1)

# img = cv2.imread('ColorCalibration/calibrated_images/corrected_output.jpg')
img = cv2.imread(sys.argv[1])
if img is None:
    logger.error("File not found")
    sys.exit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# calcHist Calculates a histogram of a set of arrays and it receive
# the following parameters:
#   * [Image]: A list of 3 dimension array.
#   * channels: Which channels from the given array will be used, in a
#       hsv image, if we want to use Hue and Saturation, we should pass
#       [0, 1]
#   * mask:
#   * histSize:
# images, channels, mask, histSize, ranges[, hist[, accumulate]]

mask = np.ones(img.shape[:2], dtype="uint8") * 255

hist = cv2.calcHist([hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])

plt.imshow(hist, interpolation='nearest')
plt.show()
