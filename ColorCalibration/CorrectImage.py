import argparse
import sys

import cv2

from ColorCalibration.ColorCalibration import ColorCorrection
from utils.opencvLogger import logger

# Create argument parser
parser = argparse.ArgumentParser(description="Color correction using ColorChecker")
parser.add_argument('color_checker_path', type=str, help="Path to the color checker image")
parser.add_argument('target_image_path', type=str, help="Path to the target image to be corrected")
parser.add_argument('--output_file_name', type=str, default=None, help="Output file name for corrected image")
parser.add_argument('--ccm_type', type=int, default=cv2.ccm.CCM_3x3, help="Type of color correction matrix. CCM_3x3=0 or CCM_4x3=1")
parser.add_argument('--gamma', type=float, default=2.2, help="Gamma value for linearization")
parser.add_argument('--degree', type=int, default=3, help="Degree value for linearization")
parser.add_argument('--distance', type=int, default=cv2.ccm.DISTANCE_RGB, help="Distance type for CCM")
parser.add_argument('--saturated_threshold_min', type=float, default=0, help="Minimum saturated threshold")
parser.add_argument('--saturated_threshold_max', type=float, default=0.98, help="Maximum saturated threshold")
parser.add_argument('--linearization', type=int, default=cv2.ccm.LINEARIZATION_GAMMA, help="Linearization type")

args = parser.parse_args()

# Retrieve paths and filenames
color_checker_path = args.color_checker_path
target_image_path = args.target_image_path
output_file_name = args.output_file_name or f"{target_image_path.split('.')[0]}_corrected.jpg"
ccm_type = args.ccm_type
gamma = args.gamma
degree = args.degree
distance = args.distance
saturated_threshold_min = args.saturated_threshold_min
saturated_threshold_max = args.saturated_threshold_max
linearization = args.linearization

logger.debug(f"Color checker: {color_checker_path}")
logger.debug(f"Image to be corrected: {target_image_path}")
logger.debug(f"Output file name: {output_file_name}")

if "NEF" in output_file_name or "ARW" in output_file_name:
    logger.error("Output file name can not contain NEF or ARW, OpenCV doesn't support it")
    sys.exit(1)

# Create ColorChecker
color_correction = ColorCorrection(color_checker_path, target_image_path)

# Detect any Color Chart in ColorChecker
checkers = color_correction.detect_color_checker()
if checkers is None:
    logger.error("ColorChecker chart not detected.")
    sys.exit(1)

# Iterate all the found Color Charts
for checker in checkers:
    # draw box on Color chart
    color_checker_rgb = color_correction.draw_color_checker(checker)
    # Get Charts from Color Charts in form of |p_size|average|stddev|max|min|
    src = color_correction.get_charts_rgb(checker)
    model = color_correction.create_color_correction_model(src, ccm_type=ccm_type, gamma=gamma, distance=distance, saturated_threshold_min=saturated_threshold_min, saturated_threshold_max=saturated_threshold_max, linearization=linearization)

    ccm = model.getCCM()
    loss = model.getLoss()
    logger.debug(f'ccm: {ccm}')
    logger.info(f'{output_file_name} loss: {loss}')

    # Correct colors in the target image
    out_img = color_correction.apply_color_correction(model)
    saveSuccess = color_correction.save_result(
        out_img=out_img,
        output_filename=output_file_name
        )
    if not saveSuccess:
        logger.error(f"Error saving the result {saveSuccess}")
        sys.exit(1)
    logger.debug(f"{target_image_path} calibrated. Result {output_file_name}")
