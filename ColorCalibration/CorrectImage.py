import sys
from ColorCalibration.ColorCalibration import ColorCorrection
from utils.opencvLogger import logger

if len(sys.argv) < 3:
    logger.error("Usage: python CorrectImage.py <color_checker_path> <target_image_path> [<output_file_name>]")
    sys.exit(1)

# Retrieve path from arguments
color_checker_path = sys.argv[1]
target_image_path = sys.argv[2]
if len(sys.argv) == 3:
    output_file_name = target_image_path.split('.')[0] + '_corrected.jpg'
elif len(sys.argv) == 4:
    output_file_name = sys.argv[3]
else:
    logger.error("Invalid number of arguments.")
    sys.exit(1)
logger.info(f"Color checker: {color_checker_path}")
logger.info(f"Image to be corrected: {target_image_path}")
logger.info(f"Output file name: {output_file_name}")

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
    model = color_correction.create_color_correction_model(src)

    ccm = model.getCCM()
    loss = model.getLoss()
    logger.info(f'ccm: {ccm}')
    logger.info(f'loss: {loss}')

    # Correct colors in the target image
    out_img = color_correction.apply_color_correction(model)
    color_correction.save_and_display_results(out_img, output_file_name)
