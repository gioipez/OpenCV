import sys

import cv2
import numpy as np

from utils.opencvLogger import logger
from utils.rawImageReader import rawImage


class ColorCorrection:
    """_summary_

    When doing color correction a color chart pattern is needed to
    use as a guide to calibrate the input image and get a calibrated
    image as output with the correct colors.

    Args:
        color_checker_path (_str_): image path where the color chart is located
        target_image_path (_str_): image to be calibrated

    Returns:
        _CCheckerDetector_: when object is created returns an object ready to
                            look if there is any color checker in the reference
                            picture.
    """
    def __init__(self, color_checker_path, target_image_path) -> cv2.mcc.CCheckerDetector:
        isNEFinImageCC = ".NEF" in color_checker_path
        isARWinImageCC = ".ARW" in color_checker_path
        isNEFinImageTI = ".NEF" in target_image_path
        isARWinImageTI = ".ARW" in target_image_path
        if ((isNEFinImageCC or isARWinImageCC) and (isNEFinImageTI or isARWinImageTI)):
            logger.debug("Both images are raw images")
            ccImage = rawImage(color_checker_path)
            tImage = rawImage(target_image_path)
            self.color_checker_img = ccImage.read_raw_image_bgr()
            self.target_image = tImage.read_raw_image_bgr()
        elif ((isNEFinImageCC and not isNEFinImageTI) or (not isNEFinImageCC and isNEFinImageTI) or (isARWinImageCC and not isARWinImageTI) or (not isARWinImageCC and isARWinImageTI)):
            logger.error("One of the images is not a raw image")
            sys.exit(1)
        else:
            logger.debug("Image aren't raw, found JPG, BIMP, or other")
            # BGR order
            self.color_checker_img = cv2.imread(color_checker_path)
            self.target_image = cv2.imread(target_image_path)

        if self.color_checker_img is None or self.target_image is None:
            logger.error(
                f"""Image not found.
                Color checker: {self.color_checker_img}.
                Target Image: {self.target_image}
                """)
            sys.exit(1)
        self.detector = cv2.mcc.CCheckerDetector.create()

    def detect_color_checker(self):
        """
        Detect color checker in the reference image.
        """
        if not self.detector.process(image=self.color_checker_img, chartType=0):
            logger.error("ColorChecker chart not detected.")
            return None
        return self.detector.getListColorChecker()

    def draw_color_checker(self, checker):
        """
        Draw the detected color checker on the image.
        """
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        cdraw.draw(self.color_checker_img)
        return cv2.cvtColor(self.color_checker_img, cv2.COLOR_BGR2RGB)

    def get_charts_rgb(self, checker):
        """
        Extract the RGB values of the detected color checker.
        """
        chartsRGB = checker.getChartsRGB()
        src = chartsRGB[:, 1].copy().reshape(int(len(chartsRGB[:])/3), 1, 3)
        src /= 255
        return src

    def create_color_correction_model(self, src, color_checker=cv2.ccm.COLORCHECKER_Macbeth, color_space=cv2.ccm.COLOR_SPACE_sRGB, ccm_type=cv2.ccm.CCM_4x3, gamma=2.2, degree=4, distance=cv2.ccm.DISTANCE_RGB, linearization=cv2.ccm.LINEARIZATION_GAMMA, saturated_threshold_min=0, saturated_threshold_max=0.98) -> cv2.ccm_ColorCorrectionModel:
        """
        Create and configure the Color Correction Model (CCM).
        """
        model = cv2.ccm_ColorCorrectionModel(src, color_checker)
        model.setColorSpace(color_space)
        model.setCCM_TYPE(ccm_type)
        model.setDistance(distance)
        model.setLinear(linearization)
        if linearization == cv2.ccm.LINEARIZATION_GAMMA:
            model.setLinearGamma(gamma)
        else:
            model.setLinearDegree(degree)
        model.setSaturatedThreshold(saturated_threshold_min, saturated_threshold_max)
        model.run()
        return model

    def apply_color_correction(self, model):
        """
        Apply the color correction model to the target image.
        """
        to_be_calibrated_img = cv2.cvtColor(
            self.target_image, cv2.COLOR_BGR2RGB
            )
        to_be_calibrated_img = to_be_calibrated_img.astype(np.float64) / 255
        calibrated_image = model.infer(to_be_calibrated_img)
        out_img = np.clip(calibrated_image * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    def save_and_display_results(self, out_img, output_filename='output.jpg'):
        """
        Save and display the results of the color correction.
        :param out_img: The corrected image to be saved and displayed.
        :param output_filename: The name of the output file where the
                                corrected image will be saved.
        """
        cv2.imwrite(output_filename, out_img)
        cv2.imshow("Original Image", self.target_image)
        cv2.imshow("ColorChecker Image", self.color_checker_img)
        cv2.imshow("Corrected Image", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_result(self, out_img, output_filename) -> bool:
        return cv2.imwrite(output_filename, out_img)
