import sys
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


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

    def create_color_correction_model(self, src) -> cv2.ccm_ColorCorrectionModel:
        """
        Create and configure the Color Correction Model (CCM).
        """
        model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
        model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        model.setLinearGamma(2.2)
        model.setLinearDegree(3)
        model.setSaturatedThreshold(0, 0.98)
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


if __name__ == "__main__":
    color_checker_path = 'ColorCalibration/sample_images/cc_sample_2.jpeg'
    target_image_path = 'ColorCalibration/sample_images/flower_sample_2.jpeg'
    output_file_name = 'ColorCalibration/calibrated_images/corrected_output.jpg'

    color_correction = ColorCorrection(color_checker_path, target_image_path)

    checkers = color_correction.detect_color_checker()
    if checkers is None:
        logger.error("ColorChecker chart not detected.")
        sys.exit(1)

    for checker in checkers:
        color_checker_rgb = color_correction.draw_color_checker(checker)
        src = color_correction.get_charts_rgb(checker)
        model = color_correction.create_color_correction_model(src)

        ccm = model.getCCM()
        loss = model.getLoss()
        logger.info(f'ccm: {ccm}')
        logger.info(f'loss: {loss}')

        out_img = color_correction.apply_color_correction(model)
        color_correction.save_and_display_results(out_img, output_file_name)
