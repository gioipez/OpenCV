# Color Calibration

This project implements color calibration using two primary modules: [Macbeth Chart Module](https://docs.opencv.org/4.x/dd/d19/group__mcc.html) for color checker detection and [Color Correction Model](https://docs.opencv.org/4.x/de/df4/group__color__correction.html) for color correction.

## Overview

Color calibration is essential when using color charts to ensure accurate color reproduction in images. This implementation will guide you through the necessary steps to calibrate an image using a reference color checker.

## Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- A compatible camera or image files containing color charts

## Steps for Color Calibration

### Color Checker Detection

Follow these steps to detect and use a Macbeth color checker in your images:

1. **Read the Image**: Load the image containing the color checker using `cv2.imread()`.
2. **Create a Detector**: Instantiate the color checker detector:
   ```python
   detector = cv2.mcc.CCheckerDetector.create()
   ```
3. **Process the Image**: Validate if the image contains a color chart
   ```python
   detector.process(image=color_checker, chartType=0)
   ```
   where `color_checker` is the loaded image.
4. **Detect Multiple Checkers**: If multiple color checkers may exist, use:
   ```python
   detector.getListColorChecker()
   ```
   To select the best one based on confidence:
   ```python
   detector.getBestColorChecker()
   ```
5. **Extract RGB Values**: Get RGB values from the detected color checker:
   ```python
   checker.getChartsRGB()
   ```
6. **Create the Color Correction Model (CCM)**:
   ```python
   model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
   model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
   model.setCCM_TYPE(cv2.ccm.CCM_3x3)
   model.setDistance(cv2.ccm.DISTANCE_CIE2000)
   model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
   model.setLinearGamma(2.2)
   model.setLinearDegree(3)
   model.setSaturatedThreshold(0, 0.98)
   model.run()
   ```
7. **Calibrate the Image**:
   ```python
   model.infer(to_be_calibrated_img)
   ```

### Results

![ColorCalibratedWithCC](https://github.com/user-attachments/assets/8ac9ceb8-155e-457c-91a2-8032ba693c50)

> Note: The color chart used for calibration may not be in the same image as the target flower. Consistent exposure conditions while capturing images will yield better calibration results.

### Common Questions

#### What does cv::mcc::CChecker::getChartsRGB() return?

`getChartsRGB()` generates a table of charts' information, including metrics such as average, standard deviation, maximum, and minimum values. More details can be found in the OpenCV [Github Repository](https://github.com/opencv/opencv_contrib/blob/4.8.0/modules/mcc/src/checker_detector.cpp#L1237).

# Executing the Image Correction Script

   To execute the image correction script, use the following command from the root directory:

   ```python
   python -m ColorCalibration.CorrectImage <color_checker_path> <target_image_path> [<output_file_name>]
   ```

   Example:

   ```python
   python -m ColorCalibration.CorrectImage ~/Downloads/2024-08-15_Cultivos/camara3/phenotype_10_CAJAMARCA/cc_DSC_4559.JPG  ~/Downloads/2024-08-15_Cultivos/camara3/phenotype_10_CAJAMARCA/flower_DSC_4560.JPG --output_file_name /tmp/flower_DSC_4560_JPG4x3.jpg --ccm_type 1 --distance 6
   ```

> NOTE: If the CC is in the same image to be corrected use the same path for <color_checker_path> and <target_image_path>


# References
* [Color Correction Model](https://docs.opencv.org/4.x/d1/dc1/tutorial_ccm_color_correction_model.html).
* [Converting Color Correction opencv module example from C++ to python](https://stackoverflow.com/questions/66302777/converting-color-correction-opencv-module-example-from-c-to-python).
* [Unable to perform color correction from opencv tutorial](https://forum.opencv.org/t/unable-to-perform-color-correction-from-opencv-tutorial/2141).
* [X-Rite](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic)
* [Macbeth Chart module](https://docs.opencv.org/4.x/dd/d19/group__mcc.html)
* [Color Correction Model](https://docs.opencv.org/4.x/de/df4/group__color__correction.html)
* [CCheckerDetector](https://docs.opencv.org/4.x/d9/d53/classcv_1_1mcc_1_1CCheckerDetector.html)
* [CCM](https://docs.opencv.org/4.x/de/df4/group__color__correction.html)