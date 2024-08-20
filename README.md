
# OpenCV

This repo contain code to run OpenCV [core module](https://pypi.org/project/opencv-python/) and [contrib](https://pypi.org/project/opencv-contrib-python/) in Python

# Color Calibration

Two modules are used, [Macbeth Chart module](https://docs.opencv.org/4.x/dd/d19/group__mcc.html) for color checker and [Color Correction Model](https://docs.opencv.org/4.x/de/df4/group__color__correction.html).

## Steps for Color Calibration

### Color checker and model for image calibration

Used module was Macbeth Chart Module, it has several steps:

1. Read the image where the color checker is located
2. Create a detector `cv2.mcc.CCheckerDetector.create()`. [[Documentation](https://docs.opencv.org/4.x/d9/d53/classcv_1_1mcc_1_1CCheckerDetector.html)].
3. Validate if a given image contain a color chart `cv2.mcc.CCheckerDetector.create().process(image=color_checker, chartType=0)`, where `color_checker` is an image readed that could/couldn't contain a color chart as [X-Rite](https://www.xrite.com/categories/calibration-profiling/colorchecker-classic) and is read i.e. `cv2.imread('sample_images/cc_sample_2.jpeg')`.
4. If there is a posibility to have several color checkers in the same picture, `detector.getListColorChecker()` is used, if want to use the best with highest confidences, use `detector.getBestColorChecker()` from the detector (`detector = cv2.mcc.CCheckerDetector.create()`)
5. Get the RGB Charts from the Color Checker `checker.getChartsRGB()`. Note: `checker` one color checker inside a list of color checkers inside a picture or the best color checker got from a picture.
6. Create the Color Correction Model ([CCM](https://docs.opencv.org/4.x/de/df4/group__color__correction.html))
   ```
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
8. Use the mode to calibrate the image
   ```
   model.infer(to_be_calibrated_img)
   ```

## Questions

### What does `cv::mcc::CChecker::getChartsRGB()` return?

`getChartsRGB` Create table charts information in form of `|p_size|average|stddev|max|min|`. [[Github Repository](https://github.com/opencv/opencv_contrib/blob/4.8.0/modules/mcc/src/checker_detector.cpp#L1237)]



# References
* [Color Correction Model](https://docs.opencv.org/4.x/d1/dc1/tutorial_ccm_color_correction_model.html).
* [Converting Color Correction opencv module example from C++ to python](https://stackoverflow.com/questions/66302777/converting-color-correction-opencv-module-example-from-c-to-python).
* [Unable to perform color correction from opencv tutorial](https://forum.opencv.org/t/unable-to-perform-color-correction-from-opencv-tutorial/2141).
