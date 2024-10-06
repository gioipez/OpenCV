#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/line_descriptor.hpp> // For Intelligent Scissors
#include <iostream>

bool hasMap = false;
cv::segmentation::IntelligentScissorsMB tool; // Global tool instance

// Mouse callback function
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    cv::Mat* src = static_cast<cv::Mat*>(userdata); // Retrieve image pointer

    if (event == cv::EVENT_LBUTTONDOWN) {
        std::cout << "Start Coordinates: (" << x << ", " << y << ")" << std::endl;

        if (x < src->cols && y < src->rows) {
            cv::TickMeter tm;
            tm.start();
            tool.buildMap(cv::Point(x, y));
            tm.stop();
            std::cout << "Build Map Time: " << tm.getTimeMilli() << " ms" << std::endl;
            hasMap = true;
        }
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (hasMap && x >= 0 && x < src->cols && y >= 0 && y < src->rows) {
            cv::Mat dst = src->clone();
            cv::Mat contour;
            tool.getContour(cv::Point(x, y), contour);

            std::vector<cv::Mat> contours;
            contours.push_back(contour);
            cv::Scalar color(0, 255, 0, 255); // RGBA
            cv::polylines(dst, contours, false, color, 1, cv::LINE_8);

            cv::imshow("canvasOutput", dst);
            dst.release();
        }
    }
}

int main() {
    // Load the image
    cv::Mat src = cv::imread("/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated/flower_DSC09100_JPG.jpg");
    if (src.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Resize the image (if needed)
    cv::resize(src, src, cv::Size(src.cols/2, src.rows/2));

    cv::imshow("canvasOutput", src);

    // Initialize the Intelligent Scissors tool
    tool.setEdgeFeatureCannyParameters(32, 100);
    tool.setGradientMagnitudeMaxLimit(200);
    tool.applyImage(src);

    // Set mouse callback with userdata (the image)
    cv::setMouseCallback("canvasOutput", mouseCallback, &src);

    // Wait until a key is pressed
    cv::waitKey(0);

    // Clean up
    src.release();
    // tool.delete(); // If necessary, depending on how IntelligentScissorsMB is implemented

    return 0;
}

