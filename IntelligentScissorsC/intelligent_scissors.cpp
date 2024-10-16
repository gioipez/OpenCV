#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/line_descriptor.hpp>
#include <iostream>
#include <vector>
#include <filesystem> // For handling file paths

bool hasMap = false;
cv::segmentation::IntelligentScissorsMB tool; // Global tool instance
std::vector<cv::Point> finalContour; // Store the final contour points

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

            // Clear the contour points only on first click
            if (finalContour.empty()) {
                finalContour.push_back(cv::Point(x, y)); // Store initial point
            }
        }
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (hasMap && x >= 0 && x < src->cols && y >= 0 && y < src->rows) {
            cv::Mat dst = src->clone();
            cv::Mat contour;
            tool.getContour(cv::Point(x, y), contour);

            // Draw the current contour
            if (!contour.empty()) {
                std::vector<cv::Mat> contours;
                contours.push_back(contour);
                cv::Scalar color(0, 255, 0); // Green for the current contour
                cv::polylines(dst, contours, false, color, 1, cv::LINE_8);
            }

            // Draw previously stored points
            for (const auto& point : finalContour) {
                cv::circle(dst, point, 3, cv::Scalar(255, 0, 0), -1); // Draw stored points
            }

            cv::imshow("canvasOutput", dst);
            dst.release();
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        if (hasMap) {
            cv::Mat contour;
            tool.getContour(cv::Point(x, y), contour);
            if (!contour.empty()) {
                // Append the new contour points to the final contour
                finalContour.insert(finalContour.end(), contour.begin<cv::Point>(), contour.end<cv::Point>());
            }
        }
    }
}

void showSelectedArea(const cv::Mat& src, const std::vector<cv::Point>& contour) {
    // Create a mask for the selected area
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours = {contour};
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create an output image and apply the mask
    cv::Mat output;
    src.copyTo(output, mask);

    cv::imshow("Selected Area", output);
}

void saveSegmentedImage(const cv::Mat& src, const std::vector<cv::Point>& contour, const std::string& outputDir, const std::string& inputImageName) {
    // Create a mask for the selected area
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours = {contour};
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create an output image and apply the mask
    cv::Mat output;
    src.copyTo(output, mask);

    // Construct the output file path
    std::string outputPath = outputDir + "/segmented_" + inputImageName + ".jpg"; // Use .jpg as the output format

    // Save the segmented image
    cv::imwrite(outputPath, output);
    std::cout << "Segmented image saved to: " << outputPath << std::endl;
}

std::string getBaseName(const std::string& filePath) {
    size_t lastSlash = filePath.find_last_of("/\\");
    size_t lastDot = filePath.find_last_of(".");
    std::string baseName = (lastSlash == std::string::npos) ? filePath : filePath.substr(lastSlash + 1);
    if (lastDot != std::string::npos) {
        baseName = baseName.substr(0, lastDot - lastSlash - 1); // Remove extension
    }
    return baseName;
}


int main(int argc, char** argv) {
    // Check for correct number of arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <output_directory>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    std::string outputDir = argv[2];

    // Load the image
    cv::Mat src = cv::imread(imagePath);
    if (src.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Resize the image (if needed)
    cv::resize(src, src, cv::Size(src.cols / 2, src.rows / 2));

    cv::imshow("canvasOutput", src);

    // Initialize the Intelligent Scissors tool
    tool.setEdgeFeatureCannyParameters(32, 100);
    tool.setGradientMagnitudeMaxLimit(200);
    tool.applyImage(src);

    // Set mouse callback with userdata (the image)
    cv::setMouseCallback("canvasOutput", mouseCallback, &src);

    // Get the base name of the input image
    std::string inputImageName = getBaseName(imagePath);

    while (true) {
        // Wait until a key is pressed
        int key = cv::waitKey(1);
        if (key == 27) { // Esc key
            break; // Exit loop
        } else if (key == ' ') { // Space key
            // Show the selected area based on the final contour
            showSelectedArea(src, finalContour); // Show selected area
        } else if (key == 'c') { // Clear points
            finalContour.clear(); // Clear the contour points
            std::cout << "Cleared points. Start a new selection." << std::endl;
        } else if (key == 's') { // Save segmented image
            if (!finalContour.empty()) {
                saveSegmentedImage(src, finalContour, outputDir, inputImageName); // Save segmented image
            } else {
                std::cout << "No contour points to save." << std::endl;
            }
        }
    }

    // Clean up
    src.release();
    // tool.delete(); // If necessary, depending on how IntelligentScissorsMB is implemented

    return 0;
}
