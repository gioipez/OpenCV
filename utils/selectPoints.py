import cv2
import os
from utils.opencvLogger import logger


def select_points_from_image(image_path):
    """
    Opens an image and allows the user to select points using the mouse.
    The points are returned as a list of (x, y) tuples.

    :param image_path: Path to the image file.
    :return: A tuple with the image name and a list of selected points.
    """
    points = []

    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', img)

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        logger.error(f"Error: Could not load the image {image_path}. Please check the file path.")
        return None, []
    else:
        # Display the image and set up the mouse callback function
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', select_point)

        logger.debug("Click on the image to select points. Press '0' to finish.")

        # Keep the window open until the '0' key is pressed
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):
                break

        # Close the window
        cv2.destroyAllWindows()

        # Extract the image name from the path
        image_name = os.path.basename(image_path)

        return image_name, points


def main():
    # Directory containing images
    calibrated_image_directory = "/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated"

    # Get a list of all image files in the directory (e.g., JPG, PNG)
    image_files = [f for f in os.listdir(calibrated_image_directory)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(calibrated_image_directory, image_file)

        # Call the function and get the result
        result_image_name, selected_points = select_points_from_image(image_path)

        # Print the result in the desired format
        logger.debug(f'{result_image_name}, {selected_points}')
        print({"flower_name": result_image_name, "points": selected_points })


if __name__ == "__main__":
    main()
