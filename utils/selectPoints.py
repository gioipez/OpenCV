import cv2
import os
from utils.opencvLogger import logger


def select_boxes_from_image(image_path):
    """
    Opens an image and allows the user to select rectangular boxes using the mouse.
    The boxes are returned as a list of tuples containing the top-left and bottom-right coordinates.

    :param image_path: Path to the image file.
    :return: A tuple with the image name and a list of selected boxes.
    """
    boxes = []
    start_point = None
    drawing = False

    def select_box(event, x, y, flags, param):
        nonlocal start_point, drawing

        # Start drawing the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)

        # Update the rectangle while dragging the mouse
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = img.copy()
                cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Image', img_copy)

        # Finalize the rectangle on mouse button up
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            boxes.append((start_point, end_point))
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow('Image', img)

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        logger.error(f"Error: Could not load the image {image_path}. Please check the file path.")
        return None, []
    else:
        # Display the image and set up the mouse callback function
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', select_box)

        logger.debug("Draw boxes on the image to select objects. Press '0' to finish.")

        # Keep the window open until the '0' key is pressed
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):
                break

        # Close the window
        cv2.destroyAllWindows()

        # Extract the image name from the path
        image_name = os.path.basename(image_path)

        return image_name, boxes


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
        result_image_name, selected_boxes = select_boxes_from_image(image_path)

        # Print the result in the desired format
        logger.debug(f'{result_image_name}, {selected_boxes}')
        # What does selected_boxes means?:
        # * First Point (x1, y1): This is the top-left corner of the rectangle.
        # * Second Point (x2, y2): This is the bottom-right corner of the rectangle.
        print({"flower_name": result_image_name, "boxes": selected_boxes})


if __name__ == "__main__":
    main()
