# from IntelligentScissors.scissors.gui import GuiManager
# from IntelligentScissors.scissors.feature_extraction import Scissors
# from PIL import ImageTk 
# import numpy as np
# from tkinter import *
# from utils.ManageImage import read_image

# # image_pil = Image.open('/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated/flower_DSC09100_JPG.jpg')
# image_pil = read_image('/home/lopezge/OpenCV/SegmentAnything/images/calibrated/flower_DSC09100_JPG.jpg')
# w = image_pil.shape[1]
# h = image_pil.shape[0]

# scissors = Scissors(np.asarray(image_pil))

# root = Tk()
# stage = Canvas(root, bg="black", width=w, height=h)
# tk_image = ImageTk.PhotoImage(image_pil.any())
# stage.create_image(0, 0, image=tk_image, anchor=NW)

# manager = GuiManager(stage, scissors)
# stage.bind('<Button-1>', manager.on_click)

# stage.pack(expand=YES, fill=BOTH)
# root.resizable(False, False)
# root.mainloop()



import cv2
import numpy as np
from utils.opencvLogger import logger

# Global variables
drawing = False  # True if the mouse is pressed
points = []      # Store points clicked by the user

def mouse_callback(event, x, y, flags, param):
    global drawing, points

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))

def intelligent_scissors(image):
    global points

    # Create a copy of the image to draw the path
    output_image = image.copy()

    logger.info("Click on the image to define the path. Press 'q' to quit.")

    while True:
        # Draw the points on the output image
        for point in points:
            cv2.circle(output_image, point, 20, (0, 255, 0), -1)

        cv2.imshow('Intelligent Scissors', output_image)

        # Break the loop when the user presses 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("User pressed 'q' to quit.")
            break

    return points

def main():
    # Load an image
    image_path = '/home/lopezge/OpenCV/SegmentAnything/images/calibrated/flower_DSC09100_JPG.jpg'
    # BGR order
    image = cv2.imread(image_path)

    if image is None:
        logger.info("Error loading image")
        return

    # Set the mouse callback function
    cv2.namedWindow('Intelligent Scissors')
    cv2.setMouseCallback('Intelligent Scissors', mouse_callback)

    # Get the user-defined points
    points = intelligent_scissors(image)

    # Convert points to a NumPy array for further processing
    if len(points) > 1:
        points = np.array(points)

        # Create a mask for segmentation
        logger.info(f"Creating mask for segmentation, image shape {image.shape[:2]}")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.polylines(mask, [points], isClosed=True, color=255, thickness=2)

        # Show the mask
        logger.info("Showing mask")
        cv2.imshow('Mask', mask)

        # Create a segmented output image
        logger.info("Creating segmented output image")
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('Segmented Image', segmented_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
