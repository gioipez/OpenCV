import cv2
import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from MeanShift.MeanShift import apply_mask

def load_image(image_path):
    """Load image, supporting both RAW and regular image formats."""
    ext = Path(image_path).suffix.lower()
    
    try:
        # Handle RAW images
        if ext in ['.nef', '.arw', '.cr2', '.dng']:  # add other RAW formats if needed
            with rawpy.imread(image_path) as raw:
                rgb_image = raw.postprocess()  # Process RAW image to get RGB data
            return rgb_image
        else:
            # Handle other image formats using OpenCV
            image = cv2.imread(image_path)  # Loads image in BGR format
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    except rawpy._rawpy.LibRawIOError as e:
        print(f"Failed to load RAW image: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def compute_histograms(image):
    """Compute histograms for each color channel (R, G, B)."""
    channels = cv2.split(image)
    colors = ('r', 'g', 'b')  # Since the image is now in RGB, order is (R, G, B)
    histograms = {}
    
    # Compute histogram for each channel
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        # Remove black from Histogram
        hist[0] = 0
        histograms[color] = hist
    return histograms

def show_image_and_histograms(image, histograms, space='RGB'):
    """Show image and its histograms in a single figure."""
    plt.figure(figsize=(12, 6))
    
    # Show the image in the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(image)  # No need for conversion here, the image is already in RGB
    plt.title(f'Image {space}')
    plt.axis('off')  # Turn off axis labels for the image
    
    # Show the histograms in the second subplot
    plt.subplot(1, 2, 2)
    for color, hist in histograms.items():
        plt.plot(hist, color=color)  # Red for 'r', green for 'g', blue for 'b'
        plt.xlim([0, 256])
    plt.title('Color Histograms')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    
    # Adjust layout for better presentation
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # image_path = 'ImageProcessing/images/flower_DSC09037.ARW'
    # image_path = 'ImageProcessing/images/flower_DSC09037_4x3_jpg.jpg'
    # image_path = 'ImageProcessing/images/flower_DSC09037_3x3_nef.jpg'
    # image_path = 'ImageProcessing/images/flower_DSC09037.JPG'
    # image_path = 'ImageProcessing/images/flower_DSC09037_3x3_jpg.jpg'
    image_path = '/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated_all_flowers_cie/flower_DSC09028_JPG.jpg'
    # image_path = '/Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated_all_flowers_rgb/flower_DSC09070.jpg'
    mask_path = '/Users/giovannilopez/Downloads/2024-08-15_Cultivos/segmented_images_intelligent_scissors/segmented_flower_DSC09028_JPG.jpg'

    
    # Load the image (supports both RAW and standard image formats)
    image = load_image(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
    else:
        segmented_image = apply_mask(image, mask_path)
        # Compute histograms for each color channel
        histograms = compute_histograms(segmented_image)
        
        # Display the image and its histograms
        
        max_values = (histograms['r'].tolist().index(histograms['r'].max())+1,  histograms['g'].tolist().index(histograms['g'].max())+1, histograms['b'].tolist().index(histograms['b'].max())+1)
        show_image_and_histograms(segmented_image, histograms, f"CIE - RGB{max_values}")
