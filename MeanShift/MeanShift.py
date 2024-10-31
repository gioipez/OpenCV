import cv2
import os
import sys
import numpy as np
from collections import Counter
from utils.ManageImage import save_result, show_image
from utils.opencvLogger import logger
import argparse

# Function to load an image
def load_image(image_path):
    return cv2.imread(image_path)

# Function to load and apply mask if available
def apply_mask(image, mask_path=None):
    if mask_path:
        # Load the mask as a grayscale image
        image_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if mask dimensions match the image dimensions
        if image.shape[:2] != image_mask.shape:
            # Resize the mask to double its size
            image_mask = cv2.resize(image_mask, (image_mask.shape[1] * 2, image_mask.shape[0] * 2))
            
            # Check again if resizing made dimensions match
            if image.shape[:2] != image_mask.shape:
                logger.error(f"Mask dimensions {image_mask.shape} do not match image dimensions {image.shape[:2]}")
                raise ValueError(f"Mask dimensions {image_mask.shape} do not match image dimensions {image.shape[:2]}")
        
        # Apply a threshold to ensure it's binary (CV_8U type)
        _, binary_mask = cv2.threshold(image_mask, 0, 255, cv2.THRESH_BINARY)
        
        # Debug information to verify the types and shapes
        logger.info(f"Image shape: {image.shape}, type: {image.dtype}")
        logger.info(f"Mask shape: {binary_mask.shape}, type: {binary_mask.dtype}")
        
        # Apply the binary mask to the image
        return cv2.bitwise_and(image, image, mask=binary_mask)
    return image



# Function to perform MeanShift filtering
def mean_shift_segmentation(image_hsv, spatial_radius=30, color_radius=25, max_level=1):
    return cv2.pyrMeanShiftFiltering(image_hsv, spatial_radius, color_radius, max_level)

# Function to extract and count colors
def analyze_colors(image, black_threshold=10, top_n=5):
    pixels = image.reshape(-1, 3)
    pixels_list = [tuple(pixel) for pixel in pixels]
    filtered_pixels = [p for p in pixels_list if not all(c < black_threshold for c in p)]
    color_counter = Counter(filtered_pixels)
    sorted_colors = color_counter.most_common()
    top_colors = [(tuple(reversed(color)), count, (count / len(filtered_pixels)) * 100) 
                  for color, count in sorted_colors[:top_n]]
    return top_colors, len(filtered_pixels), color_counter, sorted_colors


# Function to create and save HTML report
def save_html_report(result_dir, image_path, source_img_filename, result_img_filename, top_colors, total_pixels, spatial_radius, color_radius, cluster_count, sorted_colors):
    image_name = os.path.basename(image_path)

    html_content = f"""
    <html>
    <head>
        <title>Image Segmentation Result</title>
        <style>
            .color-box {{
                display: inline-block;
                width: 50px;
                height: 20px;
                margin-right: 10px;
                vertical-align: middle;
                border: 1px solid #000;
            }}
            .image-container {{
                display: flex;
                gap: 20px;
            }}
            .image-container img {{
                width: 400px;
            }}
        </style>
    </head>
    <body>
        <h1>Flower Segmentation Result</h1>
        
        <div class="image-container">
            <div>
                <h2>Original Image</h2>
                <img src="{source_img_filename}" alt="Original Image"/>
            </div>
            <div>
                <h2>Segmented Image</h2>
                <img src="{result_img_filename}" alt="Segmented Image"/>
            </div>
        </div>
        
        <p>Total segmented pixels: {total_pixels}</p>
        <p>Total clusters: {cluster_count}</p>
        <p>spatial_radius: {spatial_radius}. color_radius: {color_radius}</p>
        
        <h2>Top 5 Colors in RGB (excluding black)</h2>
        <ul>
    """
    
    # Add top colors in the format displayed in your example
    for color, count, percentage in top_colors:
        rgb_color = f'rgb({color[0]}, {color[1]}, {color[2]})'
        html_content += f"""
        <li>
            <span class="color-box" style="background-color: {rgb_color};"></span>
            Color: {rgb_color}, Count: {count}, Percentage: {percentage:.2f}%
        </li>
        """

    # Add information about the least common color
    least_common_rgb_color = f'rgb({sorted_colors[-1][0][0]}, {sorted_colors[-1][0][1]}, {sorted_colors[-1][0][2]})'

    html_content += f"""
        </ul>
        <p>
        <span class="color-box" style="background-color: {least_common_rgb_color};"></span>
        Cluster with least occurrence {least_common_rgb_color}, Count: {sorted_colors[-1][1]}, Percentage: {(sorted_colors[-1][1]/total_pixels)*100:.5f},
        </p>
    </body>
    </html>
    """

    # Save the HTML file in the result directory
    html_filename = os.path.join(result_dir, f"segmentation_result_{image_name.split('.')[0]}_{spatial_radius}_{color_radius}.html")
    with open(html_filename, 'w') as html_file:
        html_file.write(html_content)

    logger.info(f"HTML file created: {html_filename}")


# Main function for processing
def process_image(image_path, mask_path=None):
    # Create result directory if it doesn't exist
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    # Load and segment image
    image = load_image(image_path)
    if image is None:
        logger.error(f"Error loading image {image_path}")
        return

    segmented_image = apply_mask(image, mask_path)
    image_hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)

    spatial_radius=400
    color_radius=50
    mean_shift_result = mean_shift_segmentation(image_hsv, spatial_radius, color_radius)
    result_image = cv2.cvtColor(mean_shift_result, cv2.COLOR_HSV2BGR)

    # Save results and log colors
    source_img_filename = f'image_source_{os.path.basename(image_path)}'
    result_img_filename = f'flower_segmented_result_{spatial_radius}_{color_radius}_{os.path.basename(image_path)}'
    save_result(image, os.path.join(result_dir, source_img_filename))
    save_result(result_image, os.path.join(result_dir, result_img_filename))
    
    # Analyze colors and get cluster count
    top_colors, total_pixels, color_counter, sorted_colors = analyze_colors(result_image)
    cluster_count = len(color_counter)  # Count all unique colors

    # Generate HTML report
    save_html_report(result_dir, image_path, source_img_filename, result_img_filename, top_colors, total_pixels, spatial_radius, color_radius, cluster_count, sorted_colors)
    

# Set up argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment a flower image and analyze its colors.')
    parser.add_argument('image_path', type=str, help='The full path to the image file to process')
    parser.add_argument('--mask_path', type=str, help='Optional full path to the mask image file')
    args = parser.parse_args()
    
    # Process image with optional mask
    process_image(args.image_path, args.mask_path)
