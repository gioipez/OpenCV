import numpy as np
import argparse
from sklearn.cluster import KMeans
from collections import Counter
from ImageProcessing.histogram import load_image, cv2
from MeanShift.MeanShift import apply_mask
from utils.rhs_color_mapper import rgb_to_hex, find_closest_colors_with_ucl
from utils.opencvLogger import logger

def get_dominant_color(image_path, num_colors=5, mask_path=None):
    # Open the image file
    image = load_image(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return []
    else:
        segmented_image = apply_mask(image, mask_path)
        # Resize image to reduce computation time
        segmented_image_resized = cv2.resize(segmented_image, (100, 100))
        # Flatten image data to a list of RGB tuples
        pixels = np.array(segmented_image_resized).reshape((-1, 3))

        # Filter out black or near-black pixels
        non_black_pixels = [pixel for pixel in pixels if not (pixel < [30, 30, 30]).all()]
        
        if not non_black_pixels:
            logger.error("No significant non-black pixels found.")
            return []
        
        # Use KMeans clustering to find the dominant colors
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(non_black_pixels)
        
        # Get colors and their frequency
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = Counter(labels)

        # Sort colors by frequency
        sorted_colors = sorted(zip(counts.values(), colors), reverse=True)
        
        # Get the dominant color(s)
        dominant_colors = [color for _, color in sorted_colors]
        return dominant_colors


def main():
    parser = argparse.ArgumentParser(description='Get dominant colors of an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--mask_path', type=str, help='Path to the mask file')
    args = parser.parse_args()
    dominant_colors = get_dominant_color(args.image_path, num_colors=5, mask_path=args.mask_path)
    for color in dominant_colors:      
        r, g, b = map(int, color)
        rgb_hex = rgb_to_hex(r, g, b)
        top_5_closest_in_rgb = find_closest_colors_with_ucl(rgb_hex)
        # Display the results
        logger.info("Top 5 closest colors for RGB ({r},{g},{b}). HEX: {rgb_hex}:")
        for color in top_5_closest_in_rgb:
            label, rgb, distance, ucl_name = color
            logger.info(f"\t\tLabel: {label}, RGB: {rgb}, Distance: {distance:.2f}, UCL Name: {ucl_name}")

if __name__ == "__main__":
    main()

"""
python -m ImageProcessing.Kmeans /Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated_all_flowers_cie/flower_DSC09028_JPG.jpg --mask_path /Users/giovannilopez/Downloads/2024-08-15_Cultivos/segmented_images_intelligent_scissors/segmented_flower_DSC09028_JPG.jpg
"""
