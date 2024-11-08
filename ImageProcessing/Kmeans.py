import numpy as np
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from ImageProcessing.histogram import load_image, cv2
from MeanShift.MeanShift import apply_mask
from utils.rhs_color_mapper import rgb_to_hex, find_closest_colors_with_ucl
from utils.opencvLogger import logger

def get_dominant_color(image_path, max_k=10, mask_path=None):
    # Open the image file
    image = load_image(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return []
    
    segmented_image = apply_mask(image, mask_path)
    segmented_image_resized = cv2.resize(segmented_image, (100, 100))
    pixels = np.array(segmented_image_resized).reshape((-1, 3))

    # Filter out black or near-black pixels
    b_pixel_threshold = 10
    non_black_pixels = [pixel for pixel in pixels if not (pixel < [b_pixel_threshold, b_pixel_threshold, b_pixel_threshold]).all()]
    if not non_black_pixels:
        logger.error("No significant non-black pixels found.")
        return []

    # Find the optimal K by silhouette score
    best_k = 1
    best_score = -1
    best_kmeans = None

    for k in range(2, max_k + 1):  # Start from 2 to avoid single-cluster scenario
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(non_black_pixels)
        labels = kmeans.labels_
        score = silhouette_score(non_black_pixels, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_kmeans = kmeans

    # Use the best KMeans model
    colors = best_kmeans.cluster_centers_.astype(int)
    labels = best_kmeans.labels_
    counts = Counter(labels)
    
    # Sort colors by frequency
    sorted_colors = sorted(zip(counts.values(), colors), reverse=True)
    dominant_colors = [color for _, color in sorted_colors]
    return segmented_image, dominant_colors, best_k, best_score, (sorted_colors, len(non_black_pixels))


def main():
    parser = argparse.ArgumentParser(description='Get dominant colors of an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--mask_path', type=str, help='Path to the mask file')
    args = parser.parse_args()
    segmented_image, dominant_colors, best_k, best_score, len_info = get_dominant_color(args.image_path, max_k=10, mask_path=args.mask_path)
    logger.info(f"Best K: {best_k}, Best Silhouette Score: {best_score:.4f}.")
    logger.info(f"Number of non-black pixels analized with K-Means: {len_info[1]}")
    for index, color in enumerate(dominant_colors):      
        r, g, b = map(int, color)
        rgb_hex = rgb_to_hex(r, g, b)
        top_5_closest_in_rgb = find_closest_colors_with_ucl(rgb_hex)
        # Display the results
        logger.info(f"{index + 1}. closest colors for RGB ({r},{g},{b}). HEX: {rgb_hex}:")
        logger.info(f"\tPercentage of values {(len_info[0][index][0]/len_info[1])*100:.2f}%")
        for color in top_5_closest_in_rgb:
            label, rgb, distance, ucl_name = color
            logger.info(f"\t\tLabel: {label}, RGB: {rgb}, CIE2000 Distance: {distance:.2f}, UCL Name: {ucl_name}")

if __name__ == "__main__":
    main()

"""
python -m ImageProcessing.Kmeans /Users/giovannilopez/Downloads/2024-08-15_Cultivos/calibrated_all_flowers_cie/flower_DSC09028_JPG.jpg --mask_path /Users/giovannilopez/Downloads/2024-08-15_Cultivos/segmented_images_intelligent_scissors/segmented_flower_DSC09028_JPG.jpg
"""
