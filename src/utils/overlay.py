"""
    Script to overlay an image with a mask, highlighting the masked areas with a specified color.
    Developed by Yuri Junqueira Tobias, 2025.
    Date: 2025-07-01
    Computer Science Course, Federal University of Paraná (UFPR), Brazil.
"""

import cv2
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay an image with a mask.")
    parser.add_argument("-i", "--image_path", help="Path to the input image", type=str, required=True)
    parser.add_argument("-m", "--mask_path", help="Path to the mask image", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Path to save the output image", type=str, required=True)
    args = parser.parse_args()

    # Load the specified image and mask
    image = cv2.imread(args.image_path)
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise FileNotFoundError("Image or mask not found. Check the file paths.")

    # Check if the mask is binary and adjust if necessary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # Create an overlay with the desired highlight color
    highlight_color = (144, 238, 144)  # Light green (in BGR)
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[:, :] = highlight_color

    # Create a 3-channel mask for applying the overlay
    mask_3ch = cv2.merge([binary_mask, binary_mask, binary_mask])

    # Apply the overlay where the mask is white
    highlighted = np.where(mask_3ch == 255, overlay, image)

    # Combine the original image with the overlay using transparency
    alpha = 0.2  # Transparency (0 = only original image, 1 = only overlay)
    output = image.copy()
    output = np.where(mask_3ch == 255, cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0), image)

    # Save the output image
    cv2.imwrite(args.output_path, output)