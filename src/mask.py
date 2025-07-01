"""
    Script to generate masks from CSV or JSON files containing vertebra coordinates (4 points per vertebra).
    And to generate masks using the SAM (Segment Anything Model) from Meta.
    Developed by Yuri Junqueira Tobias, 2025.
    Date: 2025-07-01
    Computer Science Course, Federal University of Paraná (UFPR), Brazil.
"""

import os
import re
import cv2
import csv
import torch
import argparse
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

def sort_files(annotations_path, images_path):
    """
    Sorts the annotations and images files based on the file number extracted from the filename.
    """
    
    sorted_annotations_file = sorted(
        [f for f in os.listdir(annotations_path) if f.endswith('.csv')],
        key=lambda x: int(x.split('-')[0])  # Extract the file number before the first hyphen
    )
    
    sorted_images_file = sorted(
        [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))],
        key=lambda x: int(x.split('-')[0])  # Extract the file number before the first hyphen
    )

    return sorted_annotations_file, sorted_images_file

def filter_annotations_file(sorted_annotations_file, sorted_images_file, filter_file, min_file_number=None, max_file_number=None):
    """
    Filters the sorted annotations and images files based on a filter file and optional min/max file numbers.
    """
    if filter_file:
        with open(filter_file, 'r') as f:
            filter_lines = f.readlines()
            filter_numbers = [int(line.strip().split('-')[0]) for line in filter_lines[1:]]  # Skip header

        filtered_annotations_file = [f for f in sorted_annotations_file if int(f.split('-')[0]) in filter_numbers]
        filtered_images_file = [f for f in sorted_images_file if int(f.split('-')[0]) in filter_numbers]
    else:
        filtered_annotations_file = sorted_annotations_file
        filtered_images_file = sorted_images_file

    if min_file_number is not None:
        filtered_annotations_file = [f for f in filtered_annotations_file if int(f.split('-')[0]) >= min_file_number]
        filtered_images_file = [f for f in filtered_images_file if int(f.split('-')[0]) >= min_file_number]

    if max_file_number is not None:
        filtered_annotations_file = [f for f in filtered_annotations_file if int(f.split('-')[0]) <= max_file_number]
        filtered_images_file = [f for f in filtered_images_file if int(f.split('-')[0]) <= max_file_number]

    return filtered_annotations_file, filtered_images_file

def create_bounding_boxes_from_coordinates(annotations_path):
    bboxes = []
    with open(annotations_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)
        if len(lines) % 2 != 0:
            lines = lines[:-1] # Remove the last line if it is odd
        for i in range(0, len(lines), 2):
            line1 = list(map(float, lines[i][:4]))
            line2 = list(map(float, lines[i+1][:4]))
            points = line1 + line2  # x1, y1, x2, y2, x3, y3, x4, y4
            xs = points[0::2]
            ys = points[1::2]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)
            bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

def generate_masks_from_annotations(args):
    """
    Generates masks from annotations in CSV or JSON format.
    """
    sorted_annotations_file, sorted_images_file = sort_files(args.annotations_path, args.images_path)
    filtered_annotations_file, _ = filter_annotations_file(sorted_annotations_file, sorted_images_file, args.filter_file, args.min_file_number, args.max_file_number)

    for i, annotation_file in enumerate(filtered_annotations_file, start=1):
        # Extract the file id from the filename that comes before the first hyphen
        file_id = os.path.splitext(annotation_file)[0].split("-")[0]
        print(f"Processing file {i}/{len(filtered_annotations_file)}: {annotation_file} (ID: {file_id})")

        # Construct the full path to the annotation file
        annotation_file_path = os.path.join(args.annotations_path, annotation_file)
        # Construct the full path to the corresponding image file
        image_file_path = os.path.join(args.images_path, f"{os.path.splitext(annotation_file)[0]}.jpg")
        # Construct the output path for the mask
        output_mask_path = os.path.join(args.output_path, f"{os.path.splitext(annotation_file)[0]}.png")

        # Check if the image file exists
        if not os.path.exists(image_file_path):
            print(f"Image file {image_file_path} not found. Skipping...")
            continue

        # Load the image to get its shape
        img = cv2.imread(image_file_path)
        shape_imagem = img.shape[:2]  # (height, width)

        # Generate the mask
        mask = np.zeros(shape_imagem, dtype=np.uint8)

        with open(annotation_file_path, 'r') as f:
            reader = list(csv.reader(f)) 

        if len(reader) % 2 != 0:
            print(f"Odd number of lines in {annotation_file}. The last line will be ignored.")
            reader = reader[:-1]

        for j in range(0, len(reader), 2):
            try:
                sup = list(map(float, reader[j][:4]))  # Superior corners
                inf = list(map(float, reader[j+1][:4]))  # Inferior corners

                # Extract points
                top_left = (sup[0], sup[1])
                top_right = (sup[2], sup[3])
                bottom_left = (inf[0], inf[1])
                bottom_right = (inf[2], inf[3])

                # Create the vertebra polygon
                polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))

                # Fill the polygon in the mask
                cv2.fillPoly(mask, [polygon], 255)

            except Exception as e:
                print(f"Error processing lines {j} and {j+1} in {annotation_file}: {e}")
                continue
        
        # Save the mask
        cv2.imwrite(output_mask_path, mask)
        print(f"Mask saved to {output_mask_path}")

def generate_masks_from_sam(args):
    """
    Generates masks using the Segment Anything Model (SAM) from Meta.
    """
    sorted_annotations_file, sorted_images_file = sort_files(args.annotations_path, args.images_path)
    filtered_annotations_file, sorted_images_file = filter_annotations_file(sorted_annotations_file, sorted_images_file, args.filter_file, args.min_file_number, args.max_file_number)

    # Configuration and initialization for the Segment Anything Model (SAM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_checkpoint = "sam_vit_h.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    for annotation_file, image_file in zip(filtered_annotations_file, sorted_images_file):
        # Defining the paths for the annotation, image and output files
        annotation_file_path = os.path.join(args.annotations_path, annotation_file)
        image_file_path = os.path.join(args.images_path, image_file)
        output_mask_path = os.path.join(args.output_path, f"{os.path.splitext(annotation_file)[0]}.png")
        
        # Load the image
        image = cv2.imread(image_file_path)
        if image is None:
            print(f"Image {image_file_path} not found or could not be read. Skipping...")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Create bounding boxes from the annotation file
        boxes = create_bounding_boxes_from_coordinates(annotation_file_path)
        print(f"Bounding boxes for {annotation_file}: {boxes}")

        # Generate the combined mask using the SAM model
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for box in boxes:
            input_box = np.array(box)[None, :]  # shape (1, 4)
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=False
            )
            mask = masks[0].astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)
        
        # Save the combined mask
        cv2.imwrite(output_mask_path, (combined_mask * 255).astype(np.uint8))
        print(f"Mask saved to {output_mask_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create masks from vertebra annotations in CSV or JSON format. And also from SAM model.")
    parser.add_argument("-ap", "--annotations_path", help="Path to the annotations file (CSV or JSON).",
                        type=str, required=True)
    parser.add_argument("-ip", "--images_path", help="Path to the image file to generate the mask for.",
                        type=str, required=True)
    parser.add_argument("-op", "--output_path", help="Path to save the generated mask.",
                        type=str, required=True)
    parser.add_argument("-ff", "--filter_file", help="Path to the filter file (CSV) to limit the images processed.",
                        type=str, default=None)
    parser.add_argument("-mn", "--min_file_number", help="Minimum file number to process.",
                        type=int, default=None)
    parser.add_argument("-mx", "--max_file_number", help="Maximum file number to process.",
                        type=int, default=None)
    parser.add_argument("-a", "--aproach", help="Approach to use for generating the mask: 'c' for coordinates or 's' for SAM model.",
                        type=str, choices=["c", "s"], default="c")
    args = parser.parse_args()

    print(f"Parameters received: {args}")

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path , exist_ok=True)

    # Calls the function to generate the mask based on the approach
    if args.aproach == "c":
        generate_masks_from_annotations(args)
    elif args.aproach == "s":
        generate_masks_from_sam(args)
