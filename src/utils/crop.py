"""
    Script to crop images to a square format, resize them, and save them in a specified format (PNG or NumPy array).
    Developed by Yuri Junqueira Tobias, 2025.
    Date: 2025-07-01
    Computer Science Course, Federal University of Paraná (UFPR), Brazil.
"""

import os
import re
import argparse
import numpy as np
from PIL import Image

def extract_number(filename):
    base = os.path.splitext(filename)[0]
    match = re.match(r"(\d+)", base.split("-")[0])
    return int(match.group(1)) if match else float('inf')

def sort_and_filter_files(input_folder, filter_file=None, file_number_min=None, file_number_max=None):
    """Ordena e filtra os arquivos de imagem no diretório de entrada."""
    files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ], key=extract_number)

    if filter_file:
        with open(filter_file, "r") as f:
            filter_lines = f.readlines()
            filter_numbers = [int(line.strip().split('-')[0]) for line in filter_lines[1:]]
        
        files = [f for f in files if extract_number(f) in filter_numbers]
    if file_number_min is not None:
        files = [f for f in files if extract_number(f) >= file_number_min]
    if file_number_max is not None:
        files = [f for f in files if extract_number(f) <= file_number_max]

    return files
    
def crop_images(args, filtered_files):
    for i, filename in enumerate(filtered_files, start=1):
        input_path = os.path.join(args.input_folder, filename)
        if args.output_type == 'npy':
            output_filename = f"{os.path.splitext(filename)[0]}.npy"
        else:
            output_filename = f"m{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(args.output_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                # img = img.point(lambda x: 255 if x > 250 else 0, mode='1')
                
                width, height = img.size
                square_size = min(width, height) 

                left = (width - square_size) // 2
                top = (height - square_size) // 2
                right = left + square_size
                bottom = top + square_size
                crop_box = (left, top, right, bottom)

                cropped = img.crop(crop_box)
                resized = cropped.resize(args.resize_to, Image.LANCZOS)
                if args.output_type == 'png':
                    resized.save(output_path)
                    print(f"{i}/{len(filtered_files)} - Saved: {output_filename}")
                else:
                    arr = np.array(resized).astype(np.uint8)
                    np.save(output_path, arr)
                    print(f"{i}/{len(filtered_files)} - Array saved: {output_filename}")

        except Exception as e:
            print(f"Error with {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and crop images.")
    parser.add_argument("-if", "--input_folder", help="Folder containing the input images.",
                        type=str, required=True)
    parser.add_argument("-of", "--output_folder", help="Folder to save the processed images.",
                        type=str, required=True)
    parser.add_argument("-ff", "--filter_file", help="Path to the filter file (CSV) to limit the images processed.",
                        type=str, default=None)
    parser.add_argument("-mn", "--min_file_number", help="Minimum file number to process.",
                        type=int, default=None)
    parser.add_argument("-mx", "--max_file_number", help="Maximum file number to process.",
                        type=int, default=None)
    parser.add_argument("-rt", "--resize_to", help="Resize dimensions (width, height) as a tuple.",
                        type=int, nargs=2, default=(256, 256))
    parser.add_argument("-ot", "--output_type", help="Output type: 'png' or 'npy'.",
                        type=str, choices=['png', 'npy'], default='png')
    args = parser.parse_args()

    print(f"Parameters received: {args}")

    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder , exist_ok=True)

    # Sort and filter files
    filtered_files = sort_and_filter_files(
        args.input_folder,
        filter_file=args.filter_file,
        file_number_min=args.min_file_number,
        file_number_max=args.max_file_number
    )

    print(f"Filtered files: {filtered_files}")

    # Crop and process images
    crop_images(args, filtered_files)
