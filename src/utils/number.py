"""
    Script to list image files in a specified range and create a CSV file with their names (without extensions).
    Developed by Yuri Junqueira Tobias, 2025.
    Date: 2025-07-01
    Computer Science Course, Federal University of Paraná (UFPR), Brazil.
"""

import os
import re
import argparse

def extract_number(filename):
    base = os.path.splitext(filename)[0]
    match = re.match(r"(\d+)", base.split("-")[0])
    return int(match.group(1)) if match else float('inf')

def list_numbers(args):
    # Sort and filter files based on the provided range
    image_files = sorted([
        f for f in os.listdir(args.input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ], key=extract_number)

    filtered_files = [
        f for f in image_files
        if args.initial_value <= extract_number(f) <= args.end_value
    ]

    # Write the filtered file names to the output CSV
    with open(args.output_path, "w") as f:
        f.write("img\n")
        for filename in filtered_files:
            name_without_ext = os.path.splitext(filename)[0]
            f.write(f"{name_without_ext}\n")

    print(f"File '{args.output_path}' created with file names (without extension) from {args.initial_value} to {args.end_value}, sorted by prefix.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image files and create a CSV with file names in a specified range.")
    parser.add_argument("-iv", "--initial_value", help="Initial value of the range", type=int, required=True)
    parser.add_argument("-ev", "--end_value", help="End value of the range", type=int, required=True)
    parser.add_argument("-i", "--input_folder", help="Folder to be scanned", type=str, required=True)
    parser.add_argument("-o", "--output_path", help="Output file path", type=str, required=True)
    args = parser.parse_args()

    if args.initial_value > args.end_value:
        raise ValueError("Initial value must be less than or equal to end value.")
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Input folder '{args.input_folder}' does not exist.")
    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f"Input path '{args.input_folder}' is not a directory.")
    if not args.output_path.endswith(".csv"):
        raise ValueError("Output path must end with '.csv'.")
    
    list_numbers(args)