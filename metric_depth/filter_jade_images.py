"""
Script to copy images associated with a specific user ('Jade') from a source directory 
to a target directory based on a CSV file. The script reads the CSV file to determine 
the deployment code, calibration ID, and frame number, constructs the corresponding 
image filenames, and copies them if they exist in the source directory.

Usage:
    python script_name.py -c <csv_file> -s <source_dir> -t <target_dir>

Arguments:
    -c, --csv_file  : Path to the CSV file containing image metadata.
    -s, --source_dir: Path to the source directory containing images.
    -t, --target_dir: Path to the target directory where images will be copied.
"""

import os
import pandas as pd
import shutil
import argparse


def copy_images(csv_file, source_dir, target_dir):
    """
    Copy images associated with the user 'Jade' from the source directory to
    the target directory based on the information in the CSV file.

    Args:
        csv_file (str): Path to the CSV file containing image metadata.
        source_dir (str): Path to the directory containing the source images.
        target_dir (str): Path to the directory where the images will be copied.

    Returns:
        None
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over the DataFrame and copy images
    count = 0
    for index, row in df.iterrows():
        if row["user"] == "Jade":
            # Construct the image filename
            image_name = f"{row['deployment-code']}_{row['calibration-id']}_L_{row['frame']}.jpg"  # Add "pred04.png" for depth maps

            # Construct full file paths
            source_path = os.path.join(source_dir, image_name)
            target_path = os.path.join(target_dir, image_name)

            # Check if the image exists in the source directory and copy it
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                count += 1
                print(f"Copied {image_name} to {target_dir}")
            else:
                print(f"Image {image_name} not found in the source directory.")
        print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy images associated with user 'Jade' from source to target directory."
    )
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True, help="Path to the CSV file."
    )
    parser.add_argument(
        "-s",
        "--source_dir",
        type=str,
        required=True,
        help="Path to the source directory containing images.",
    )
    parser.add_argument(
        "-t",
        "--target_dir",
        type=str,
        required=True,
        help="Path to the target directory where images will be copied.",
    )

    args = parser.parse_args()

    copy_images(args.csv_file, args.source_dir, args.target_dir)
