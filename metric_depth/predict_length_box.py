"""
This script calculates object length from depth maps based on bounding box coordinates
and metadata provided in a CSV file. It processes images and depth maps, computes the
object length in millimeters, and visualizes the results.

Usage:
    python script.py -i input_directory -o output_directory -csv metadata.csv
"""

import os
import numpy as np
import pandas as pd
import cv2
import re
import argparse
import matplotlib.pyplot as plt
from skimage.draw import polygon

SIZE = "size00"  # Change survey name here (only alters output file name) | Options: "size00", s"ize01", "size02" or "size03"
MODEL = "indoor"  # Change name model here | Options: "indoor" or "outdoor" | Changes column names
CAMERA = "left"  # Change camera here | Options: "left" or "right" | This will adjust which camera focal length will be used


def mask_image2image(image_foreground, mask, image_background):
    """
    Combines the foreground and background images using a mask.

    Args:
        image_foreground (np.array): The foreground image.
        mask (np.array): The mask indicating the region to combine.
        image_background (np.array): The background image.

    Returns:
        np.array: The combined image.
    """
    masked_foreground = cv2.bitwise_or(image_foreground, image_foreground, mask=mask)
    mask_inverted = cv2.bitwise_not(mask)
    masked_background = cv2.bitwise_or(
        image_background, image_background, mask=mask_inverted
    )
    out = cv2.bitwise_or(masked_foreground, masked_background)
    return out


def swap_y_if_inverted(y0, y1, inverted):
    """
    Swaps y-coordinates if the line is inverted.

    Args:
        y0 (int): The y-coordinate of the first point.
        y1 (int): The y-coordinate of the second point.
        inverted (bool): Indicates if the line is inverted.

    Returns:
        tuple: The possibly swapped y-coordinates.
    """
    if inverted:
        y0, y1 = y1, y0
    return y0, y1


def calculate_length(dmap, mask, bounding_box_coords, focal_length, zmin, zmax):
    """
    Calculates the length between two points in the depth map.

    Args:
        dmap (np.array): The depth map.
        mask (np.array): The mask indicating the object region.
        bounding_box_coords (np.array): The coordinates of the bounding box.
        focal_length (float): The focal length of the camera.
        zmin (float): The minimum depth value.
        zmax (float): The maximum depth value.

    Returns:
        float: The length in millimeters.

    Raises:
        ValueError: If no depth values are found in the masked region.
    """
    masked_foreground = cv2.bitwise_or(dmap, dmap, mask=mask)
    depth_values = dmap[np.where(mask == 255)].astype(np.float32) / 255.0

    if len(depth_values) == 0:
        raise ValueError("No depth values found in the masked region.")

    depth_mean = depth_values.mean()
    value_metric = (depth_mean * (zmax - zmin)) + zmin
    obj_length_pixels = np.linalg.norm(bounding_box_coords[0] - bounding_box_coords[1])
    obj_length_metric = (value_metric * obj_length_pixels) / focal_length
    return obj_length_metric * 1000  # Convert to millimeters


def process_image(input_image_path, output_depth_map_path, row, results):
    """
    Processes an individual image and its depth map, calculates the object length, and appends the result to the results list.

    Args:
        input_image_path (str): Path to the input image.
        output_depth_map_path (str): Path to the output depth map.
        row (pd.Series): Metadata row containing relevant information.
        results (list): List to append the result to.

    Raises:
        FileNotFoundError: If the input image or depth map is not found.
    """
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dmap = cv2.imread(output_depth_map_path, cv2.IMREAD_UNCHANGED)
    if dmap is None:
        raise FileNotFoundError(f"Depth map not found: {output_depth_map_path}")

    camera_id = row["camera"]
    frame_number = row["frame"]
    calibration_id = row["calibration-id"]  # Uses Left camera Focal Length

    FL = row[f"focal-length-{CAMERA}"]
    zmin = row[f"z-min-{MODEL}-{CAMERA}"]  # Change column name according to model used
    zmax = row[f"z-max-{MODEL}-{CAMERA}"]  # Change column name according to model used

    # if camera_id == "L":
    x0, y0 = row["Lx0"], row["Ly0"]  # Always use L camera
    x1, y1 = row["Lx1"], row["Ly1"]
    # else:
    #    camera_id == "L"
    #    x0, y0 = row["Rx0"], row["Ry0"]
    #    x1, y1 = row["Rx1"], row["Ry1"]

    inverted = str(row["inverted_line"]).strip().lower() == "yes"  # Check if inverted
    y0, y1 = swap_y_if_inverted(y0, y1, inverted)

    fixed_height = 75  # Fixed height
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)

    unit_vector = np.array([dx, dy]) / length
    perpendicular_vector = np.array([-unit_vector[1], unit_vector[0]]) * (
        fixed_height / 2
    )

    top_left = np.array([x0, y0]) - perpendicular_vector
    top_right = np.array([x1, y1]) - perpendicular_vector
    bottom_left = np.array([x0, y0]) + perpendicular_vector
    bottom_right = np.array([x1, y1]) + perpendicular_vector

    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    # Check if any point is outside the image boundaries
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= img.shape[1])
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= img.shape[0])
    ):
        print(f"Skipping image {input_image_path} due to out-of-bound coordinates.")
        return

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    rr, cc = polygon(points[:, 1], points[:, 0])
    mask[rr, cc] = 255

    try:
        length_mm = calculate_length(dmap, mask, points, FL, zmin, zmax)
        image_name = os.path.basename(input_image_path)
        results.append(
            {
                "image_name": image_name,
                "frame": frame_number,
                f"predicted_length_{MODEL}_{CAMERA}": length_mm,
                f"focal_length_{CAMERA}": FL,
                f"zmin-{MODEL}-{CAMERA}": zmin,  # Change column name according to model used
                f"zmax-{MODEL}-{CAMERA}": zmax,  # Change column name according to model used
                "inverted_line": inverted,
            }
        )
        print(
            f"Frame: {frame_number} | Length: {length_mm} mm | FL: {FL} | zmin: {zmin}, zmax: {zmax} | Coords: {x0}, {y0}, {x1}, {y1} | Inverted: {inverted} | Image {image_name}"
        )
    except ValueError as e:
        print(e)


def main(input_dir, output_dir, csv_file_path):
    """
    Main function that iterates through the metadata, processes each image, and saves the results to a CSV file.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory containing output depth maps.
        csv_file_path (str): Path to the CSV file containing metadata.
    """
    metadata = pd.read_csv(csv_file_path)
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
    results = []

    for _, row in metadata.iterrows():
        deployment_code = int(row["deployment-code"])
        camera = row["camera"]
        frame = int(row["frame"])
        calibration_id = row["calibration-id"]

        image_filename = (
            f"{deployment_code}_{calibration_id}_L_{frame}.jpg"  # Always use L camera
        )
        input_image_path = os.path.join(input_dir, image_filename)

        if os.path.exists(input_image_path):
            output_depth_map_path = os.path.join(
                output_dir, image_filename.replace(".jpg", "_pred04.png")
            )
            try:
                process_image(input_image_path, output_depth_map_path, row, results)
            except (FileNotFoundError, ValueError) as e:
                print(e)
        else:
            print(f"Image file not found: {input_image_path}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{SIZE}_predicted_length_box_{MODEL}_{CAMERA}.csv", index=False)
    print(f"{SIZE}_predicted_length_box_{MODEL}_{CAMERA}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate object length from depth map."
    )
    parser.add_argument(
        "-i", "--input_dir", required=True, help="Directory containing input images."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory containing output depth maps.",
    )
    parser.add_argument(
        "-csv",
        "--csv_file",
        required=True,
        help="Path to the CSV file containing metadata.",
    )

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.csv_file)
