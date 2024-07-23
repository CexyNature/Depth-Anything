import os
import numpy as np
import pandas as pd
import cv2
import argparse

SIZE = "size00"  # Change survey name here (only alters output file name) | Options: "size00", s"ize01", "size02" or "size03"
MODEL = "indoor"  # Change name model here | Options: "indoor" or "outdoor" | Changes column names
CAMERA = "left"  # Change camera here | Options: "left" or "right" | This will adjust which camera focal length will be used


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


def calculate_length(dmap, point1, point2, focal_length, zmin, zmax):
    """
    Calculates the length between two points in the depth map.

    Args:
        dmap (np.array): The depth map.
        point1 (tuple): The (x, y) coordinates of the first point.
        point2 (tuple): The (x, y) coordinates of the second point.
        focal_length (float): The focal length of the camera.
        zmin (float): The minimum depth value.
        zmax (float): The maximum depth value.

    Returns:
        tuple: The length in millimeters and the value metric in millimeters.
    """
    x0, y0 = point1
    x1, y1 = point2
    dx = x1 - x0
    dy = y1 - y0
    # This is the Euclidean distance between point1 and point2 in pixel units, calculated using the Pythagorean theorem.
    length_pixels = np.sqrt(dx**2 + dy**2)

    # This is the midpoint of the line segment connecting point1 and point2. It is used to find a representative depth value from the depth map for the segment.
    midpoint = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))

    # This is the depth value at the midpoint in the depth map. It is scaled from an 8-bit value (0-255) to a floating-point value (0.0-1.0).
    depth_value = dmap[midpoint[1], midpoint[0]].astype(np.float32) / 255.0

    # This is the real-world depth value at the midpoint, calculated by scaling depth_value using zmin and zmax.
    value_metric = (depth_value * (zmax - zmin)) + zmin

    # This is the length of the line segment connecting point1 and point2 in real-world units (meters), calculated by converting length_pixels to real-world distance using value_metric and focal_length.
    length_metric = (value_metric * length_pixels) / focal_length

    # The function returns a tuple containing length_metric and value_metric, both converted to millimeters.
    return length_metric * 1000, value_metric * 1000  # Convert to millimeters


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
    calibration_id = row["calibration-id"]

    FL = row[f"focal-length-{CAMERA}"]  # Uses Left camera Focal Length
    zmin = row[f"z-min-{MODEL}-{CAMERA}"]  # Change column name according to model used
    zmax = row[f"z-max-{MODEL}-{CAMERA}"]  # Change column name according to model used

    # if camera_id == "L":
    x0, y0 = row["Lx0"], row["Ly0"]  # Always use L camera
    x1, y1 = row["Lx1"], row["Ly1"]
    # else:
    #    camera_id == "L"
    #    x0, y0 = row["Rx0"], row["Ry0"]
    #    x1, y1 = row["Rx1"], row["Ry1"]

    inverted = str(row["inverted_line"]).strip().lower() == "yes"
    y0, y1 = swap_y_if_inverted(y0, y1, inverted)

    point1 = (x0, y0)
    point2 = (x1, y1)

    try:
        length_mm, value_metric_mm = calculate_length(
            dmap, point1, point2, FL, zmin, zmax
        )
        image_name = os.path.basename(input_image_path)
        results.append(
            {
                "image_name": image_name,
                "frame": frame_number,
                f"predicted_length_{MODEL}_{CAMERA}": length_mm,
                f"value_metric_{MODEL}_{CAMERA}": value_metric_mm,
                f"focal_length_{CAMERA}": FL,
                f"zmin-{MODEL}-{CAMERA}": zmin,  # Change column name according to model used
                f"zmax-{MODEL}-{CAMERA}": zmax,  # Change column name according to model used
                "inverted_line": inverted,
            }
        )
        print(
            f"Frame: {frame_number} | Length: {length_mm} mm | Value Metric: {value_metric_mm} mm | FL: {FL} | zmin: {zmin}, zmax: {zmax} | Coords: {x0}, {y0}, {x1}, {y1} | Inverted: {inverted} | Image {image_name}"
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
                results.append(
                    {
                        "image_name": image_filename,
                        "frame": frame,
                        f"predicted_length_{MODEL}_{CAMERA}": "N/A",
                        f"value_metric_{MODEL}_{CAMERA}": "N/A",
                        f"focal_length_{CAMERA}": row[f"focal-length-{CAMERA}"],
                        f"zmin-{MODEL}-{CAMERA}": row[
                            f"z-min-{MODEL}-{CAMERA}"
                        ],  # Change column name according to model used
                        f"zmax-{MODEL}-{CAMERA}": row[
                            f"z-max-{MODEL}-{CAMERA}"
                        ],  # Change column name according to model used
                        "inverted_line": str(row["inverted_line"]).strip().lower()
                        == "yes",
                    }
                )
        else:
            print(f"Image file not found: {input_image_path}")
            results.append(
                {
                    "image_name": image_filename,
                    "frame": frame,
                    f"predicted_length_{MODEL}_{CAMERA}": "N/A",
                    f"value_metric_{MODEL}_{CAMERA}": "N/A",
                    f"focal_length_{CAMERA}": row[f"focal-length-{CAMERA}"],
                    f"zmin-{MODEL}-{CAMERA}": "N/A",  # Change column name according to model used
                    f"zmax-{MODEL}-{CAMERA}": "N/A",  # Change column name according to model used
                    "inverted_line": "N/A",
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{SIZE}_predicted_length_line_{MODEL}_{CAMERA}.csv", index=False)
    print(f"Results saved to {SIZE}_predicted_length_line_{MODEL}_{CAMERA}.csv")


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
