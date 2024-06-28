"""
This script processes images using a depth estimation model to generate depth maps and update the input CSV file with relevant metadata (z-values).

It reads the focal length and other necessary information from a CSV file, applies the model to each image, and saves
the depth maps and corresponding metadata (z-min and z-max values) to the specified output directory and updates the input CSV file.

Set global variables: 
    MODEL: Options: "indoor" or "outdoor" - This is change which model will be used to generate depth maps
    CAMERA: Options: "left" or "right" - This will adjust which camera focal length will be used impacting z-values

Usage:
    python script.py -m model_name -p pretrained_resource -f focal_length_file -i input_directory -o output_directory

Arguments:
    -m, --model: Name of the model to test (default: zoedepth)
    -p, --pretrained_resource: Pretrained resource to use for fetching weights (default: local::./checkpoints/depth_anything_metric_depth_{MODEL}.pt)
    -f, --focal_length_file: CSV file containing the focal length (required)
    -i, --input_dir: Directory containing input images (default: ./my_test/input)
    -o, --output_dir: Directory to save output images (default: ./my_test/output)
"""

import argparse
import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize

# Global settings
FY = 1109
FX = 1109
NYU_DATA = True  # This uses FL as FY and FX
FINAL_HEIGHT = 1080
FINAL_WIDTH = 1920
DATASET = "nyu"  # Let's not pick a fight with the model's dataloader
MODEL = "outdoor"  # Change model here | Options: "indoor" or "outdoor"
CAMERA = "left"  # Change camera here | Options: "left" or "right" | This will adjust which camera focal length will be used


def process_images(model, focal_length_df, focal_length_file):
    """
    Processes images using the specified depth estimation model. It reads the focal length from the CSV file,
    applies the model to each image, and saves the depth maps images to the output directory.
    It will also record calibration-id, deployment-code, frame, focal_length, z-min, and z-max for each image on to a CSV file.

    Args:
        model: The depth estimation model.
        focal_length_df (pd.DataFrame): DataFrame containing focal length data.
        focal_length_file (str): Path to the input CSV file containing the focal length.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(
        os.path.join(INPUT_DIR, "*.jpg")
    )

    # Ensure the new columns exist in the DataFrame
    focal_length_df[f"z-min-{MODEL}-{CAMERA}"] = np.nan
    focal_length_df[f"z-max-{MODEL}-{CAMERA}"] = np.nan

    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Extract the deployment_code, calibration-id, and frame from the image filename
            image_name = os.path.basename(image_path)
            parts = image_name.split("_")
            deployment_code = parts[0]
            calibration_id = parts[1]
            frame = parts[3].split(".")[0]

            # Get the corresponding focal length and other info from the CSV file
            focal_length_row = focal_length_df[
                (focal_length_df["calibration-id"] == calibration_id)
                & (
                    focal_length_df["deployment-code"] == int(deployment_code)
                )  # Convert to int
                & (focal_length_df["frame"] == int(frame))  # Convert to int
            ]
            if focal_length_row.empty:
                print(f"No focal length found for {image_name}, skipping.")
                continue

            FL = focal_length_row[f"focal-length-{CAMERA}"].values[0]
            deployment_code = focal_length_row["deployment-code"].values[0]
            calibration_id = focal_length_row["calibration-id"].values[0]  # exp
            frame = focal_length_row["frame"].values[0]

            # Print the focal length used for the current image
            print(
                f"Processing {image_name} | FL: {FL} | deployment: {deployment_code} | cal_id: {calibration_id} | frame: {frame}"
            )

            color_image = Image.open(image_path).convert("RGB")
            original_width, original_height = color_image.size
            image_tensor = (
                transforms.ToTensor()(color_image)
                .unsqueeze(0)
                .to("cuda" if torch.cuda.is_available() else "cpu")
            )

            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get("metric_depth", pred.get("out"))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]

            predm = pred.squeeze().detach().cpu().numpy()
            if True:
                print("Saving images ...")
                resized_color_image = color_image.resize(
                    (FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS
                )
                resized_pred = Image.fromarray(predm).resize(
                    (FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST
                )

                focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
                x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
                x = (x - FINAL_WIDTH / 2) / focal_length_x
                y = (y - FINAL_HEIGHT / 2) / focal_length_y
                z = np.array(resized_pred)
                points = np.stack(
                    (np.multiply(x, z), np.multiply(y, z), z), axis=-1
                ).reshape(-1, 3)

                z_min = z.min()  # These values will change depending on the MODEL used
                z_max = z.max()  # These values will change depending on the MODEL used

                # Update the DataFrame with z-min and z-max values
                focal_length_df.loc[
                    (focal_length_df["calibration-id"] == calibration_id)
                    & (focal_length_df["deployment-code"] == int(deployment_code))
                    & (focal_length_df["frame"] == int(frame)),
                    [f"z-min-{MODEL}-{CAMERA}", f"z-max-{MODEL}-{CAMERA}"],
                ] = [z_min, z_max]

                print(
                    f"z-min-{MODEL}: {z_min} | z-max-{MODEL}: {z_max}"
                )  # Print z-min and z-max

                z_norm = (z - z_min) / (z_max - z_min)
                imgdepth = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
                o3d.io.write_image(
                    os.path.join(
                        OUTPUT_DIR,
                        os.path.splitext(os.path.basename(image_path))[0]
                        + "_pred04.png",
                    ),
                    imgdepth,
                )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save the updated DataFrame back to the input CSV file
    focal_length_df.to_csv(focal_length_file, index=False)


def main(
    model_name,
    pretrained_resource,
    focal_length_file,
    input_dir,
    output_dir,
):
    """
    Main function that sets up the model and processes images using the specified parameters.

    Args:
        model_name (str): Name of the model to test.
        pretrained_resource (str): Pretrained resource to use for fetching weights.
        focal_length_file (str): CSV file containing the focal length.
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output images.
    """
    global INPUT_DIR, OUTPUT_DIR

    # Read focal length from CSV
    focal_length_df = pd.read_csv(focal_length_file)

    INPUT_DIR = input_dir  # Image input directory
    OUTPUT_DIR = output_dir  # Images output directory

    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource.format(MODEL=MODEL)
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    process_images(model, focal_length_df, focal_length_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="zoedepth", help="Name of the model to test"
    )
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default="local::./checkpoints/depth_anything_metric_depth_{MODEL}.pt",  # Default model
        help="Pretrained resource to use for fetching weights.",
    )
    parser.add_argument(
        "-f",
        "--focal_length_file",
        type=str,
        required=True,
        help="CSV file containing the focal length.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="./my_test/input",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./my_test/output",
        help="Directory to save output images.",
    )

    args = parser.parse_args()
    main(
        args.model,
        args.pretrained_resource,
        args.focal_length_file,
        args.input_dir,
        args.output_dir,
    )
