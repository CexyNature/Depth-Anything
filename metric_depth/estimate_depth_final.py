"""
Summary:
---------
The script reads images from an input directory, extracts depth information using a depth estimation model, and saves the processed images and depth maps to an output directory.
It retrieves focal length values from a CSV file and uses these values during the depth estimation process.
The script handles errors related to file reading and model processing.

Functions:
----------
- process_images(model, focal_length_df):
    Processes images using the specified depth estimation model. It reads the focal length from the CSV file, applies the model to each image, and saves the depth maps and point clouds to the output directory.

- main(model_name, pretrained_resource, focal_length_file, input_dir, output_dir):
    Main function that sets up the model and processes images using the specified parameters.

Example usage:
--------------
python script.py -m zoedepth -p local::./checkpoints/depth_anything_metric_depth_outdoor.pt -f path/to/focal_lengths.csv -i path/to/input/images -o path/to/output/images

Args:
-----
-m, --model: Name of the model to test. (optional)
-p, --pretrained_resource: Pretrained resource to use for fetching weights. (optional)
-f, --focal_length_file: CSV file containing the focal length.
-i, --input_dir: Directory containing input images.
-o, --output_dir: Directory to save output images.
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
NYU_DATA = True
FINAL_HEIGHT = 1080
FINAL_WIDTH = 1920
DATASET = "nyu"  # Let's not pick a fight with the model's dataloader


def process_images(model, focal_length_df):
    """
    Processes images using the specified depth estimation model. It reads the focal length from the CSV file,
    applies the model to each image, and saves the depth maps and point clouds to the output directory.

    Args:
        model: The depth estimation model.
        focal_length_df (pd.DataFrame): DataFrame containing focal length data.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(
        os.path.join(INPUT_DIR, "*.jpg")
    )

    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            # Extract the calibration-id from the image filename
            image_name = os.path.basename(image_path)
            calibration_id = image_name.split("_")[1]

            # Get the corresponding focal length from the CSV file
            focal_length_row = focal_length_df[
                focal_length_df["calibration-id"] == calibration_id
            ]
            if focal_length_row.empty:
                print(f"No focal length found for {image_name}, skipping.")
                continue

            FL = focal_length_row["focal_length"].values[0]

            # Print the focal length used for the current image
            print(f"Processing {image_name} with focal length {FL}")

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
                # If NYU_DATA is False, use FX for focal_length_x and FY for focal_length_y.
                # If NYU_DATA is True, use FL for both focal_length_x and focal_length_y.
                x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
                x = (x - FINAL_WIDTH / 2) / focal_length_x
                y = (y - FINAL_HEIGHT / 2) / focal_length_y
                z = np.array(resized_pred)
                points = np.stack(
                    (np.multiply(x, z), np.multiply(y, z), z), axis=-1
                ).reshape(-1, 3)

                #                Image.fromarray(predm).convert("L").save(
                #                    os.path.join(
                #                        OUTPUT_DIR,
                #                        os.path.splitext(os.path.basename(image_path))[0]
                #                        + "_pred01.png",
                #                    )
                #                )
                #                p = colorize(pred.squeeze().detach().cpu().numpy(), cmap="magma_r")
                #                Image.fromarray(p).save(
                #                    os.path.join(
                #                        OUTPUT_DIR,
                #                        os.path.splitext(os.path.basename(image_path))[0]
                #                        + "_pred02.png",
                #                    )
                #                )
                #
                #                pm = colorize(z, cmap="magma_r")
                #                Image.fromarray(pm).save(
                #                    os.path.join(
                #                        OUTPUT_DIR,
                #                        os.path.splitext(os.path.basename(image_path))[0]
                #                        + "_pred03.png",
                #                    )
                #                )

                z_norm = (z - z.min()) / (z.max() - z.min())
                imgdepth = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
                o3d.io.write_image(
                    os.path.join(
                        OUTPUT_DIR,
                        os.path.splitext(os.path.basename(image_path))[0]
                        + "_pred04.png",
                    ),
                    imgdepth,
                )
                print(z.min(), z.max())  # Print z-min and z-max
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main(model_name, pretrained_resource, focal_length_file, input_dir, output_dir):
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
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    process_images(model, focal_length_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="zoedepth", help="Name of the model to test"
    )
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default="local::./checkpoints/depth_anything_metric_depth_outdoor.pt",
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

# Example Code:
# cd to metric_depth dir then:
# python3 estimate_depth_final.py -f /home/shakyafernando/projects/monocular-depth/home/ubuntu/stereo-app-tnc/data/sizes00.csv -i /home/shakyafernando/projects/monocular-depth/frames/predictions/input/ -o /home/shakyafernando/projects/monocular-depth/frames/predictions/output/
