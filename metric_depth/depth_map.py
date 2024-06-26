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


def process_images(model, focal_length_df, output_csv_file):
    """
    Processes images using the specified depth estimation model. It reads the focal length from the CSV file,
    applies the model to each image, and saves the depth maps images to the output directory.
    It will also record calibration-id, deployment-code, frame, focal_length, z-min, and z-max for each image on to a CSV file.

    Args:
        model: The depth estimation model.
        focal_length_df (pd.DataFrame): DataFrame containing focal length data.
        output_csv_file (str): Path to the output CSV file.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(
        os.path.join(INPUT_DIR, "*.jpg")
    )

    results = []

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

            FL = focal_length_row["focal-length-left"].values[0]
            deployment_code = focal_length_row["deployment-code"].values[0]
            calibration_id = focal_length_row["calibration-id"].values[0]
            frame = focal_length_row["frame"].values[0]

            # Print the focal length used for the current image
            print(
                f"Processing {image_name} with FL: {FL} | deployment: {deployment_code} | cal_id: {calibration_id} | frame: {frame}"
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

                z_min = z.min()
                z_max = z.max()

                results.append(
                    {
                        "calibration-id": calibration_id,
                        "deployment-code": deployment_code,
                        "frame": frame,
                        "focal-length-left": FL,
                        "z-min-indoor": z_min,
                        "z-max-indoor": z_max,
                    }
                )
                print(z_min, z_max)  # Print z-min and z-max

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

    # Create a new DataFrame with the results and save it to a new CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_file, index=False)


def main(
    model_name,
    pretrained_resource,
    focal_length_file,
    input_dir,
    output_dir,
    output_csv_file,
):
    """
    Main function that sets up the model and processes images using the specified parameters.

    Args:
        model_name (str): Name of the model to test.
        pretrained_resource (str): Pretrained resource to use for fetching weights.
        focal_length_file (str): CSV file containing the focal length.
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save output images.
        output_csv_file (str): Path to the output CSV file.
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
    process_images(model, focal_length_df, output_csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="zoedepth", help="Name of the model to test"
    )
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default="local::./checkpoints/depth_anything_metric_depth_indoor.pt",
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
    parser.add_argument(
        "-oc",
        "--output_csv_file",
        type=str,
        required=True,
        help="Path to the output CSV file.",
    )

    args = parser.parse_args()
    main(
        args.model,
        args.pretrained_resource,
        args.focal_length_file,
        args.input_dir,
        args.output_dir,
        args.output_csv_file,
    )
