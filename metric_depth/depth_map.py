import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from zoedepth.utils.misc import colorize

# Global settings
# FL = 715.0873
# FY = 256 * 0.6
# FX = 256 * 0.6
FL = 1316.34  # 1419.13
FY = 1109  # 1316.34
FX = 1109  # 1316.34
NYU_DATA = False
FINAL_HEIGHT = 1080
FINAL_WIDTH = 1920
INPUT_DIR = "./my_test/input"
OUTPUT_DIR = "./my_test/output"
DATASET = "nyu"  # Lets not pick a fight with the model's dataloader


def process_images(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png")) + glob.glob(
        os.path.join(INPUT_DIR, "*.jpg")
    )
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
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
            # pred = pred.squeeze().detach().cpu().numpy()

            predm = pred.squeeze().detach().cpu().numpy()
            if True:
                print("Saving images ...")
                # Resize color image and depth to final size
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

                # Image.fromarray(points).convert("L").save(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + "_pred01.png"))

                # os.makedirs(config.save_images, exist_ok=True)
                # def save_image(img, path):
                # d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
                # p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
                # im = transforms.ToPILImage()(image.squeeze().cpu())
                # im.save(os.path.join(config.save_images, f"{i}_img.png"))
                # Image.fromarray(d).save(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".png"))
                # if new_p.mode != 'RGB':
                Image.fromarray(predm).convert("L").save(
                    os.path.join(
                        OUTPUT_DIR,
                        os.path.splitext(os.path.basename(image_path))[0]
                        + "_pred01.png",
                    )
                )
                # pred = colorize(pred, 0, 10)
                p = colorize(pred.squeeze().detach().cpu().numpy(), cmap="magma_r")
                Image.fromarray(p).save(
                    os.path.join(
                        OUTPUT_DIR,
                        os.path.splitext(os.path.basename(image_path))[0]
                        + "_pred02.png",
                    )
                )

                pm = colorize(z, cmap="magma_r")
                Image.fromarray(pm).save(
                    os.path.join(
                        OUTPUT_DIR,
                        os.path.splitext(os.path.basename(image_path))[0]
                        + "_pred03.png",
                    )
                )

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
                # Image.fromarray(z).save(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + "_pred04.png"))
                # print(pred.shape, predm.shape, z.shape, p.shape, pm.shape, points.shape, x.shape, y.shape, z.dtype)
                print(z.min(), z.max())
                # Image.fromarray(pred).convert("L").save(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + "_pred02.png"))

            # # Resize color image and depth to final size
            # resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
            # resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

            # focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
            # x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
            # x = (x - FINAL_WIDTH / 2) / focal_length_x
            # y = (y - FINAL_HEIGHT / 2) / focal_length_y
            # z = np.array(resized_pred)
            # points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            # colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    process_images(model)


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

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
