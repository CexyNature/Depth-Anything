import os
import re
import subprocess
import argparse


def create_video_for_each_sequence(input_directory, output_directory, framerate=2):
    """
    Creates a video for each sequence of images using ffmpeg.

    Args:
        input_directory (str): Directory containing the image frames.
        output_directory (str): Directory to save the output videos.
        framerate (int): Frame rate for the videos. Default is 2 frames per second.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the input directory
    files = sorted(os.listdir(input_directory))

    # Regular expression to extract sequence information from the filename
    pattern = re.compile(r"(\d+)_exp(\d+)_L_(\d+)_pred04\.png")

    # Group files by sequence
    sequences = {}
    for file in files:
        match = pattern.match(file)
        if match:
            deployment_code = match.group(1)
            exp_code = match.group(2)
            frame_number = int(match.group(3))
            sequence_key = f"{deployment_code}_exp{exp_code}"

            if sequence_key not in sequences:
                sequences[sequence_key] = []

            sequences[sequence_key].append((frame_number, file))

    # Sort each sequence by frame number
    for sequence_key in sequences:
        sequences[sequence_key].sort()

    # Create a video for each sequence
    for sequence_key, frames in sequences.items():
        sequence_output_dir = os.path.join(output_directory, sequence_key)
        if not os.path.exists(sequence_output_dir):
            os.makedirs(sequence_output_dir)

        # Prepare the sequence of images to be used by ffmpeg
        for i, (frame_number, filename) in enumerate(frames):
            new_filename = f"{i:04d}.png"
            os.rename(
                os.path.join(input_directory, filename),
                os.path.join(sequence_output_dir, new_filename),
            )

        # Define output video file name
        output_video = os.path.join(output_directory, f"{sequence_key}.mp4")

        # Run ffmpeg to create the video
        command = [
            "ffmpeg",
            "-framerate",
            str(framerate),
            "-i",
            os.path.join(sequence_output_dir, "%04d.png"),
            "-c:v",
            "libx264",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            output_video,
        ]

        subprocess.run(command, check=True)

        # Clean up renamed files (optional)
        for i in range(len(frames)):
            os.remove(os.path.join(sequence_output_dir, f"{i:04d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create videos from sequences of image frames."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the image frames.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the output videos will be saved.",
    )
    parser.add_argument(
        "-f",
        "--framerate",
        type=int,
        default=2,
        help="Frame rate for the videos (default: 2 frames per second).",
    )

    args = parser.parse_args()

    create_video_for_each_sequence(args.input_dir, args.output_dir, args.framerate)
