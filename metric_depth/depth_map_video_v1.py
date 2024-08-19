import os
import re
import subprocess
import argparse
import shutil


def create_video_for_each_sequence(input_directory, output_directory, framerate=2):
    """
    Creates a video for each sequence of images using ffmpeg, skipping sequences with only one image.

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
    pattern = re.compile(r"(\d+)_exp(\d+)_L_(\d+)\.jpg")

    # Group files by deployment and exp code, then by sequences
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

    # Sort each sequence by frame number and break into sub-sequences
    for sequence_key in sequences:
        sequences[sequence_key].sort()
        sub_sequences = []
        current_sub_sequence = []

        for i, (frame_number, filename) in enumerate(sequences[sequence_key]):
            if i == 0:
                current_sub_sequence.append((frame_number, filename))
            else:
                previous_frame_number = sequences[sequence_key][i - 1][0]
                if frame_number == previous_frame_number + 1:
                    current_sub_sequence.append((frame_number, filename))
                else:
                    sub_sequences.append(current_sub_sequence)
                    current_sub_sequence = [(frame_number, filename)]

        if current_sub_sequence:
            sub_sequences.append(current_sub_sequence)

        # Now process each sub-sequence
        for seq_index, sub_sequence in enumerate(sub_sequences):
            # Skip sub-sequences that contain only one image
            if len(sub_sequence) <= 1:
                continue

            sequence_temp_dir = os.path.join(
                output_directory, f"temp_{sequence_key}_seq{seq_index+1}"
            )
            if not os.path.exists(sequence_temp_dir):
                os.makedirs(sequence_temp_dir)

            # Prepare the sequence of images by copying and renaming
            for i, (frame_number, filename) in enumerate(sub_sequence):
                src_path = os.path.join(input_directory, filename)
                dest_filename = f"{i:04d}.jpg"
                dest_path = os.path.join(sequence_temp_dir, dest_filename)
                shutil.copy(src_path, dest_path)

            # Define output video file name
            output_video = os.path.join(
                output_directory, f"{sequence_key}_seq{seq_index+1}.mp4"
            )

            # Run ffmpeg to create the video
            command = [
                "ffmpeg",
                "-v",
                "debug",  # Add this line
                "-framerate",
                str(framerate),
                "-i",
                os.path.join(sequence_temp_dir, "%04d.jpg"),
                "-c:v",
                "libx264",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                output_video,
            ]

            subprocess.run(command, check=True)

            # Clean up the temporary directory
            shutil.rmtree(sequence_temp_dir)


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

# python3 depth_map_video_v1.py -i /home/shakyafernando/projects/monocular-depth/frames/jade/img/ -o /home/shakyafernando/projects/monocular-depth/frames/jade/vid/
