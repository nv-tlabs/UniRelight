# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import os
import glob


def extract_frames_from_folder(input_folder, output_base_dir, frame_rate=10, resize=None, max_frames=None):
    """
    Extract frames from all MP4 videos in a folder and save them as images.

    :param input_folder: Path to the folder containing MP4 video files.
    :param output_base_dir: Directory where the frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    """
    # Find all MP4 files in the input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4")) + glob.glob(os.path.join(input_folder, "*.MP4"))

    if not video_files:
        print(f"No MP4 files found in {input_folder}")
        return

    for video_file in video_files:
        # Get the video filename without extension
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_dir = os.path.join(output_base_dir, video_name)

        # Extract frames for the current video
        extract_frames(video_file, output_dir, frame_rate, resize, max_frames)


def extract_frames(video_path, output_dir, frame_rate=None, resize=None, max_frames=None):
    """
    Extract frames from a video and save them as images.

    :param video_path: Path to the input video file.
    :param output_dir: Directory where the frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    :param resize: Optional tuple (width, height) to resize frames before saving.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, int(fps / frame_rate)) if frame_rate is not None else 1

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if max_frames is not None and saved_frame_count >= max_frames:
                break

            if resize:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame_filename = os.path.join(output_dir, f"{saved_frame_count:05d}.rgb.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    print(f"Frames extracted from {video_path}. Total frames saved: {saved_frame_count}")
    cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="Path to the folder containing MP4 video files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Directory where the frames will be saved.",
    )
    parser.add_argument(
        "--frame_rate",
        type=int,
        default=None,
        help="Number of frames to extract per second.",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize extracted frames to WIDTHxHEIGHT (e.g., 1280x704).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract per video (optional).",
    )

    args = parser.parse_args()

    # Parse resize argument if provided
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.lower().split('x'))
            resize = (width, height)
        except ValueError:
            raise ValueError("Invalid format for --resize. Use WIDTHxHEIGHT, e.g., 640x480.")

    # Call the main function with parsed arguments
    extract_frames_from_folder(args.input_folder, args.output_folder, args.frame_rate, resize, args.max_frames)
