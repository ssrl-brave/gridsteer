#!/usr/bin/env python3
"""Combine PNG frames into a video, sorted by numbers in filenames."""

import argparse
from pathlib import Path
import cv2
from natsort import natsorted


def create_video_from_frames(input_folder, output_name='output.mp4', fps=30):
    """Create a video from PNG frames in a folder."""
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' Does Not Exist")
        return
    
    png_files = list(input_path.glob('*.png'))
    
    if not png_files:
        print(f"Error: No PNG Files Found In '{input_folder}'")
        return
    
    print(f"Found {len(png_files)} PNG Files")
    
    # Sort files naturally by name
    sorted_files = natsorted(png_files, key=lambda x: x.name)
    
    # Read first frame to determine video dimensions
    first_frame = cv2.imread(str(sorted_files[0]))
    if first_frame is None:
        print(f"Error: Could Not Read {sorted_files[0]}")
        return
    
    height, width, _ = first_frame.shape
    output_path = input_path / output_name
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could Not Create Video Writer")
        return
    
    print(f"Creating Video At {fps} FPS ({width}x{height})")
    
    # Process each frame
    for i, file_path in enumerate(sorted_files):
        frame = cv2.imread(str(file_path))
        
        if frame is None:
            print(f"Warning: Could Not Read {file_path}, Skipping")
            continue
        
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
        
        if (i + 1) % 10 == 0 or i == len(sorted_files) - 1:
            print(f"Processed {i + 1}/{len(sorted_files)} Frames", end='\r')
    
    print()
    video_writer.release()
    
    print(f"Video Created Successfully: {output_path}")
    print(f"Total Frames: {len(sorted_files)}")
    print(f"Duration: {len(sorted_files) / fps:.2f} Seconds")


def main():
    parser = argparse.ArgumentParser(
        description='Combine PNG frames into a video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frames_to_video.py /path/to/frames
  python frames_to_video.py /path/to/frames --output my_video.mp4 --fps 24
        """
    )
    
    parser.add_argument(
        'input_folder',
        help='Path to folder containing PNG frames'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output.mp4',
        help='Output video filename (default: output.mp4)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second (default: 10)'
    )
    
    args = parser.parse_args()
    create_video_from_frames(args.input_folder, args.output, args.fps)


if __name__ == '__main__':
    main()