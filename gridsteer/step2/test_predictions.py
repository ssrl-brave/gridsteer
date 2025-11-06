"""
Test Predictions - Find and Save Frames Closest to Predicted Motor Positions.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse


def load_frame_data(data_path: str, frame_number: int) -> Optional[Tuple[Dict, np.ndarray]]:
    """Load frame data from .npz file."""
    try:
        data = np.load(os.path.join(f"{data_path}", f"test{frame_number}.npz"))
        motor_pos = {
            'x': float(data['x']),
            'y': float(data['y']),
            'z': float(data['z']),
            'phi': float(data['phi'])
        }
        return motor_pos, data['sample']
    except Exception as e:
        return None


def calculate_motor_distance(pos1: Dict, pos2: Dict, use_z: bool = True, use_phi: bool = True) -> float:
    """Calculate Euclidean distance between two motor positions."""
    dx = pos1['x'] - pos2['x']
    dy = pos1['y'] - pos2['y']
    dist_sq = dx**2 + dy**2

    if use_z:
        dz = pos1['z'] - pos2['z']
        dist_sq += dz**2

    if use_phi:
        dphi = pos1['phi'] - pos2['phi']
        if abs(dphi) > 180:
            dphi = 360 - abs(dphi)
        dist_sq += dphi**2

    return np.sqrt(dist_sq)


def find_closest_frames(json_path: str, data_path: str, min_frame: int, max_frame: int,
                       use_z: bool = True, use_phi: bool = True) -> Dict:
    """Find frame closest to each predicted motor position."""
    with open(json_path, 'r') as f:
        predictions = json.load(f)

    print(f"Loaded Predictions From: {json_path}")
    print(f"Reference Frame: {predictions['reference_frame_number']}")
    print(f"Total Wells In Predictions: {len(predictions['well_centering_positions'])}")
    print()

    print(f"Loading Frames {min_frame} To {max_frame}...")
    frames_data = {}
    for frame_num in range(min_frame, max_frame + 1):
        result = load_frame_data(data_path, frame_num)
        if result is not None:
            motor_pos, img = result
            frames_data[frame_num] = {'motor': motor_pos, 'image': img}

    print(f"Loaded {len(frames_data)} Frames")
    print()

    results = {}

    for well_key, well_data in predictions['well_centering_positions'].items():
        predicted_motor = well_data['motor_position']
        closest_frame = None
        min_distance = float('inf')

        for frame_num, frame_data in frames_data.items():
            distance = calculate_motor_distance(
                predicted_motor,
                frame_data['motor'],
                use_z=use_z,
                use_phi=use_phi
            )

            if distance < min_distance:
                min_distance = distance
                closest_frame = frame_num

        results[well_key] = {
            'well_id': well_data['well_id'],
            'row': well_data['row'],
            'column': well_data['column'],
            'predicted_motor': predicted_motor,
            'closest_frame': closest_frame,
            'closest_motor': frames_data[closest_frame]['motor'],
            'distance': min_distance,
            'image': frames_data[closest_frame]['image']
        }

        print(f"Well {well_key}: Closest Frame = {closest_frame}, Distance = {min_distance:.4f}")

    return results


def save_closest_frames(results: Dict, output_dir: str):
    """Save closest frames for each well."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print()
    print(f"Saving Frames To: {output_path}")
    print()

    for well_key, data in results.items():
        row = data['row']
        col = data['column']
        frame_num = data['closest_frame']
        distance = data['distance']

        filename = f"frame_{frame_num}_well_row_{row}_col_{col}.png"
        filepath = output_path / filename

        img = data['image']
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.title(f"Well {well_key} - Frame {frame_num}\n"
                 f"Distance To Predicted: {distance:.4f}\n"
                 f"Motor X: {data['closest_motor']['x']:.2f}, Y: {data['closest_motor']['y']:.2f}, "
                 f"Z: {data['closest_motor']['z']:.2f}, Phi: {data['closest_motor']['phi']:.2f}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {filename}")

    print()
    print(f"All {len(results)} Frames Saved To {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test predictions by finding and saving closest frames"
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default="/people/chen541/bioprep-dev/bioprep-autolab/Test/circ.2/",
        help="Path to data directory containing .npz files"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="test_predictions_frames",
        help="Directory to save output frames"
    )
    parser.add_argument(
        '--min-frame',
        type=int,
        default=0,
        help="Minimum frame number to search"
    )
    parser.add_argument(
        '--max-frame',
        type=int,
        default=48,
        help="Maximum frame number to search"
    )
    parser.add_argument(
        '--no-z',
        dest='use_z',
        action='store_false',
        help="Exclude Z Coordinate From Distance Calculation (Included By Default)"
    )
    parser.add_argument(
        '--no-phi',
        dest='use_phi',
        action='store_false',
        help="Exclude Phi Coordinate From Distance Calculation (Included By Default)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Testing Predictions - Finding Closest Frames")
    print("="*80)
    print()

    results = find_closest_frames(
        args.json,
        args.data_path,
        args.min_frame,
        args.max_frame,
        use_z=args.use_z,
        use_phi=args.use_phi
    )

    save_closest_frames(results, args.output_dir)

    print()
    print("="*80)
    print("Summary Statistics")
    print("="*80)
    distances = [data['distance'] for data in results.values()]
    print(f"Average Distance To Predicted: {np.mean(distances):.4f}")
    print(f"Min Distance: {np.min(distances):.4f}")
    print(f"Max Distance: {np.max(distances):.4f}")
    print(f"Std Deviation: {np.std(distances):.4f}")


if __name__ == "__main__":
    main()
