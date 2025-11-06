"""
Well Tracking System for Laboratory Image Analysis
Main orchestration module
"""

import argparse
import glob
import os
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

from gridsteer.step2.well_tracking import WellTrackingSystem

def setup_logging(verbose_mode: bool, log_directory: str = "logs"):
    """Configure logging based on verbose mode."""
    import warnings
    
    #for handler in logger.handlers[:]:
    #    logger.removeHandler(handler)
    
    if verbose_mode:
        log_dir = Path(log_directory)
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"well_tracking_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.addHandler(file_handler)
        warnings_logger.setLevel(logging.WARNING)
        
        logger.info(f"Verbose Logging Enabled - Log File: {log_file}")
    else:
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.setLevel(logging.CRITICAL)


@dataclass
class Config:
    """Configuration parameters for well tracking system."""
    verbose_mode: bool = True

    data_path: str = "/people/chen541/bioprep-dev/bioprep-autolab/Test/circ.2/"
    output_root: str = "./"
    json_filename: str = "mapping.json"

    min_frame: int = 0
    max_frame: int = 48

    target_radius: int = 85
    radius_min: Optional[int] = None
    radius_max: Optional[int] = None
    radius_range: int = 5
    min_x_distance: int = 150
    min_y_distance: int = 150
    hough_num_peaks: int = 19
    hough_threshold: float = 0.2

    total_wells_row1: int = 9
    total_wells_row2: int = 10
    total_wells: int = 19

    row_y_tolerance: int = 40
    row_separation_min: int = 150
    initial_row_layout_flipped: bool = False

    ransac_max_trials: int = 100
    association_distance_threshold: float = 50.0

    min_circles_per_row: int = 2
    max_line_angle_degrees: float = 10.0

    line_hough_threshold: int = 80
    line_min_distance: int = 20
    line_min_angle: int = 80
    line_num_peaks: int = 4

    backup_edge_sigma: float = 15.0
    backup_edge_low_threshold: float = 0.2
    backup_edge_high_threshold: float = 0.7

    enable_edge_detection: bool = False
    edge_distance_multiplier: float = 2.0

    use_background_removal: bool = False
    rembg_model: Optional[str] = "birefnet-general-lite"

    video_fps: int = 10
    save_video: bool = False
    save_individual_frames: bool = True
    save_json_output: bool = True
    display_frames: bool = False

    enable_well_tracking: bool = True
    track_well_centers: bool = True
    enable_motor_calibration: bool = True

    calibration_pairing_strategy: str = "random"
    calibration_use_multi_frame: bool = True
    calibration_use_average_movement: bool = True
    calibration_min_common_wells: int = 1

    calibration_min_samples: int = 10
    calibration_max_samples: Optional[int] = 100
    calibration_method: str = "linear"
    calibration_alpha: float = 1.0
    calibration_spline_smoothing: float = 0.5

    edge_sigma: float = 10.0
    edge_low_threshold: float = 0.15
    edge_high_threshold: float = 0.7
    hull_min_area: int = 200
    border_buffer: int = 2

    @property
    def output_dir(self):
        return os.path.join( self.output_root,"output_video")
    
    @property
    def output_images_dir( self): 
        return os.path.join( self.output_root, "output_images")

    @property
    def output_json_dir(self):
        return os.path.join(self.output_root,  "output_json")
    
    @property
    def log_directory(self):
        return os.path.join(self.output_root,  "logs")

    def get_radius_range(self) -> Tuple[int, int]:
        """Get minimum and maximum radius for circle detection."""
        if self.radius_min is not None and self.radius_max is not None:
            return self.radius_min, self.radius_max
        else:
            return (self.target_radius - self.radius_range,
                    self.target_radius + self.radius_range)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Persistent Frame Processor For Line Analysis')
    parser.add_argument('data_path', type=str, help='Path To Data Directory Containing .npz Files')
    parser.add_argument('--imgs_to_proc', type=int, help='Number Of Images To Process', default=None)
    parser.add_argument('--target_radius', type=int, help='Target Radius For Circle Detection', default=85)
    parser.add_argument("--outdir", type=str, help="output folder path")
    parser.add_argument("--saveVideo", action="store_true")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    all_frames = glob.glob(f"{args.data_path}/test*npz")
    frames_total = len(all_frames)
    print("Frames total=%d" % frames_total)

    config = Config()
    config.output_root = args.outdir
    os.makedirs(args.outdir, exist_ok=True)
    config.data_path = args.data_path
    config.target_radius = args.target_radius
    config.min_frame = 0
    config.max_frame = frames_total
    config.save_video = args.saveVideo

    # Calculate distance values based on target radius
    config.min_x_distance = round(1.5 * config.target_radius)
    config.min_y_distance = round(1.5 * config.target_radius)
    config.row_separation_min = round(1.5 * config.target_radius)
    config.row_y_tolerance = math.floor(config.target_radius / 2)

    config.max_frames_to_process = args.imgs_to_proc if args.imgs_to_proc is not None else frames_total

    system = WellTrackingSystem(config)
    system.run()


if __name__ == "__main__":
    main()
