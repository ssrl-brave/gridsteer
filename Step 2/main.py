"""
Well Tracking System for Laboratory Image Analysis
Main orchestration module
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

from well_tracking import WellTrackingSystem

def setup_logging(verbose_mode: bool, log_directory: str = "logs"):
    """Configure logging based on verbose mode"""
    import warnings
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
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
    """Configuration parameters for the well tracking system"""
    verbose_mode: bool = True
    log_directory: str = "logs"

    data_path: str = "/people/chen541/bioprep-dev/bioprep-autolab/Test/circ.2/"
    output_dir: str = "output_video"
    output_images_dir: str = "output_images"
    output_json_dir: str = "output_json"

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

    row_y_tolerance: float = 40
    row_separation_min: float = 150
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
    save_video: bool = True
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

    def get_radius_range(self) -> Tuple[int, int]:
        """Get the minimum and maximum radius for circle detection"""
        if self.radius_min is not None and self.radius_max is not None:
            return self.radius_min, self.radius_max
        else:
            return (self.target_radius - self.radius_range,
                    self.target_radius + self.radius_range)


def main():
    """Main entry point with centralized configuration"""
    config = Config()
    system = WellTrackingSystem(config)
    system.run()


if __name__ == "__main__":
    main()