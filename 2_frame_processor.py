"""
Well Tracking System for Laboratory Image Analysis
Maps pixel coordinates to motor coordinates for two-row well configurations
"""

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks
from sklearn.linear_model import RANSACRegressor, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        
        logger.info(f"Verbose Logging Enabled. Log File: {log_file}")
    else:
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.setLevel(logging.CRITICAL)


def well_id_to_row_col(well_id: int, config: 'Config') -> Tuple[int, int]:
    """Convert well ID to (row, column) format"""
    if well_id <= config.total_wells_row1:
        return (1, well_id)
    else:
        return (2, well_id - config.total_wells_row1)


def row_col_to_well_id(row: int, col: int, config: 'Config') -> int:
    """Convert (row, column) to well ID"""
    if row == 1:
        return col
    else:
        return config.total_wells_row1 + col


def format_well_label(well_id: int, config: 'Config') -> str:
    """Format well ID as (row, column) string"""
    row, col = well_id_to_row_col(well_id, config)
    return f"({row},{col})"


def calculate_angle_difference(phi1: float, phi2: float) -> float:
    """Calculate minimum angle difference between two phi values, accounting for wraparound"""
    diff = abs(phi1 - phi2)
    return min(diff, 360 - diff) if diff > 180 else diff


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def create_circular_mask(shape: Tuple[int, int], center: Tuple[float, float], 
                        outer_radius: float, inner_radius: float = 0) -> np.ndarray:
    """Create circular or annular mask"""
    y_indices, x_indices = np.ogrid[:shape[0], :shape[1]]
    distances_sq = (x_indices - center[0])**2 + (y_indices - center[1])**2
    
    if inner_radius > 0:
        return (distances_sq <= outer_radius**2) & (distances_sq >= inner_radius**2)
    else:
        return distances_sq <= outer_radius**2


@dataclass
class Config:
    """Configuration parameters for the well tracking system"""
    # Verbose mode and logging
    verbose_mode: bool = False
    log_directory: str = "logs"
    
    # Data paths
    data_path: str = "/qfs/projects/bioprep/data/automation/bl71.1_for_center/"
    output_dir: str = "output_videos_2"
    output_images_dir: str = "output_images_2"
    output_json_dir: str = "output_json_2"
    
    # Frame processing
    min_frame: int = 1450
    max_frame: int = 1850
    phi_min: float = 0.0
    phi_max: float = 360.0
    loop_count: int = 1
    
    # Circle detection parameters
    target_radius: int = 70
    radius_min: Optional[int] = None
    radius_max: Optional[int] = None
    radius_range: int = 5
    min_x_distance: int = 120
    min_y_distance: int = 120
    hough_num_peaks: int = 19
    hough_threshold: float = 0.2
    
    # Detection parameters
    total_wells_row1: int = 9
    total_wells_row2: int = 10
    total_wells: int = 19
    
    # Two-row configuration parameters
    row_y_tolerance: float = 35.0
    row_separation_min: float = 120.0
    initial_row_layout_flipped: bool = False
    
    # Tracking parameters
    ransac_residual_threshold: float = 5.0
    ransac_max_trials: int = 100
    association_distance_threshold: float = 50.0
    
    # Line fitting constraints
    min_circles_per_row: int = 2
    max_line_angle_degrees: float = 10.0
    
    # Line detection parameters
    line_hough_threshold: int = 80
    line_min_distance: int = 20
    line_min_angle: int = 80
    line_num_peaks: int = 4
    
    # Backup edge detection parameters
    backup_edge_sigma: float = 15.0
    backup_edge_low_threshold: float = 0.2
    backup_edge_high_threshold: float = 0.7
    
    # Edge detection parameters
    enable_edge_detection: bool = True
    edge_distance_multiplier: float = 2.0
    
    # Background removal parameters
    use_background_removal: bool = False
    rembg_model: Optional[str] = "birefnet-general-lite"
    
    # Output options
    video_fps: int = 10
    save_video: bool = True
    save_individual_frames: bool = True
    save_json_output: bool = True
    display_frames: bool = False
    
    # Feature flags
    enable_well_tracking: bool = True
    track_well_centers: bool = True
    enable_motor_calibration: bool = True
    use_phi_in_calibration: bool = False
    phi_calibration_weight: float = 1.0
    
    # Multi-frame calibration
    calibration_pairing_strategy: str = "random"
    calibration_use_multi_frame: bool = True
    calibration_use_average_movement: bool = True
    calibration_min_common_wells: int = 1
    
    # Motor calibration parameters
    calibration_min_samples: int = 10
    calibration_max_samples: Optional[int] = 100
    calibration_use_polynomial: bool = True
    calibration_alpha: float = 1.0
    
    # Well tracker parameters
    spacing_history_maxlen: int = 20
    
    # Advanced parameters
    min_edge_points: int = 0
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


@dataclass
class MotorPosition:
    """Motor position data"""
    x: float
    y: float
    z: float
    phi: float


@dataclass
class FrameObservation:
    """Single frame observation for calibration"""
    frame_number: int
    motor_position: np.ndarray
    pixel_positions: Dict[int, np.ndarray]
    average_pixel_position: np.ndarray
    detected_well_ids: Set[int]
    timestamp: float


@dataclass
class AverageMovementData:
    """Data for average movement between two frames"""
    frame1: int
    frame2: int
    motor_delta: np.ndarray
    pixel_delta: np.ndarray
    num_common_wells: int
    common_well_ids: Set[int]
    individual_deltas: List[np.ndarray]


class GeometryUtils:
    """Utility class for geometric operations"""
    
    @staticmethod
    def fit_line_ransac(points: np.ndarray, config: Config) -> Tuple[float, float, np.ndarray]:
        """Fit line using RANSAC"""
        if len(points) < 2:
            return 0, np.mean(points[:, 1]), np.ones(len(points), dtype=bool)
            
        X = points[:, 0].reshape(-1, 1)
        y = points[:, 1]
        
        if len(points) > 2:
            ransac = RANSACRegressor(
                random_state=42,
                min_samples=2,
                residual_threshold=None,
                max_trials=config.ransac_max_trials
            )
            ransac.fit(X, y)
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            inlier_mask = ransac.inlier_mask_
        else:
            slope = (y[1] - y[0]) / (X[1, 0] - X[0, 0]) if X[1, 0] != X[0, 0] else 0
            intercept = y[0] - slope * X[0, 0]
            inlier_mask = np.ones(len(points), dtype=bool)
        
        # Fallback to median fit if too many outliers
        if np.sum(inlier_mask) < len(points) * 0.5:
            slope, intercept = GeometryUtils.fit_line_median(X.flatten(), y)
            inlier_mask = np.ones(len(points), dtype=bool)
        
        return slope, intercept, inlier_mask
    
    @staticmethod
    def fit_line_median(x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[float, float]:
        """Robust median-based line fitting"""
        if len(x_coords) < 2:
            return 0, np.mean(y_coords)
            
        slopes = []
        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                if x_coords[j] != x_coords[i]:
                    slopes.append((y_coords[j] - y_coords[i]) / (x_coords[j] - x_coords[i]))
        
        if slopes:
            slope = np.median(slopes)
            intercepts = [y_coords[i] - slope * x_coords[i] for i in range(len(x_coords))]
            intercept = np.median(intercepts)
            return slope, intercept
        
        return 0, np.mean(y_coords)
    
    @staticmethod
    def is_line_horizontal(slope: float, max_angle_degrees: float) -> bool:
        """Check if a line is approximately horizontal"""
        max_angle_rad = np.deg2rad(max_angle_degrees)
        max_allowed_slope = np.tan(max_angle_rad)
        return abs(slope) <= max_allowed_slope
    
    @staticmethod
    def cluster_points_by_y(points: List[Tuple], y_tolerance: float, min_samples: int = 2) -> Dict[int, List[Tuple]]:
        """Cluster points by y-coordinate using DBSCAN"""
        if len(points) < min_samples:
            return {}
            
        y_coords = np.array([p[1] for p in points]).reshape(-1, 1)
        clustering = DBSCAN(eps=y_tolerance, min_samples=min_samples).fit(y_coords)
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label == -1:
                continue
            if label + 1 not in clusters:
                clusters[label + 1] = []
            clusters[label + 1].append(points[i])
        
        return clusters


class CircleDetector:
    """Handles circle detection using Hough transform"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect_circles(self, img: np.ndarray) -> Tuple:
        """Detect circles using Hough transform"""
        edge = canny(img, sigma=self.config.edge_sigma,
                    low_threshold=self.config.edge_low_threshold,
                    high_threshold=self.config.edge_high_threshold,
                    use_quantiles=True)
        
        radius_min, radius_max = self.config.get_radius_range()
        rads = np.arange(radius_min, radius_max + 1)
        
        out = hough_circle(edge, rads)
        
        accum, cx, cy, radii = hough_circle_peaks(
            out, rads,
            min_xdistance=self.config.min_x_distance,
            min_ydistance=self.config.min_y_distance,
            num_peaks=self.config.hough_num_peaks,
            threshold=self.config.hough_threshold
        )
        
        return edge, (accum, cx, cy, radii)


class LineDetector:
    """Handles line detection using Hough transform"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect_lines(self, contour_coords: Optional[np.ndarray], img_shape: Tuple, 
                    segments: Optional[List] = None,
                    backup_img: Optional[np.ndarray] = None) -> Tuple:
        """Detect lines using Hough transform"""
        threshold = self.config.line_hough_threshold
        min_distance = self.config.line_min_distance
        min_angle = self.config.line_min_angle
        num_peaks = self.config.line_num_peaks
        border_buffer = self.config.border_buffer
        
        primary_lines = []
        contour_img = None
        ph = None
        
        # Try hull/contour coordinates first
        if contour_coords is not None and len(contour_coords) > 0 and img_shape is not None:
            height, width = img_shape
            contour_img = np.zeros((height, width), dtype=np.uint8)
            
            if segments is not None:
                for segment in segments:
                    if len(segment) > 1:
                        for i in range(len(segment) - 1):
                            pt1 = tuple(segment[i].astype(int))
                            pt2 = tuple(segment[i + 1].astype(int))
                            cv2.line(contour_img, pt1, pt2, 255, thickness=2)
            else:
                for point in contour_coords:
                    cv2.circle(contour_img, tuple(point.astype(int)), 1, 255, -1)
            
            primary_lines = self._extract_lines_from_image(contour_img, threshold, 
                                                         min_distance, min_angle, num_peaks)
            if primary_lines:
                return contour_img, ph, primary_lines
        
        # Fallback to direct edge detection
        if backup_img is not None:
            edge = canny(backup_img, sigma=self.config.backup_edge_sigma, 
                        low_threshold=self.config.backup_edge_low_threshold, 
                        high_threshold=self.config.backup_edge_high_threshold, 
                        use_quantiles=True)
            
            backup_lines = self._extract_lines_from_image(edge, threshold, 
                                                        min_distance, min_angle, num_peaks)
            backup_lines = self._filter_border_lines(backup_lines, edge.shape, border_buffer)
            return edge, ph, backup_lines
        
        return None, None, []
    
    def _extract_lines_from_image(self, img: np.ndarray, threshold: int, 
                                min_distance: int, min_angle: int, num_peaks: int) -> List:
        """Extract lines from image using Hough transform"""
        angs = np.linspace(-np.pi/2, np.pi/2, 360, endpoint=False)
        h, theta, d = hough_line(img, angs)
        ph, pang, pdist = hough_line_peaks(h, theta, d, threshold=threshold, 
                                          min_distance=min_distance, min_angle=min_angle, 
                                          num_peaks=num_peaks)
        
        lines = []
        xline = np.arange(img.shape[1])
        for ang, dist in zip(pang, pdist):
            x0, y0 = dist * np.cos(ang), dist * np.sin(ang)
            if x0 == 0 or y0 == 0:
                continue
            m = y0 / x0
            m2 = -1 / m
            yline = m2 * (xline - x0) + y0
            sel = np.logical_and(yline > 0, yline < img.shape[0])
            if np.any(sel):
                lines.append((xline[sel], yline[sel]))
        
        return lines
    
    def _filter_border_lines(self, lines: List, shape: Tuple[int, int], border_buffer: int) -> List:
        """Filter out lines that run along image borders"""
        height, width = shape
        filtered_lines = []
        
        for line_data in lines:
            x_coords, y_coords, _, _ = line_data
            
            # Skip lines that run along borders
            if all(y <= border_buffer for y in y_coords):
                continue
            if all(y >= height - border_buffer for y in y_coords):
                continue
            if all(x <= border_buffer for x in x_coords):
                continue
            if all(x >= width - border_buffer for x in x_coords):
                continue
                
            filtered_lines.append(line_data)
        
        return filtered_lines


class ContourProcessor:
    """Handles contour extraction and processing"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def extract_contour_coordinates(self, edge_image: np.ndarray, min_area: Optional[int] = None, 
                                  remove_border_points: bool = True, 
                                  border_buffer: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """Extract and process contour coordinates from edge image"""
        if min_area is None:
            min_area = self.config.hull_min_area
        if border_buffer is None:
            border_buffer = self.config.border_buffer
            
        if edge_image.dtype != np.uint8:
            edge_uint8 = (edge_image * 255).astype(np.uint8)
        else:
            edge_uint8 = edge_image
        
        height, width = edge_image.shape
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed = cv2.morphologyEx(edge_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        processed = cv2.dilate(processed, kernel, iterations=2)
        
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
        hull = cv2.convexHull(all_points)
        hull_coords = hull.reshape(-1, 2)
        
        hull_area = cv2.contourArea(hull)
        if hull_area < min_area:
            return None, None
        
        if remove_border_points:
            mask = (
                (hull_coords[:, 0] >= border_buffer) &
                (hull_coords[:, 0] < width - border_buffer) &
                (hull_coords[:, 1] >= border_buffer) &
                (hull_coords[:, 1] < height - border_buffer)
            )
            
            kept_indices = np.where(mask)[0]
            filtered_hull_coords = hull_coords[mask]
            segments = self._create_segments_from_hull(hull_coords, kept_indices)
        else:
            filtered_hull_coords = hull_coords
            segments = [hull_coords]
        
        return filtered_hull_coords, segments
    
    def _create_segments_from_hull(self, hull_coords: np.ndarray, kept_indices: np.ndarray) -> List:
        """Create contiguous segments from hull coordinates"""
        segments = []
        if len(kept_indices) <= 1:
            return segments
        
        gaps = np.diff(kept_indices) > 1
        
        if not np.any(gaps):
            segments.append(hull_coords[kept_indices])
        else:
            gap_positions = np.where(gaps)[0] + 1
            start_idx = 0
            for gap_pos in gap_positions:
                if gap_pos > start_idx:
                    segment_indices = kept_indices[start_idx:gap_pos]
                    if len(segment_indices) > 1:
                        segments.append(hull_coords[segment_indices])
                start_idx = gap_pos
            
            if start_idx < len(kept_indices):
                segment_indices = kept_indices[start_idx:]
                if len(segment_indices) > 1:
                    segments.append(hull_coords[segment_indices])
        
        return segments


class WellCenterTracker:
    """Tracks when each well is closest to the frame center"""
    
    def __init__(self, frame_shape: Optional[Tuple[int, int]] = None, config: Optional[Config] = None):
        self.frame_shape = frame_shape
        self.frame_center = None
        self.config = config or Config()
        if frame_shape:
            self.set_frame_shape(frame_shape)
        
        self.best_positions: Dict[int, Dict] = {}
        self.all_positions: Dict[int, List] = {}
    
    def set_frame_shape(self, frame_shape: Tuple[int, int]):
        self.frame_shape = frame_shape
        height, width = frame_shape[:2]
        self.frame_center = (width / 2, height / 2)
    
    def update(self, frame_number: int, detected_wells: Dict, motor_data: MotorPosition,
               row_params: Dict[int, Tuple] = None, row_spacing: Dict[int, float] = None):
        """Update tracking with new frame data"""
        if not self.frame_center:
            return
        
        for well_id, well_info in detected_wells.items():
            if well_id is None:
                continue
            
            distance = calculate_distance(
                (well_info['x'], well_info['y']), self.frame_center
            )
            
            if well_id not in self.all_positions:
                self.all_positions[well_id] = []
            
            self.all_positions[well_id].append({
                'frame': frame_number,
                'distance': distance,
                'position': (well_info['x'], well_info['y']),
                'motor_data': asdict(motor_data)
            })
            
            if well_id not in self.best_positions or distance < self.best_positions[well_id]['distance']:
                self.best_positions[well_id] = {
                    'frame': frame_number,
                    'distance': distance,
                    'position': (well_info['x'], well_info['y']),
                    'motor_data': asdict(motor_data),
                    'radius': well_info.get('radius')
                }
    
    def _estimate_unseen_well_position(self, well_id: int, well_tracker: Optional['WellTracker']) -> Optional[Tuple[float, float]]:
        """Estimate pixel position for an unseen well using geometric layout"""
        if not well_tracker or not well_tracker.established_spacing:
            return None
        
        row, col = well_id_to_row_col(well_id, self.config)
        
        # Get reference wells from the same row
        row_wells = {}
        for wid, data in self.best_positions.items():
            if well_id_to_row_col(wid, self.config)[0] == row:
                row_wells[wid] = data
        
        if not row_wells:
            if row in well_tracker.row_params:
                slope, intercept = well_tracker.row_params[row]
                spacing = well_tracker.established_spacing
                
                if self.best_positions:
                    ref_well_id = min(self.best_positions.keys())
                    ref_data = self.best_positions[ref_well_id]
                    ref_x = ref_data['position'][0]
                    
                    ref_row, ref_col = well_id_to_row_col(ref_well_id, self.config)
                    if row == ref_row:
                        col_diff = col - ref_col
                    else:
                        if row == 1:
                            col_diff = col - 1
                        else:
                            col_diff = col - 1
                    
                    estimated_x = ref_x - col_diff * spacing
                    estimated_y = slope * estimated_x + intercept
                    
                    return (estimated_x, estimated_y)
            return None
        
        spacing = well_tracker.established_spacing
        
        ref_well_id = min(row_wells.keys(), key=lambda wid: abs(well_id_to_row_col(wid, self.config)[1] - col))
        ref_data = row_wells[ref_well_id]
        ref_col = well_id_to_row_col(ref_well_id, self.config)[1]
        
        col_diff = col - ref_col
        estimated_x = ref_data['position'][0] - col_diff * spacing
        
        if row in well_tracker.row_params:
            slope, intercept = well_tracker.row_params[row]
            estimated_y = slope * estimated_x + intercept
        else:
            estimated_y = ref_data['position'][1]
        
        return (estimated_x, estimated_y)
    
    def _get_reference_motor_position(self) -> MotorPosition:
        """Get a reference motor position for unseen wells"""
        if not self.best_positions:
            return MotorPosition(x=0.0, y=0.0, z=0.0, phi=0.0)
        
        central_well_id = min(self.best_positions.keys(), 
                             key=lambda wid: self.best_positions[wid]['distance'])
        motor_data = self.best_positions[central_well_id]['motor_data']
        
        return MotorPosition(
            x=motor_data['x'],
            y=motor_data['y'],
            z=motor_data['z'],
            phi=motor_data['phi']
        )
    
    def save_to_json(self, filename: Optional[str] = None, motor_calibration: Optional['InverseMotorCalibration'] = None, 
                     well_tracker: Optional['WellTracker'] = None) -> str:
        """Save predicted motor positions for centering all wells"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"well_centering_positions_{timestamp}.json"
        
        if not motor_calibration or not motor_calibration.is_calibrated:
            logger.warning("Motor Calibration Not Available Or Not Calibrated. Cannot Generate Centering Positions.")
            return ""
        
        if not self.frame_center:
            logger.warning("Frame Center Not Available. Cannot Generate Centering Positions.")
            return ""
        
        json_data = {
            'metadata': {
                'description': 'Predicted Motor Positions To Center Each Well In The Frame (All Wells - Seen And Unseen)',
                'frame_center': list(self.frame_center),
                'timestamp': datetime.now().isoformat(),
                'labeling_format': '(row, column)',
                'calibration_status': 'calibrated',
                'prediction_method': 'motor_calibration_with_geometric_estimation',
                'output_coordinates': 'x, y, z'
            },
            'well_centering_positions': {}
        }
        
        successful_predictions = 0
        failed_predictions = 0
        seen_wells = 0
        unseen_wells = 0
        
        for well_id in range(1, self.config.total_wells + 1):
            try:
                if well_id in self.best_positions:
                    data = self.best_positions[well_id]
                    current_motor = MotorPosition(
                        x=data['motor_data']['x'],
                        y=data['motor_data']['y'], 
                        z=data['motor_data']['z'],
                        phi=data['motor_data']['phi']
                    )
                    current_pixel_position = data['position']
                    seen_wells += 1
                    
                else:
                    unseen_wells += 1
                    
                    if (well_tracker and well_tracker.predicted_positions and 
                        well_id in well_tracker.predicted_positions):
                        pred = well_tracker.predicted_positions[well_id]
                        current_pixel_position = (pred['x'], pred['y'])
                    else:
                        current_pixel_position = self._estimate_unseen_well_position(well_id, well_tracker)
                    
                    if current_pixel_position is None:
                        failed_predictions += 1
                        logger.warning(f"Could Not Estimate Position For Unseen Well {format_well_label(well_id, self.config)}")
                        continue
                    
                    current_motor = self._get_reference_motor_position()
                
                predicted_motor = motor_calibration.estimate_motor_for_well_centering(
                    current_motor=current_motor,
                    well_pixel_position=current_pixel_position,
                    frame_center=self.frame_center
                )
                
                if predicted_motor is not None:
                    row, col = well_id_to_row_col(well_id, self.config)
                    well_key = f"({row},{col})"
                    
                    json_data['well_centering_positions'][well_key] = {
                        'well_id': int(well_id),
                        'row': row,
                        'column': col,
                        'status': 'observed' if well_id in self.best_positions else 'estimated',
                        'motor_position': {
                            'x': float(predicted_motor.x),
                            'y': float(predicted_motor.y),
                            'z': float(predicted_motor.z)
                        }
                    }
                    successful_predictions += 1
                else:
                    failed_predictions += 1
                    logger.warning(f"Could Not Predict Motor Position For Well {format_well_label(well_id, self.config)}")
                    
            except Exception as e:
                failed_predictions += 1
                logger.error(f"Error Predicting Motor Position For Well {well_id}: {e}")
        
        json_data['metadata']['total_wells'] = self.config.total_wells
        json_data['metadata']['successful_predictions'] = successful_predictions
        json_data['metadata']['failed_predictions'] = failed_predictions
        json_data['metadata']['wells_observed'] = seen_wells
        json_data['metadata']['wells_estimated'] = unseen_wells
        json_data['metadata']['estimation_details'] = {
            'geometric_estimation_used': unseen_wells > 0,
            'well_tracker_available': well_tracker is not None,
            'established_spacing_available': well_tracker is not None and well_tracker.established_spacing is not None,
            'row_parameters_available': well_tracker is not None and len(well_tracker.row_params) > 0
        }
        
        output_dir = Path(self.config.output_json_dir)
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Complete Motor Centering Positions Saved: {successful_predictions}/{self.config.total_wells} Wells")
        logger.info(f"  Observed Wells: {seen_wells}")
        logger.info(f"  Estimated Wells: {unseen_wells}")
        logger.info(f"  Failed Predictions: {failed_predictions}")
        
        return str(filepath)


class WellIdentifier:
    """Handles well identification logic for staggered layout"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def identify_well_using_stagger(self, x: float, y: float, row_id: int, 
                                   other_row_wells: Dict[int, Dict], spacing: float) -> Optional[int]:
        """Identify well ID using staggered layout relationship"""
        if not other_row_wells or not spacing:
            return None
        
        best_id = None
        min_error = float('inf')
        
        if row_id == 1:
            # Row 1 wells are between adjacent Row 2 wells
            for col in range(1, self.config.total_wells_row1 + 1):
                row1_well_id = col
                row2_right_id = self.config.total_wells_row1 + col
                row2_left_id = self.config.total_wells_row1 + col + 1
                
                row2_right = other_row_wells.get(row2_right_id)
                row2_left = other_row_wells.get(row2_left_id)
                
                expected_x = None
                
                if row2_right and row2_left:
                    expected_x = (row2_right['x'] + row2_left['x']) / 2
                elif row2_right:
                    expected_x = row2_right['x'] - spacing / 2
                elif row2_left:
                    expected_x = row2_left['x'] + spacing / 2
                
                if expected_x is not None:
                    error = abs(x - expected_x)
                    
                    if error < spacing * 0.5 and error < min_error:
                        min_error = error
                        best_id = row1_well_id
        
        else:  # row_id == 2
            for col in range(1, self.config.total_wells_row2 + 1):
                row2_well_id = self.config.total_wells_row1 + col
                expected_x = None
                
                if col == 1:
                    row1_well_1 = other_row_wells.get(1)
                    if row1_well_1:
                        expected_x = row1_well_1['x'] + spacing / 2
                    
                elif col == self.config.total_wells_row2:
                    row1_well_9 = other_row_wells.get(9)
                    if row1_well_9:
                        expected_x = row1_well_9['x'] - spacing / 2
                    
                else:
                    row1_right_col = col - 1
                    row1_left_col = col
                    
                    row1_right_well = other_row_wells.get(row1_right_col)
                    row1_left_well = other_row_wells.get(row1_left_col) if row1_left_col <= self.config.total_wells_row1 else None
                    
                    if row1_right_well and row1_left_well:
                        expected_x = (row1_right_well['x'] + row1_left_well['x']) / 2
                    elif row1_right_well:
                        expected_x = row1_right_well['x'] - spacing / 2
                    elif row1_left_well:
                        expected_x = row1_left_well['x'] + spacing / 2
                
                if expected_x is not None:
                    error = abs(x - expected_x)
                    if error < spacing * 0.5 and error < min_error:
                        min_error = error
                        best_id = row2_well_id
        
        return best_id
    
    def determine_well_id_from_spacing(self, x: float, y: float, row_id: int, 
                                     reference_wells: Dict[int, Dict], spacing: float) -> Optional[int]:
        """Determine well ID based on spacing from reference wells"""
        if not reference_wells or not spacing:
            return None
        
        row_refs = {wid: w for wid, w in reference_wells.items() if w.get('row') == row_id}
        
        if not row_refs:
            return None
        
        best_id = None
        min_error = float('inf')
        
        if row_id == 1:
            min_id, max_id = 1, self.config.total_wells_row1
        else:
            min_id = self.config.total_wells_row1 + 1
            max_id = self.config.total_wells
        
        for candidate_id in range(min_id, max_id + 1):
            expected_x = None
            
            if candidate_id in row_refs:
                ref_x = row_refs[candidate_id]['x']
                expected_x = ref_x
                error = abs(x - expected_x)
            else:
                for ref_id, ref_info in row_refs.items():
                    if row_id == 1:
                        id_diff = candidate_id - ref_id
                    else:
                        id_diff = (candidate_id - self.config.total_wells_row1) - (ref_id - self.config.total_wells_row1)
                    
                    estimated_x = ref_info['x'] - id_diff * spacing
                    
                    if expected_x is None:
                        expected_x = estimated_x
                        error = abs(x - expected_x)
                    else:
                        expected_x = (expected_x + estimated_x) / 2
                        error = abs(x - expected_x)
            
            if error < spacing * 0.5 and error < min_error:
                min_error = error
                best_id = candidate_id
        
        return best_id


class WellTracker:
    """Adaptive well tracking for two-row configuration using line fitting and spatial inference"""
    
    def __init__(self, config: Config):
        self.config = config
        self.geometry_utils = GeometryUtils()
        self.well_identifier = WellIdentifier(config)
        self.frame_number = 0
        
        self.row_params = {}
        self.row_spacing = {}
        
        self.detected_wells: Dict[int, Dict] = {}
        self.unassigned_detections: List[Tuple] = []
        
        self.last_successful_frame_wells: Dict[int, Dict] = {}
        self.last_successful_frame_number: Optional[int] = None
        
        self.average_radius = None
        self.spacing_history = {
            'row1': deque(maxlen=config.spacing_history_maxlen),
            'row2': deque(maxlen=config.spacing_history_maxlen)
        }
        
        self.predicted_positions = {}
        self.previous_detected_wells = {}
        self.original_hough_circles = None
        
        self.reference_frame_wells = {}
        self.reference_frame_number = None
        self.established_spacing = None
        
        # Edge detection for labeling initiation
        self.edge_condition_satisfied = False
        self.edge_detection_frame = None
        self.edge_circle_info = None
        
        # Row layout flipping based on phi changes
        self.last_perpendicular_phi = None
        self.row_layout_flipped = config.initial_row_layout_flipped
        self.phi_flip_history = []
    
    def get_row_params(self, row_id: int) -> Optional[Tuple]:
        """Get line parameters for a specific row"""
        return self.row_params.get(row_id)
    
    def get_row_spacing(self, row_id: int) -> Optional[float]:
        """Get current spacing for a specific row"""
        return self.row_spacing.get(row_id)
    
    def _check_and_handle_phi_flip(self, current_phi: float) -> bool:
        """Check if phi has changed by >90° and handle row flipping"""
        flip_occurred = False
        
        if self.last_perpendicular_phi is not None:
            phi_diff = calculate_angle_difference(current_phi, self.last_perpendicular_phi)
            
            if phi_diff > 90.0:
                self.row_layout_flipped = not self.row_layout_flipped
                flip_occurred = True
                
                flip_info = {
                    'frame': self.frame_number,
                    'previous_phi': self.last_perpendicular_phi,
                    'current_phi': current_phi,
                    'phi_difference': phi_diff,
                    'new_layout_flipped': self.row_layout_flipped
                }
                self.phi_flip_history.append(flip_info)
                
                logger.info(f"Row Layout Flip Detected At Frame {self.frame_number}")
                logger.info(f"  Previous Phi: {self.last_perpendicular_phi:.1f}°")
                logger.info(f"  Current Phi: {current_phi:.1f}°")
                logger.info(f"  Phi Difference: {phi_diff:.1f}°")
                logger.info(f"  Row Layout Now: {'Flipped' if self.row_layout_flipped else 'Normal'}")
                layout_desc = "Row 2 Top, Row 1 Bottom" if self.row_layout_flipped else "Row 1 Top, Row 2 Bottom"
                logger.info(f"  Current Layout: {layout_desc}")
                
                # Clear established knowledge since layout changed
                self.established_spacing = None
                self.reference_frame_wells = {}
                self.reference_frame_number = None
                self.last_successful_frame_wells = {}
                self.last_successful_frame_number = None
                logger.info("  Cleared Established Spacing And Reference Frame Due To Layout Flip")
        
        self.last_perpendicular_phi = current_phi
        
        return flip_occurred
    
    def _check_edge_condition(self, detections: List[Tuple], lines: List) -> Tuple[bool, Optional[Dict]]:
        """Check if any circle is at the edge of a row (near a non-horizontal line)"""
        if not self.config.enable_edge_detection or not lines or not detections:
            return False, None
        
        for det_x, det_y, det_r in detections:
            edge_distance_threshold = det_r * self.config.edge_distance_multiplier
            
            for xline, yline in lines:
                if len(xline) < 2 or len(yline) < 2:
                    continue
                    
                dx = xline[-1] - xline[0]
                dy = yline[-1] - yline[0]
                
                if abs(dx) > 1e-6:
                    slope = dy / dx
                    
                    if not self.geometry_utils.is_line_horizontal(slope, self.config.max_line_angle_degrees):
                        line_points = np.column_stack([xline, yline])
                        distances = cdist([(det_x, det_y)], line_points)[0]
                        min_distance = np.min(distances)
                        closest_idx = np.argmin(distances)
                        closest_line_point = (xline[closest_idx], yline[closest_idx])
                        
                        if min_distance <= edge_distance_threshold:
                            edge_info = {
                                'circle': {'x': det_x, 'y': det_y, 'radius': det_r},
                                'line_slope': slope,
                                'distance_to_line': min_distance,
                                'threshold_used': edge_distance_threshold,
                                'closest_line_point': closest_line_point,
                                'frame_number': self.frame_number
                            }
                            return True, edge_info
        
        return False, None
    
    def _detect_rows(self, detections: List[Tuple], current_phi: float = None) -> Dict[int, List[Tuple]]:
        """Detect and cluster detections into two rows with phi-based row flipping"""
        if len(detections) < 2:
            return {}
        
        if current_phi is not None:
            self._check_and_handle_phi_flip(current_phi)
        
        rows = self.geometry_utils.cluster_points_by_y(detections, self.config.row_y_tolerance, min_samples=2)
        
        # Handle noise points by assigning to closest row
        noise_points = []
        for detection in detections:
            found = False
            for row_detections in rows.values():
                if detection in row_detections:
                    found = True
                    break
            if not found:
                noise_points.append(detection)
        
        for detection in noise_points:
            min_dist = float('inf')
            best_row = None
            for row_id, row_detections in rows.items():
                if row_detections:
                    avg_y = np.mean([d[1] for d in row_detections])
                    dist = abs(detection[1] - avg_y)
                    if dist < self.config.row_y_tolerance and dist < min_dist:
                        min_dist = dist
                        best_row = row_id
            
            if best_row is not None:
                rows[best_row].append(detection)
            elif rows:
                new_row_id = max(rows.keys()) + 1
                rows[new_row_id] = [detection]
        
        if len(rows) < 2:
            return {}
        
        # Validate row separation
        if len(rows) == 2:
            row_ids = list(rows.keys())
            row1_avg_y = np.mean([d[1] for d in rows[row_ids[0]]])
            row2_avg_y = np.mean([d[1] for d in rows[row_ids[1]]])
            
            if abs(row1_avg_y - row2_avg_y) < self.config.row_separation_min:
                return {}
        
        # Merge rows until we have exactly 2
        while len(rows) > 2:
            row_ids = list(rows.keys())
            avg_ys = {rid: np.mean([d[1] for d in rows[rid]]) for rid in row_ids}
            
            min_dist = float('inf')
            merge_pair = None
            for i in range(len(row_ids)):
                for j in range(i + 1, len(row_ids)):
                    dist = abs(avg_ys[row_ids[i]] - avg_ys[row_ids[j]])
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (row_ids[i], row_ids[j])
            
            if merge_pair:
                rows[merge_pair[0]].extend(rows[merge_pair[1]])
                del rows[merge_pair[1]]
        
        # Identify rows by y-position with flipping logic
        if len(rows) == 2:
            row_ids = list(rows.keys())
            row1_dets = rows[row_ids[0]]
            row2_dets = rows[row_ids[1]]
            
            row1_y = np.mean([d[1] for d in row1_dets])
            row2_y = np.mean([d[1] for d in row2_dets])
            
            if not self.row_layout_flipped:
                if row1_y < row2_y:
                    return {1: row1_dets, 2: row2_dets}
                else:
                    return {1: row2_dets, 2: row1_dets}
            else:
                if row1_y < row2_y:
                    return {2: row1_dets, 1: row2_dets}
                else:
                    return {2: row2_dets, 1: row1_dets}
        
        return rows
    
    def _update_successful_frame_tracking(self):
        """Update tracking of the last successful frame (no unassigned wells)"""
        if len(self.unassigned_detections) == 0 and len(self.detected_wells) > 0:
            self.last_successful_frame_wells = self.detected_wells.copy()
            self.last_successful_frame_number = self.frame_number
            logger.debug(f"Updated Last Successful Frame To Frame {self.frame_number} With {len(self.detected_wells)} Wells")
    
    def _find_best_temporal_match(self, x: float, y: float, row_id: int) -> Optional[int]:
        """Find the best matching well ID from the last successful frame"""
        reference_wells = self.last_successful_frame_wells if self.last_successful_frame_wells else self.previous_detected_wells
        
        if not reference_wells:
            return None
        
        ref_row_wells = {well_id: well_info for well_id, well_info in reference_wells.items()
                        if well_info.get('row') == row_id}
        
        if not ref_row_wells:
            return None
        
        min_dist = float('inf')
        best_id = None
        
        for well_id, ref_info in ref_row_wells.items():
            dist = calculate_distance((x, y), (ref_info['x'], ref_info['y']))
            
            threshold = self.row_spacing.get(row_id, self.config.association_distance_threshold) * 0.5
            threshold = min(threshold, self.config.association_distance_threshold)
            
            if dist < min_dist and dist < threshold:
                min_dist = dist
                best_id = well_id
        
        return best_id
    
    def _reevaluate_all_assignments(self, rows: Dict[int, List[Tuple]]) -> bool:
        """Re-evaluate all assignments when unassigned wells are detected"""
        logger.debug(f"Re-Evaluating All Assignments At Frame {self.frame_number}")
        
        self.detected_wells = {}
        all_detections = []
        for row_id, row_detections in rows.items():
            for det in row_detections:
                all_detections.append((det[0], det[1], det[2], row_id))
        
        all_detections.sort(key=lambda d: d[0], reverse=True)
        
        reference_wells = {}
        if self.last_successful_frame_wells:
            reference_wells = self.last_successful_frame_wells
            logger.debug(f"  Using Last Successful Frame {self.last_successful_frame_number} As Reference")
        elif self.reference_frame_wells:
            reference_wells = self.reference_frame_wells
            logger.debug(f"  Using Reference Frame {self.reference_frame_number} As Reference")
        elif self.previous_detected_wells:
            reference_wells = self.previous_detected_wells
            logger.debug("  Using Previous Frame As Reference")
        
        spacing = self.established_spacing or self.row_spacing.get(1) or self.row_spacing.get(2)
        
        assignments_made = 0
        
        for det_x, det_y, det_r, row_id in all_detections:
            best_id = None
            min_cost = float('inf')
            
            other_row_id = 2 if row_id == 1 else 1
            other_row_wells = {wid: info for wid, info in self.detected_wells.items()
                             if info.get('row') == other_row_id}
            
            # Method 1: stagger relationship with current assignments
            if other_row_wells and spacing:
                stagger_id = self.well_identifier.identify_well_using_stagger(
                    det_x, det_y, row_id, other_row_wells, spacing)
                if stagger_id and stagger_id not in self.detected_wells:
                    cost = 0.1
                    if cost < min_cost:
                        min_cost = cost
                        best_id = stagger_id
            
            # Method 2: reference frame matching
            if reference_wells and spacing:
                ref_id = self.well_identifier.determine_well_id_from_spacing(
                    det_x, det_y, row_id, reference_wells, spacing)
                if ref_id and ref_id not in self.detected_wells:
                    if ref_id in reference_wells:
                        ref_pos = (reference_wells[ref_id]['x'], reference_wells[ref_id]['y'])
                        cost = calculate_distance((det_x, det_y), ref_pos) / 100.0
                    else:
                        cost = 0.5
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_id = ref_id
            
            # Method 3: spatial consistency within current assignments
            if spacing and self.detected_wells:
                row_wells = {wid: info for wid, info in self.detected_wells.items()
                           if info.get('row') == row_id}
                
                if row_wells:
                    spatial_id = self._determine_id_from_spatial_layout(det_x, det_y, row_id, row_wells, spacing)
                    if spatial_id and spatial_id not in self.detected_wells:
                        cost = 0.3
                        if cost < min_cost:
                            min_cost = cost
                            best_id = spatial_id
            
            # Method 4: sequential assignment as fallback
            if best_id is None:
                if row_id == 1:
                    for col in range(1, self.config.total_wells_row1 + 1):
                        if col not in self.detected_wells:
                            best_id = col
                            break
                else:
                    for col in range(1, self.config.total_wells_row2 + 1):
                        well_id = self.config.total_wells_row1 + col
                        if well_id not in self.detected_wells:
                            best_id = well_id
                            break
            
            if best_id is not None:
                self.detected_wells[best_id] = {
                    'x': det_x, 'y': det_y, 'radius': det_r, 'row': row_id
                }
                assignments_made += 1
            else:
                self.unassigned_detections.append((det_x, det_y, det_r))
        
        logger.debug(f"  Re-evaluation Complete: {assignments_made} Wells Assigned, {len(self.unassigned_detections)} Unassigned")
        
        return len(self.unassigned_detections) == 0
    
    def _determine_id_from_spatial_layout(self, x: float, y: float, row_id: int, 
                                        row_wells: Dict[int, Dict], spacing: float) -> Optional[int]:
        """Determine well ID based on spatial layout within the row"""
        if not row_wells or not spacing:
            return None
        
        best_id = None
        min_error = float('inf')
        
        if row_id == 1:
            min_id, max_id = 1, self.config.total_wells_row1
        else:
            min_id = self.config.total_wells_row1 + 1
            max_id = self.config.total_wells
        
        for candidate_id in range(min_id, max_id + 1):
            if candidate_id in self.detected_wells:
                continue
            
            expected_x = None
            total_error = 0
            reference_count = 0
            
            for ref_id, ref_info in row_wells.items():
                if row_id == 1:
                    id_diff = candidate_id - ref_id
                else:
                    id_diff = (candidate_id - self.config.total_wells_row1) - (ref_id - self.config.total_wells_row1)
                
                estimated_x = ref_info['x'] - id_diff * spacing
                error = abs(x - estimated_x)
                total_error += error
                reference_count += 1
            
            if reference_count > 0:
                avg_error = total_error / reference_count
                if avg_error < spacing * 0.5 and avg_error < min_error:
                    min_error = avg_error
                    best_id = candidate_id
        
        return best_id
    
    def _process_single_row(self, row_id: int, row_detections: List[Tuple]):
        """Process detections for a single row"""
        sorted_detections = sorted(row_detections, key=lambda d: d[0], reverse=True)
        
        other_row_id = 2 if row_id == 1 else 1
        other_row_wells = {wid: info for wid, info in self.detected_wells.items()
                          if info.get('row') == other_row_id}
        
        for det_x, det_y, det_r in sorted_detections:
            best_id = None
            spacing = self.established_spacing or self.row_spacing.get(row_id)

            # Method 1: Temporal matching with previous frames
            best_id = self._find_best_temporal_match(det_x, det_y, row_id)

            # Method 2: Spacing-based matching with reference wells
            if best_id is None and spacing:
                reference_wells = self.last_successful_frame_wells or self.reference_frame_wells or self.previous_detected_wells
                best_id = self.well_identifier.determine_well_id_from_spacing(
                    det_x, det_y, row_id, reference_wells, spacing)
            
            # Method 3: Stagger relationship matching with other row
            if best_id is None and other_row_wells and spacing:
                best_id = self.well_identifier.identify_well_using_stagger(det_x, det_y, row_id, other_row_wells, spacing)
            
            # Method 4: Spatial consistency within current frame
            if best_id is None and self.detected_wells:
                best_id = self._determine_id_from_current_frame(det_x, det_y, row_id)
            
            # Method 5: Initial assignment fallback
            if best_id is None and not self.reference_frame_wells and not self.previous_detected_wells and not self.last_successful_frame_wells:
                best_id = self._assign_initial_id(det_x, det_y, row_id, sorted_detections)
            
            # Assign the well or mark as unassigned
            if best_id is not None and best_id not in self.detected_wells:
                self.detected_wells[best_id] = {
                    'x': det_x, 'y': det_y, 'radius': det_r, 'row': row_id
                }
            elif best_id is None:
                self.unassigned_detections.append((det_x, det_y, det_r))
    
    def _determine_id_from_current_frame(self, x: float, y: float, row_id: int) -> Optional[int]:
        """Determine well ID based on already detected wells in current frame"""
        row_wells = {wid: info for wid, info in self.detected_wells.items()
                    if info.get('row') == row_id}
        
        if not row_wells or not self.established_spacing:
            return None
        
        best_id = None
        min_error = float('inf')
        
        if row_id == 1:
            min_id, max_id = 1, self.config.total_wells_row1
        else:
            min_id = self.config.total_wells_row1 + 1
            max_id = self.config.total_wells
        
        for candidate_id in range(min_id, max_id + 1):
            if candidate_id in self.detected_wells:
                continue
            
            expected_x = None
            for ref_id, ref_info in row_wells.items():
                if row_id == 1:
                    id_diff = candidate_id - ref_id
                else:
                    id_diff = (candidate_id - self.config.total_wells_row1) - (ref_id - self.config.total_wells_row1)
                
                estimated_x = ref_info['x'] - id_diff * self.established_spacing
                
                if expected_x is None:
                    expected_x = estimated_x
                else:
                    expected_x = (expected_x + estimated_x) / 2
            
            if expected_x is not None:
                error = abs(x - expected_x)
                if error < self.established_spacing * 0.5 and error < min_error:
                    min_error = error
                    best_id = candidate_id
        
        return best_id
    
    def _assign_initial_id(self, x: float, y: float, row_id: int, 
                          sorted_detections: List[Tuple]) -> Optional[int]:
        """Assign initial ID based on position"""
        for idx, (det_x, det_y, _) in enumerate(sorted_detections):
            if abs(det_x - x) < 1e-6 and abs(det_y - y) < 1e-6:
                if row_id == 1:
                    if idx < self.config.total_wells_row1:
                        return idx + 1
                else:
                    if idx < self.config.total_wells_row2:
                        return self.config.total_wells_row1 + idx + 1
                break
        return None
    
    def _process_two_rows(self, rows: Dict[int, List[Tuple]]):
        """Process detections as two rows with temporal consistency"""
        self.detected_wells = {}
        self.predicted_positions = {}
        self.unassigned_detections = []
        
        # Update line parameters and spacing for each row
        for row_id, row_detections in rows.items():
            if len(row_detections) >= self.config.min_circles_per_row:
                points = np.array(row_detections)
                
                if len(row_detections) > 2:
                    slope, intercept, _ = self.geometry_utils.fit_line_ransac(points, self.config)
                else:
                    slope = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0]) if points[1, 0] != points[0, 0] else 0
                    intercept = points[0, 1] - slope * points[0, 0]
                
                if self.geometry_utils.is_line_horizontal(slope, self.config.max_line_angle_degrees):
                    self.row_params[row_id] = (slope, intercept)
                    
                    positions = sorted([d[0] for d in row_detections])
                    if len(positions) >= 2:
                        spacings = np.diff(positions)
                        self.row_spacing[row_id] = np.median(np.abs(spacings))
                        self.spacing_history[f'row{row_id}'].append(self.row_spacing[row_id])
                        
                        if self.established_spacing is None:
                            self.established_spacing = self.row_spacing[row_id]
                            
            elif len(row_detections) == 1 and self.established_spacing and row_id in self.row_params:
                det_x, det_y, _ = row_detections[0]
                old_slope, old_intercept = self.row_params[row_id]
                
                new_intercept = det_y - old_slope * det_x
                self.row_params[row_id] = (old_slope, new_intercept)
                
                if self.established_spacing:
                    self.row_spacing[row_id] = self.established_spacing

        # Process each row sequentially
        for row_id, row_detections in rows.items():
            self._process_single_row(row_id, row_detections)
        
        # Generate predictions if conditions are met
        should_generate_predictions = False
        
        if self.detected_wells:
            if self.established_spacing and any(row_id in self.row_params for row_id in [1, 2]):
                should_generate_predictions = True
            elif len(self.detected_wells) >= 2:
                should_generate_predictions = True
        
        if should_generate_predictions:
            self._generate_predictions_for_missing_wells()
    
    def _generate_predictions_for_missing_wells(self):
        """Generate predicted positions for missing wells"""
        self.predicted_positions = {}
        
        if not self.detected_wells or not self.established_spacing:
            return
        
        has_valid_row = False
        for row_id in [1, 2]:
            if (row_id in self.row_params and 
                sum(1 for w in self.detected_wells.values() if w.get('row') == row_id) >= 1):
                has_valid_row = True
                break
        
        if not has_valid_row:
            return
        
        all_expected_wells = set(range(1, self.config.total_wells + 1))
        currently_detected = set(self.detected_wells.keys())
        missing_wells = all_expected_wells - currently_detected
        
        for well_id in missing_wells:
            if well_id <= self.config.total_wells_row1:
                row_id = 1
            else:
                row_id = 2
            
            if row_id in self.row_params:
                slope, intercept = self.row_params[row_id]
                
                row_wells = {wid: info for wid, info in self.detected_wells.items()
                           if info.get('row') == row_id}
                
                if row_wells:
                    closest_id = min(row_wells.keys(), 
                                   key=lambda k: abs(k - well_id))
                    anchor_info = row_wells[closest_id]
                    
                    if row_id == 1:
                        id_offset = well_id - closest_id
                    else:
                        id_offset = (well_id - self.config.total_wells_row1) - (closest_id - self.config.total_wells_row1)
                    
                    predicted_x = anchor_info['x'] - id_offset * self.established_spacing
                    predicted_y = slope * predicted_x + intercept
                    
                    self.predicted_positions[well_id] = {
                        'x': predicted_x,
                        'y': predicted_y,
                        'radius': self.average_radius or 100,
                        'row': row_id
                    }
    
    def _establish_reference_frame(self):
        """Establish a reference frame for consistent labeling"""
        if len(self.detected_wells) >= 10:
            self.reference_frame_wells = self.detected_wells.copy()
            self.reference_frame_number = self.frame_number
            
            all_spacings = []
            for row_id in [1, 2]:
                row_wells = [(wid, w) for wid, w in self.detected_wells.items() 
                            if w.get('row') == row_id]
                if len(row_wells) >= 2:
                    row_wells.sort(key=lambda x: x[1]['x'])
                    for i in range(len(row_wells) - 1):
                        spacing = abs(row_wells[i+1][1]['x'] - row_wells[i][1]['x'])
                        all_spacings.append(spacing)
            
            if all_spacings:
                self.established_spacing = np.median(all_spacings)
    
    def update_tracks(self, detected_circles: Optional[Tuple], lines: Optional[List] = None, 
                     current_phi: Optional[float] = None) -> Tuple[Optional[Tuple], Optional[List]]:
        """Update tracking with new detections"""
        self.frame_number += 1
        self.unassigned_detections = []
        
        self.original_hough_circles = detected_circles
        self.previous_detected_wells = self.detected_wells.copy()
        
        if not detected_circles or len(detected_circles[1]) == 0:
            self.detected_wells = {}
            return None, []
        
        accum, cx, cy, radii = detected_circles
        detections = [(cx[i], cy[i], radii[i]) for i in range(len(cx))]
        
        if len(radii) > 0:
            self.average_radius = np.mean(radii)
        
        # Check edge condition if not already satisfied
        if not self.edge_condition_satisfied and self.config.enable_edge_detection:
            edge_detected, edge_info = self._check_edge_condition(detections, lines or [])
            
            if edge_detected:
                self.edge_condition_satisfied = True
                self.edge_detection_frame = self.frame_number
                self.edge_circle_info = edge_info
                logger.info(f"Edge Condition Satisfied At Frame {self.frame_number}")
                logger.info(f"Edge Circle At ({edge_info['circle']['x']:.1f}, {edge_info['circle']['y']:.1f}) "
                      f"With Radius {edge_info['circle']['radius']:.1f}")
                logger.info(f"Distance To Non-Horizontal Line (Slope={edge_info['line_slope']:.3f}): "
                      f"{edge_info['distance_to_line']:.1f} Pixels")
        
        # Only proceed with labeling if edge condition is satisfied (or disabled)
        if not self.config.enable_edge_detection or self.edge_condition_satisfied:
            rows = self._detect_rows(detections, current_phi)
            
            if len(rows) != 2:
                self.unassigned_detections = detections
                self.detected_wells = {}
                self.predicted_positions = {}
                return self._get_tracked_wells_as_circles(), self.get_well_ids()
            
            self._process_two_rows(rows)
            
            # Check if we have unassigned wells after initial processing
            if len(self.unassigned_detections) > 0 and len(self.detected_wells) > 0:
                logger.debug(f"Frame {self.frame_number}: {len(self.unassigned_detections)} Unassigned Wells Detected, Triggering Re-Evaluation")
                success = self._reevaluate_all_assignments(rows)
                if success:
                    logger.debug("Re-Evaluation Successful - All Wells Now Assigned")
            
            self._update_successful_frame_tracking()
            
            if self.reference_frame_number is None and len(self.detected_wells) >= 10:
                self._establish_reference_frame()
        else:
            self.unassigned_detections = detections
            self.detected_wells = {}
            self.predicted_positions = {}
        
        return self._get_tracked_wells_as_circles(), self.get_well_ids()
    
    def _get_tracked_wells_as_circles(self) -> Optional[Tuple]:
        """Convert tracked wells to circle format"""
        if not self.detected_wells and not self.unassigned_detections:
            return None
        
        all_detections = []
        
        for well_id, well_data in sorted(self.detected_wells.items()):
            all_detections.append({
                'x': well_data['x'],
                'y': well_data['y'],
                'radius': well_data['radius'],
                'well_id': well_id
            })
        
        for x, y, r in self.unassigned_detections:
            all_detections.append({
                'x': x, 'y': y, 'radius': r,
                'well_id': None
            })
        
        if not all_detections:
            return None
        
        return (np.ones(len(all_detections)),
                np.array([d['x'] for d in all_detections]),
                np.array([d['y'] for d in all_detections]),
                np.array([d['radius'] for d in all_detections]))
    
    def get_well_ids(self) -> List[Optional[int]]:
        """Get list of currently tracked well IDs"""
        ids = sorted(self.detected_wells.keys())
        ids.extend([None] * len(self.unassigned_detections))
        return ids
    
    def get_all_predicted_positions(self) -> Optional[Dict]:
        """Get predicted positions for missing wells"""
        return self.predicted_positions if self.predicted_positions else None
    
    def get_line_endpoints(self) -> Optional[List[Tuple]]:
        """Get endpoints of fitted lines for visualization"""
        endpoints = []
        
        if not self.detected_wells:
            return None
        
        # Row 1 endpoints
        if 1 in self.row_params and 1 in self.row_spacing:
            slope, intercept = self.row_params[1]
            spacing = self.row_spacing[1]
            
            row1_wells = [w for w_id, w in self.detected_wells.items() 
                         if w_id <= self.config.total_wells_row1 and w.get('row') == 1]
            if row1_wells:
                min_x = min(w['x'] for w in row1_wells) - spacing
                max_x = max(w['x'] for w in row1_wells) + spacing
                
                y1 = slope * min_x + intercept
                y2 = slope * max_x + intercept
                endpoints.append(((min_x, y1), (max_x, y2)))
        
        # Row 2 endpoints
        if 2 in self.row_params and 2 in self.row_spacing:
            slope, intercept = self.row_params[2]
            spacing = self.row_spacing[2]
            
            row2_wells = [w for w_id, w in self.detected_wells.items() 
                         if w_id > self.config.total_wells_row1 and w.get('row') == 2]
            if row2_wells:
                min_x = min(w['x'] for w in row2_wells) - spacing
                max_x = max(w['x'] for w in row2_wells) + spacing
                
                y1 = slope * min_x + intercept
                y2 = slope * max_x + intercept
                endpoints.append(((min_x, y1), (max_x, y2)))
        
        return endpoints if endpoints else None
    
    def get_edge_detection_status(self) -> Dict:
        """Get current edge detection status including row flipping information"""
        return {
            'edge_condition_satisfied': self.edge_condition_satisfied,
            'edge_detection_frame': self.edge_detection_frame,
            'edge_circle_info': self.edge_circle_info,
            'current_frame': self.frame_number,
            'row_layout_flipped': self.row_layout_flipped,
            'last_perpendicular_phi': self.last_perpendicular_phi,
            'phi_flip_history': self.phi_flip_history,
            'last_successful_frame': self.last_successful_frame_number,
            'num_successful_frame_wells': len(self.last_successful_frame_wells)
        }


class InverseMotorCalibration:
    """Learns inverse transformation from pixel coordinates to motor coordinates"""
    
    def __init__(self, config: Config):
        self.config = config
        
        self.min_samples = config.calibration_min_samples
        self.max_samples = config.calibration_max_samples
        self.use_polynomial = config.calibration_use_polynomial
        self.alpha = config.calibration_alpha
        self.use_phi = config.use_phi_in_calibration
        self.phi_weight = config.phi_calibration_weight
        
        self.frame_observations: Dict[int, FrameObservation] = {}
        self.frame_count = 0
        
        # Training data storage
        maxlen = self.max_samples if self.max_samples is not None else None
        self.motor_history = deque(maxlen=maxlen)
        self.pixel_history = deque(maxlen=maxlen)
        self.frame_pair_history = deque(maxlen=maxlen)
        self.common_wells_history = deque(maxlen=maxlen)
        
        # Models
        self.model_motor_x = Ridge(alpha=self.alpha)
        self.model_motor_y = Ridge(alpha=self.alpha)
        self.model_motor_z = Ridge(alpha=self.alpha)
        if self.use_phi:
            self.model_motor_phi = Ridge(alpha=self.alpha)
        
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False) if self.use_polynomial else None
        
        self.is_calibrated = False
        self.calibration_scores = {}
        self.last_prediction_error = {}
        
        # Training statistics
        self.training_stats = {
            'total_pairs_generated': 0,
            'unique_frame_pairs': set(),
            'average_frame_gap': 0,
            'min_frame_gap': float('inf'),
            'max_frame_gap': 0,
            'average_common_wells': 0,
            'min_common_wells': float('inf'),
            'max_common_wells': 0,
            'pixel_delta_std': {'x': [], 'y': []}
        }
    
    def add_observation(self, motor_data: MotorPosition, detected_wells: Dict[int, Dict],
                       frame_number: Optional[int] = None):
        """Add a new observation of motor positions and corresponding pixel positions"""
        if not detected_wells:
            return
        
        if frame_number is None:
            frame_number = self.frame_count
        self.frame_count += 1
        
        if self.use_phi:
            motor_array = np.array([motor_data.x, motor_data.y, motor_data.z, motor_data.phi])
        else:
            motor_array = np.array([motor_data.x, motor_data.y, motor_data.z])
        
        pixel_positions = {}
        pixel_arrays = []
        for well_id, well_info in detected_wells.items():
            pixel_pos = np.array([well_info['x'], well_info['y']])
            pixel_positions[well_id] = pixel_pos
            pixel_arrays.append(pixel_pos)
        
        if pixel_arrays:
            average_pixel_position = np.mean(pixel_arrays, axis=0)
        else:
            average_pixel_position = np.array([0.0, 0.0])
        
        observation = FrameObservation(
            frame_number=frame_number,
            motor_position=motor_array.copy(),
            pixel_positions=pixel_positions.copy(),
            average_pixel_position=average_pixel_position,
            detected_well_ids=set(detected_wells.keys()),
            timestamp=datetime.now().timestamp()
        )
        
        self.frame_observations[frame_number] = observation
        
        if self.config.calibration_use_average_movement:
            self._generate_training_pairs_averaged()
        else:
            self._generate_training_pairs_individual()
        
        if len(self.motor_history) >= self.min_samples:
            self._train_models()
    
    def _generate_training_pairs_averaged(self):
        """Generate training pairs using average movement across all wells"""
        if len(self.frame_observations) < 2:
            return
        
        all_frames = sorted(self.frame_observations.keys())
        current_frame = all_frames[-1]
        
        if self.config.calibration_use_multi_frame:
            pairs_to_process = []
            for other_frame in all_frames[:-1]:
                pairs_to_process.append((other_frame, current_frame))
        else:
            pairs_to_process = []
            if len(all_frames) >= 2:
                prev_frame = all_frames[-2]
                if current_frame - prev_frame == 1:
                    pairs_to_process = [(prev_frame, current_frame)]
        
        for frame1, frame2 in pairs_to_process:
            if frame1 not in self.frame_observations or frame2 not in self.frame_observations:
                continue
            
            obs1 = self.frame_observations[frame1]
            obs2 = self.frame_observations[frame2]
            
            common_wells = obs1.detected_well_ids & obs2.detected_well_ids
            
            if len(common_wells) < self.config.calibration_min_common_wells:
                continue
            
            pixel_deltas = []
            for well_id in common_wells:
                delta = obs2.pixel_positions[well_id] - obs1.pixel_positions[well_id]
                pixel_deltas.append(delta)
            
            avg_pixel_delta = np.mean(pixel_deltas, axis=0)
            std_pixel_delta = np.std(pixel_deltas, axis=0) if len(pixel_deltas) > 1 else np.array([0, 0])
            
            motor_delta = obs2.motor_position - obs1.motor_position
            
            if self.use_phi and self.phi_weight != 1.0:
                motor_delta_weighted = motor_delta.copy()
                motor_delta_weighted[3] *= self.phi_weight
            else:
                motor_delta_weighted = motor_delta
            
            if np.linalg.norm(motor_delta_weighted) > 1e-6:
                self.motor_history.append(motor_delta)
                self.pixel_history.append(avg_pixel_delta)
                self.frame_pair_history.append((frame1, frame2))
                self.common_wells_history.append(len(common_wells))
                
                self._update_training_statistics(frame1, frame2, len(common_wells), std_pixel_delta)
    
    def _generate_training_pairs_individual(self):
        """Generate training pairs from individual well movements"""
        if len(self.frame_observations) < 2:
            return
        
        all_frames = sorted(self.frame_observations.keys())
        
        for i in range(len(all_frames) - 1):
            frame1, frame2 = all_frames[i], all_frames[i + 1]
            
            if abs(frame2 - frame1) != 1:
                continue
            
            obs1 = self.frame_observations[frame1]
            obs2 = self.frame_observations[frame2]
            
            common_wells = obs1.detected_well_ids & obs2.detected_well_ids
            
            for well_id in common_wells:
                pixel_delta = obs2.pixel_positions[well_id] - obs1.pixel_positions[well_id]
                motor_delta = obs2.motor_position - obs1.motor_position
                
                if self.use_phi and self.phi_weight != 1.0:
                    motor_delta_weighted = motor_delta.copy()
                    motor_delta_weighted[3] *= self.phi_weight
                else:
                    motor_delta_weighted = motor_delta
                
                if np.linalg.norm(motor_delta_weighted) > 1e-6:
                    self.motor_history.append(motor_delta)
                    self.pixel_history.append(pixel_delta)
                    self.frame_pair_history.append((frame1, frame2))
                    self.common_wells_history.append(1)
                    
                    self._update_training_statistics(frame1, frame2, 1, np.array([0, 0]))
    
    def _update_training_statistics(self, frame1: int, frame2: int, 
                                   num_common_wells: int, std_pixel_delta: np.ndarray):
        """Update training statistics"""
        frame_gap = abs(frame2 - frame1)
        
        self.training_stats['total_pairs_generated'] += 1
        self.training_stats['unique_frame_pairs'].add((frame1, frame2))
        
        self.training_stats['min_frame_gap'] = min(self.training_stats['min_frame_gap'], frame_gap)
        self.training_stats['max_frame_gap'] = max(self.training_stats['max_frame_gap'], frame_gap)
        
        self.training_stats['min_common_wells'] = min(self.training_stats['min_common_wells'], num_common_wells)
        self.training_stats['max_common_wells'] = max(self.training_stats['max_common_wells'], num_common_wells)
        
        self.training_stats['pixel_delta_std']['x'].append(std_pixel_delta[0])
        self.training_stats['pixel_delta_std']['y'].append(std_pixel_delta[1])
        
        if self.frame_pair_history:
            gaps = [abs(p[1] - p[0]) for p in self.frame_pair_history]
            self.training_stats['average_frame_gap'] = np.mean(gaps)
        
        if self.common_wells_history:
            self.training_stats['average_common_wells'] = np.mean(list(self.common_wells_history))
    
    def _train_models(self):
        """Train the inverse regression models"""
        if len(self.pixel_history) < self.min_samples:
            return
        
        X = np.array(self.pixel_history)
        motor_deltas = np.array(self.motor_history)
        
        if self.use_polynomial and self.poly_features:
            X = self.poly_features.fit_transform(X)
        
        self.model_motor_x.fit(X, motor_deltas[:, 0])
        self.model_motor_y.fit(X, motor_deltas[:, 1])
        self.model_motor_z.fit(X, motor_deltas[:, 2])
        
        self.calibration_scores['motor_x'] = self.model_motor_x.score(X, motor_deltas[:, 0])
        self.calibration_scores['motor_y'] = self.model_motor_y.score(X, motor_deltas[:, 1])
        self.calibration_scores['motor_z'] = self.model_motor_z.score(X, motor_deltas[:, 2])
        
        if self.use_phi:
            phi_deltas = motor_deltas[:, 3]
            if self.phi_weight != 1.0:
                phi_deltas = phi_deltas / self.phi_weight
            
            self.model_motor_phi.fit(X, phi_deltas)
            self.calibration_scores['motor_phi'] = self.model_motor_phi.score(X, phi_deltas)
        
        self.is_calibrated = True
    
    def predict_motor_shifts(self, pixel_delta: np.ndarray) -> Optional[np.ndarray]:
        """Predict motor shifts based on pixel movement"""
        if not self.is_calibrated:
            return None
        
        if len(pixel_delta) != 2:
            raise ValueError("pixel_delta must be 2D (dx, dy)")
        
        X = pixel_delta.reshape(1, -1)
        
        if self.use_polynomial and self.poly_features:
            X = self.poly_features.transform(X)
        
        motor_dx = self.model_motor_x.predict(X)[0]
        motor_dy = self.model_motor_y.predict(X)[0]
        motor_dz = self.model_motor_z.predict(X)[0]
        
        if self.use_phi:
            motor_dphi = self.model_motor_phi.predict(X)[0]
            if self.phi_weight != 1.0:
                motor_dphi *= self.phi_weight
            return np.array([motor_dx, motor_dy, motor_dz, motor_dphi])
        else:
            return np.array([motor_dx, motor_dy, motor_dz])
    
    def predict_motor_position_for_pixel_shift(self, current_motor: MotorPosition,
                                              current_pixel: Tuple[float, float],
                                              target_pixel: Tuple[float, float]) -> Optional[MotorPosition]:
        """Predict motor position needed to achieve a target pixel position"""
        if not self.is_calibrated:
            return None
        
        pixel_delta = np.array([
            target_pixel[0] - current_pixel[0],
            target_pixel[1] - current_pixel[1]
        ])
        
        motor_shifts = self.predict_motor_shifts(pixel_delta)
        if motor_shifts is None:
            return None
        
        if self.use_phi:
            return MotorPosition(
                x=current_motor.x + motor_shifts[0],
                y=current_motor.y + motor_shifts[1],
                z=current_motor.z + motor_shifts[2],
                phi=current_motor.phi + motor_shifts[3]
            )
        else:
            return MotorPosition(
                x=current_motor.x + motor_shifts[0],
                y=current_motor.y + motor_shifts[1],
                z=current_motor.z + motor_shifts[2],
                phi=current_motor.phi
            )
    
    def estimate_motor_for_well_centering(self, current_motor: MotorPosition,
                                         well_pixel_position: Tuple[float, float],
                                         frame_center: Tuple[float, float]) -> Optional[MotorPosition]:
        """Estimate motor position to center a well in the frame"""
        return self.predict_motor_position_for_pixel_shift(
            current_motor, well_pixel_position, frame_center
        )
    
    def get_calibration_info(self) -> Dict:
        """Get calibration status and quality metrics"""
        if not self.is_calibrated:
            return {
                'is_calibrated': False,
                'samples_collected': len(self.motor_history),
                'samples_needed': self.min_samples,
                'max_samples': self.max_samples if self.max_samples else 'Unlimited',
                'uses_phi': self.use_phi,
                'phi_weight': self.phi_weight if self.use_phi else None,
                'mapping_direction': 'Pixel -> Motor (Inverse)',
                'averaging_enabled': self.config.calibration_use_average_movement,
                'multi_frame_enabled': self.config.calibration_use_multi_frame,
                'pairing_strategy': self.config.calibration_pairing_strategy,
                'total_frames': len(self.frame_observations)
            }
        
        avg_score = np.mean(list(self.calibration_scores.values()))
        avg_error = np.mean(list(self.last_prediction_error.values())) if self.last_prediction_error else 0
        
        method_desc = 'Polynomial' if self.use_polynomial else 'Linear'
        method_desc += ' Ridge (Pixel -> Motor)'
        if self.use_phi:
            method_desc += f' with φ [w={self.phi_weight:.2f}]'
        if self.config.calibration_use_average_movement:
            method_desc += ' [Averaged]'
        if self.config.calibration_use_multi_frame:
            method_desc += f' [{self.config.calibration_pairing_strategy}]'
        
        avg_std_x = np.mean(self.training_stats['pixel_delta_std']['x']) if self.training_stats['pixel_delta_std']['x'] else 0
        avg_std_y = np.mean(self.training_stats['pixel_delta_std']['y']) if self.training_stats['pixel_delta_std']['y'] else 0
        
        return {
            'is_calibrated': True,
            'samples_collected': len(self.motor_history),
            'max_samples': self.max_samples if self.max_samples else 'Unlimited',
            'calibration_scores': self.calibration_scores,
            'avg_score': avg_score,
            'method': method_desc,
            'uses_phi': self.use_phi,
            'phi_weight': self.phi_weight if self.use_phi else None,
            'mapping_direction': 'Pixel -> Motor (Inverse)',
            'averaging_enabled': self.config.calibration_use_average_movement,
            'multi_frame_enabled': self.config.calibration_use_multi_frame,
            'pairing_strategy': self.config.calibration_pairing_strategy,
            'training_stats': dict(self.training_stats),
            'unique_frame_pairs': len(self.training_stats['unique_frame_pairs']),
            'avg_frame_gap': self.training_stats['average_frame_gap'],
            'avg_common_wells': self.training_stats['average_common_wells'],
            'avg_pixel_std': {'x': avg_std_x, 'y': avg_std_y}
        }
    
    def export_calibration(self) -> Optional[Dict]:
        """Export calibration data for saving to JSON"""
        if not self.is_calibrated:
            return None
        
        model_type_desc = 'Inverse ' + ('Polynomial' if self.use_polynomial else 'Linear')
        model_type_desc += ' Ridge (Pixel -> Motor)'
        if self.use_phi:
            model_type_desc += f' with Phi (weight={self.phi_weight})'
        if self.config.calibration_use_average_movement:
            model_type_desc += ' [Average Movement]'
        if self.config.calibration_use_multi_frame:
            model_type_desc += f' [Multi-Frame: {self.config.calibration_pairing_strategy}]'
        
        avg_std_x = np.mean(self.training_stats['pixel_delta_std']['x']) if self.training_stats['pixel_delta_std']['x'] else 0
        avg_std_y = np.mean(self.training_stats['pixel_delta_std']['y']) if self.training_stats['pixel_delta_std']['y'] else 0
        
        calibration_data = {
            'model_type': model_type_desc,
            'mapping_direction': 'Pixel -> Motor (Inverse)',
            'alpha': float(self.alpha),
            'uses_phi': self.use_phi,
            'phi_weight': float(self.phi_weight) if self.use_phi else None,
            'num_samples': len(self.motor_history),
            'max_samples': self.max_samples if self.max_samples else 'Unlimited',
            'calibration_scores': {k: float(v) for k, v in self.calibration_scores.items()},
            'avg_score': float(np.mean(list(self.calibration_scores.values()))),
            'averaging_settings': {
                'use_average_movement': self.config.calibration_use_average_movement,
                'min_common_wells': self.config.calibration_min_common_wells,
                'avg_pixel_delta_std': {'x': float(avg_std_x), 'y': float(avg_std_y)}
            },
            'multi_frame_settings': {
                'enabled': self.config.calibration_use_multi_frame,
                'strategy': self.config.calibration_pairing_strategy
            },
            'training_statistics': {
                'total_pairs': self.training_stats['total_pairs_generated'],
                'unique_frame_pairs': len(self.training_stats['unique_frame_pairs']),
                'avg_frame_gap': float(self.training_stats['average_frame_gap']),
                'min_frame_gap': int(self.training_stats['min_frame_gap']) if self.training_stats['min_frame_gap'] != float('inf') else None,
                'max_frame_gap': int(self.training_stats['max_frame_gap']),
                'avg_common_wells': float(self.training_stats['average_common_wells']),
                'min_common_wells': int(self.training_stats['min_common_wells']) if self.training_stats['min_common_wells'] != float('inf') else None,
                'max_common_wells': int(self.training_stats['max_common_wells'])
            }
        }
        
        return calibration_data


class ImageProcessor:
    """Image processing utilities"""
    
    def __init__(self, config: Config):
        self.config = config
        self.circle_detector = CircleDetector(config)
        if config.enable_edge_detection:
            self.line_detector = LineDetector(config)
            self.contour_processor = ContourProcessor(config)
        else:
            self.line_detector = None
            self.contour_processor = None
        
        # Initialize rembg session if background removal is enabled
        self.rembg_session = None
        if config.use_background_removal and REMBG_AVAILABLE:
            try:
                if config.rembg_model:
                    self.rembg_session = new_session(config.rembg_model)
                    logger.info(f"Background Removal Initialized With Model: {config.rembg_model}")
                else:
                    logger.info("Background Removal Initialized With Default Model")
            except Exception as e:
                logger.warning(f"Failed To Initialize Rembg Session: {e}")
    
    def remove_background(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background from image using rembg and return processed image and mask"""
        import sys
        if not REMBG_AVAILABLE or not PIL_AVAILABLE or not self.config.use_background_removal:
            mask = np.ones_like(img, dtype=np.uint8) * 255
            return img, mask
        
        try:
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                img_normalized = img
            
            # Convert grayscale to RGB for rembg
            if len(img_normalized.shape) == 2:
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
            else:
                img_rgb = img_normalized
            
            pil_image = Image.fromarray(img_rgb)
            
            # Use the pre-created session if available, otherwise use default
            if self.rembg_session is not None:
                output = remove(pil_image, session=self.rembg_session)
            else:
                output = remove(pil_image)
                
            result_array = np.array(output)
            
            # Extract mask from alpha channel if available
            if result_array.shape[2] == 4:  # RGBA
                alpha_channel = result_array[:, :, 3]
                mask = (alpha_channel > 0).astype(np.uint8) * 255
                
                if len(img_normalized.shape) == 2:
                    result = np.where(alpha_channel > 0, img_normalized, 0)
                else:
                    gray_result = cv2.cvtColor(result_array[:, :, :3], cv2.COLOR_RGB2GRAY)
                    result = np.where(alpha_channel > 0, gray_result, 0)
            else:
                if len(img_normalized.shape) == 2:
                    result = cv2.cvtColor(result_array, cv2.COLOR_RGB2GRAY)
                else:
                    result = result_array
                
                mask = (result > 0).astype(np.uint8) * 255
            
            logger.debug(f"Background Removal Completed Successfully Using Model: {self.config.rembg_model or 'default'}")
            return result.astype(np.uint8), mask
            
        except Exception as e:
            logger.warning(f"Background Removal Failed: {e}. Using Original Image")
            mask = np.ones_like(img, dtype=np.uint8) * 255
            return img, mask
    
    def generate_edge_image(self, img: np.ndarray) -> np.ndarray:
        """Generate edge image using Canny edge detection"""
        return canny(img, sigma=self.config.edge_sigma,
                    low_threshold=self.config.edge_low_threshold,
                    high_threshold=self.config.edge_high_threshold,
                    use_quantiles=True)
    
    def find_circles(self, img: np.ndarray) -> Tuple:
        """Detect circles using Hough transform"""
        return self.circle_detector.detect_circles(img)
    
    def extract_contour_coordinates(self, edge_image: np.ndarray, min_area: Optional[int] = None, 
                                   remove_border_points: bool = True, 
                                   border_buffer: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """Extract and process contour coordinates from edge image"""
        if not self.config.enable_edge_detection or self.contour_processor is None:
            return None, None
        return self.contour_processor.extract_contour_coordinates(edge_image, min_area, remove_border_points, border_buffer)
    
    def find_lines(self, contour_coords: Optional[np.ndarray], img_shape: Tuple, 
                  segments: Optional[List] = None,
                  backup_img: Optional[np.ndarray] = None) -> Tuple:
        """Detect lines using Hough transform"""
        if not self.config.enable_edge_detection or self.line_detector is None:
            return None, None, []
        return self.line_detector.detect_lines(contour_coords, img_shape, segments, backup_img)


class Visualizer:
    """Visualization utilities"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def add_well_tracking_visualization(self, ax, tracked_circles: Optional[Tuple],
                                       well_ids: Optional[List],
                                       well_tracker: WellTracker):
        """Add well tracking visualization for two-row configuration"""
        if not well_tracker:
            return
        
        # Draw fitted lines for both rows
        line_endpoints = well_tracker.get_line_endpoints()
        if line_endpoints:
            for i, endpoints in enumerate(line_endpoints):
                if endpoints:
                    (x1, y1), (x2, y2) = endpoints
                    color = 'yellow' if i == 0 else 'cyan'
                    label = f'Row {i+1} Line'
                    ax.plot([x1, x2], [y1, y2],
                           color, linewidth=3, alpha=0.5, linestyle='--',
                           label=label)
        
        # Draw predicted positions for missing wells
        predicted_positions = well_tracker.get_all_predicted_positions()
        
        if predicted_positions:
            for well_id, pred in predicted_positions.items():
                row = pred.get('row', 1)
                color = 'yellow' if row == 1 else 'cyan'
                circle = plt.Circle((pred['x'], pred['y']), pred['radius'],
                                  ec=color, fc='none', ls=':', alpha=0.3, lw=2)
                ax.add_patch(circle)
                
                label = format_well_label(well_id, self.config)
                ax.text(pred['x'], pred['y'], label,
                       ha='center', va='center', fontsize=10,
                       color=color, alpha=0.5,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.3))
        
        # Draw tracked circles
        if tracked_circles and well_ids:
            accum, cx, cy, radii = tracked_circles
            
            for x, y, r, well_id, conf in zip(cx, cy, radii, well_ids, accum):
                if well_id:
                    well_info = well_tracker.detected_wells.get(well_id, {})
                    row = well_info.get('row', 1)
                    
                    color = 'lime' if row == 1 else 'aqua'
                    label = format_well_label(well_id, self.config)
                    
                    circle = plt.Circle((x, y), r, ec=color, fc='none',
                                      ls='-', alpha=0.8, lw=4)
                    ax.add_patch(circle)
                    
                    ax.text(x, y, label, ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
                else:
                    circle = plt.Circle((x, y), r, ec='orange', fc='none',
                                      ls='-', alpha=0.8, lw=4)
                    ax.add_patch(circle)
                    
                    ax.text(x, y, '?', ha='center', va='center',
                           fontsize=14, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))


class WellTrackingSystem:
    """Main system for well tracking with inverse motor calibration"""
    
    def __init__(self, config: Config):
        self.config = config
        
        setup_logging(config.verbose_mode, config.log_directory)
        
        self.well_tracker = WellTracker(config) if config.enable_well_tracking else None
        self.well_center_tracker = (WellCenterTracker(config=config)
                                   if config.track_well_centers and config.enable_well_tracking
                                   else None)
        self.motor_calibration = (InverseMotorCalibration(config)
                                 if config.enable_motor_calibration else None)
        self.image_processor = ImageProcessor(config)
        self.visualizer = Visualizer(config)
        
        self.frames_processed = 0
        self.frames_skipped = 0
        self.frames_with_tracking = 0
        
        self.prev_motor_data = None
        self.prev_detected_wells = {}
        
        for dir_path in [config.output_dir, config.output_images_dir, config.output_json_dir]:
            Path(dir_path).mkdir(exist_ok=True)
        
        # Log background removal status
        if config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                model_desc = config.rembg_model if config.rembg_model else "default"
                logger.info(f"Background Removal Enabled With Model: {model_desc}")
            else:
                missing_libs = []
                if not REMBG_AVAILABLE:
                    missing_libs.append("rembg")
                if not PIL_AVAILABLE:
                    missing_libs.append("PIL/Pillow")
                logger.warning(f"Background Removal Requested But Required Libraries Not Available: {', '.join(missing_libs)}")
        else:
            logger.info("Background Removal Disabled")
    
    def load_frame_data(self, frame_number: int) -> Tuple[Optional[MotorPosition], Optional[np.ndarray]]:
        """Load frame data from .npz files"""
        try:
            data = np.load(f"{self.config.data_path}test{frame_number}.npz")
            motor_pos = MotorPosition(
                x=float(data['x']),
                y=float(data['y']),
                z=float(data['z']),
                phi=float(data['phi'])
            )
            return motor_pos, data['sample']
        except Exception as e:
            return None, None
    
    def process_frame_detection(self, frame_number: int, img: np.ndarray,
                               motor_data: MotorPosition) -> Dict:
        """Process frame for detection"""
        if self.well_center_tracker and self.well_center_tracker.frame_shape is None:
            self.well_center_tracker.set_frame_shape(img.shape)
        
        original_img = img.copy()
        
        # Apply background removal for edge detection if enabled
        img_bg_removed = None
        bg_mask = None
        should_use_background_removal = (
            self.config.use_background_removal and 
            (not self.well_tracker or not self.well_tracker.edge_condition_satisfied)
        )
        if should_use_background_removal:
            img_bg_removed, bg_mask = self.image_processor.remove_background(img)
        
        # Circle detection uses original image
        edge_for_circles, circles = self.image_processor.find_circles(img)
        
        # Check if edge detection should be performed
        should_perform_edge_detection = (
            self.config.enable_edge_detection and 
            (not self.well_tracker or not self.well_tracker.edge_condition_satisfied)
        )
        
        contour_coords = None
        segments = None
        lines = []
        contour_img = None
        edge_for_contours = None
        
        if should_perform_edge_detection:
            # Use background-removed image for edge detection if available
            img_for_edge = img_bg_removed if img_bg_removed is not None else img
            
            edge_for_contours = self.image_processor.generate_edge_image(img_for_edge)
            
            contour_coords, segments = self.image_processor.extract_contour_coordinates(
                edge_for_contours, remove_border_points=True)
            
            if contour_coords is not None and segments is not None:
                contour_img, ph, lines = self.image_processor.find_lines(
                    contour_coords, img_for_edge.shape, segments, backup_img=img_for_edge)
            else:
                contour_img, ph, lines = self.image_processor.find_lines(
                    None, img_for_edge.shape, None, backup_img=img_for_edge)
        
        num_circles = len(circles[1]) if circles else 0
        
        return {
            'img': original_img,
            'img_bg_removed': img_bg_removed,
            'bg_mask': bg_mask,
            'edge_for_circles': edge_for_circles,
            'edge_for_contours': edge_for_contours,
            'contour_img': contour_img,
            'circles': circles,
            'lines': lines,
            'contour_coords': contour_coords,
            'segments': segments,
            'num_circles': num_circles
        }
    
    def update_tracking(self, frame_number: int, detection_results: Dict, 
                       motor_data: MotorPosition) -> Dict:
        """Update tracking based on detection results"""
        tracking_results = {
            'tracked_circles': None,
            'well_ids': None,
            'motor_calibration_info': {},
            'suggested_motor_positions': {},
            'edge_detection_status': {}
        }
        
        if not self.config.enable_well_tracking or not self.well_tracker:
            return tracking_results
        
        tracked_circles, well_ids = self.well_tracker.update_tracks(
            detection_results['circles'], 
            detection_results.get('lines', []),
            motor_data.phi
        )
        
        tracking_results['tracked_circles'] = tracked_circles
        tracking_results['well_ids'] = well_ids
        tracking_results['edge_detection_status'] = self.well_tracker.get_edge_detection_status()
        
        if self.motor_calibration and self.well_tracker.detected_wells:
            self.motor_calibration.add_observation(motor_data, self.well_tracker.detected_wells, frame_number)
            
            if self.motor_calibration.is_calibrated and self.well_center_tracker:
                frame_center = self.well_center_tracker.frame_center
                if frame_center:
                    if self.config.calibration_use_average_movement and len(self.well_tracker.detected_wells) > 0:
                        pixel_positions = []
                        for well_info in self.well_tracker.detected_wells.values():
                            pixel_positions.append([well_info['x'], well_info['y']])
                        avg_pixel_pos = np.mean(pixel_positions, axis=0)
                        
                        suggested_motor = self.motor_calibration.estimate_motor_for_well_centering(
                            motor_data,
                            tuple(avg_pixel_pos),
                            frame_center
                        )
                        if suggested_motor:
                            tracking_results['suggested_motor_positions']['average'] = suggested_motor
                    else:
                        for well_id, well_info in self.well_tracker.detected_wells.items():
                            suggested_motor = self.motor_calibration.estimate_motor_for_well_centering(
                                motor_data,
                                (well_info['x'], well_info['y']),
                                frame_center
                            )
                            if suggested_motor:
                                tracking_results['suggested_motor_positions'][well_id] = suggested_motor
            
            tracking_results['motor_calibration_info'] = self.motor_calibration.get_calibration_info()
        
        if self.well_center_tracker and self.well_tracker.detected_wells:
            self.well_center_tracker.update(frame_number,
                                           self.well_tracker.detected_wells,
                                           motor_data)
        
        self.prev_motor_data = motor_data
        
        return tracking_results
    
    def create_visualization(self, frame_number: int, results: Dict,
                           motor_data: MotorPosition) -> plt.Figure:
        """Create comprehensive debug visualization figure"""
        fig = plt.figure(figsize=(28, 20))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.2)
        
        # Create axes for debug frames
        axes = [
            fig.add_subplot(gs[0, 0]),  # Original
            fig.add_subplot(gs[0, 1]),  # Circle detection edge  
            fig.add_subplot(gs[0, 2]),  # Circle detection
            fig.add_subplot(gs[1, 0]),  # Background removed
            fig.add_subplot(gs[1, 1]),  # Contour edge
            fig.add_subplot(gs[1, 2]),  # Contours and lines
            fig.add_subplot(gs[2, :])   # Main tracking (full width)
        ]
        
        self._create_debug_subplots(axes, results, motor_data, frame_number)
        
        return fig
    
    def _create_debug_subplots(self, axes, results: Dict, motor_data: MotorPosition, frame_number: int):
        """Create all debug subplots with improved background removal visualization"""
        
        # Original image
        axes[0].imshow(results['img'], cmap='gray', aspect='equal')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Circle detection edge
        if results['edge_for_circles'] is not None:
            axes[1].imshow(results['edge_for_circles'], cmap='gray', aspect='equal')
            axes[1].set_title('Canny Edges (Circle Detection)', fontsize=12, fontweight='bold', color='blue')
        else:
            axes[1].text(0.5, 0.5, 'No Edge Data', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=14, color='red')
            axes[1].set_title('Canny Edges: Not Available', fontsize=12, fontweight='bold', color='red')
        axes[1].axis('off')
        
        # Circle detection
        axes[2].imshow(results['img'], cmap='gray', aspect='equal')
        if results['circles']:
            self._draw_circle_detections_debug(axes[2], results['circles'])
        axes[2].set_title(f'Circle Detection ({results["num_circles"]} Detected)', fontsize=12, fontweight='bold', color='blue')
        axes[2].axis('off')
        
        # Background removed image
        edge_status = results.get('edge_detection_status', {})
        
        if results['img_bg_removed'] is not None:
            axes[3].imshow(results['img_bg_removed'], cmap='gray', aspect='equal')
            axes[3].set_title('Background Removed (Complete)', fontsize=12, fontweight='bold', color='green')
            
        elif self.config.use_background_removal and edge_status.get('edge_condition_satisfied', False):
            axes[3].text(0.5, 0.5, 'Background Removal\nDisabled', 
                        ha='center', va='center', transform=axes[3].transAxes, 
                        fontsize=11, color='gray')
            axes[3].set_title('Background Removal: Disabled', fontsize=12, fontweight='bold', color='gray')
            
        elif self.config.use_background_removal:
            axes[3].imshow(results['img'], cmap='gray', aspect='equal')
            axes[3].set_title('Background Removal: Error/Failed', fontsize=12, fontweight='bold', color='red')
            
            axes[3].text(0.5, 0.5, 'Background Removal\nFailed or Not Applied', 
                        ha='center', va='center', transform=axes[3].transAxes, 
                        fontsize=10, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            axes[3].text(0.5, 0.5, 'Background Removal\nDisabled', ha='center', va='center', 
                        transform=axes[3].transAxes, fontsize=12, color='gray')
            axes[3].set_title('Background Removal: Disabled', fontsize=12, fontweight='bold', color='gray')
        
        axes[3].axis('off')
        
        # Contour edge detection
        if results['edge_for_contours'] is not None:
            axes[4].imshow(results['edge_for_contours'], cmap='gray', aspect='equal')
            if edge_status.get('edge_condition_satisfied', False):
                axes[4].set_title('Edge Detection (Complete)', fontsize=12, fontweight='bold', color='green')
            else:
                axes[4].set_title('Edge Detection (Waiting)', fontsize=12, fontweight='bold', color='orange')
        else:
            axes[4].text(0.5, 0.5, 'Edge Detection\nDisabled', ha='center', va='center', 
                        transform=axes[4].transAxes, fontsize=12, color='gray')
            axes[4].set_title('Edge Detection: Disabled', fontsize=12, fontweight='bold', color='gray')
        axes[4].axis('off')
        
        # Contours and lines overlay
        if results['edge_for_contours'] is not None:
            axes[5].imshow(results['edge_for_contours'], cmap='gray', aspect='equal')
            
            contour_coords = results.get('contour_coords')
            segments = results.get('segments')
            if contour_coords is not None and segments is not None:
                for i, segment in enumerate(segments):
                    if len(segment) > 1:
                        axes[5].plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2, 
                                    alpha=0.8, label='Hull Boundary' if i == 0 else "")
                axes[5].scatter(contour_coords[:, 0], contour_coords[:, 1], c='red', s=20, 
                            alpha=0.8, zorder=5, label='Hull Vertices')
            
            # Draw detected lines
            first_line = True
            for xline, yline in results.get('lines', []):
                label = 'Detected Lines' if first_line else None
                axes[5].plot(xline, yline, 'cyan', linewidth=2, alpha=0.9, linestyle='--', label=label)
                first_line = False
            
            if edge_status.get('edge_condition_satisfied', False):
                axes[5].set_title('Contours & Lines (Complete)', fontsize=12, fontweight='bold', color='green')
            else:
                axes[5].set_title('Contours & Lines (Waiting)', fontsize=12, fontweight='bold', color='orange')
        else:
            axes[5].text(0.5, 0.5, 'Edge Detection\nDisabled', ha='center', va='center', 
                        transform=axes[5].transAxes, fontsize=12, color='gray')
            axes[5].set_title('Contours & Lines: Disabled', fontsize=12, fontweight='bold', color='gray')
        axes[5].axis('off')
        
        # Main tracking visualization
        self._create_well_tracking_subplot(axes[6], results, motor_data, frame_number)
    
    def _draw_circle_detections_debug(self, ax, circles: Tuple):
        """Draw detected circles with confidence information for debug view"""
        accum_values, cx, cy, radii = circles
        
        if len(accum_values) == 0:
            ax.text(0.5, 0.5, 'No Circles Detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color='red', fontweight='bold')
            return
            
        max_accum = np.max(accum_values)
        min_accum = np.min(accum_values)
        accum_range = max_accum - min_accum if max_accum > min_accum else 1.0
        
        for i, (x, y, r, acc) in enumerate(zip(cx, cy, radii, accum_values)):
            normalized_conf = (acc - min_accum) / accum_range if accum_range > 0 else 1.0
            alpha = 0.5 + 0.5 * normalized_conf
            linewidth = 1 + 2 * normalized_conf
            
            # Color based on confidence
            if normalized_conf > 0.75:
                color = 'lime'
            elif normalized_conf > 0.5:
                color = 'cyan'
            else:
                color = 'yellow'
            
            circle = plt.Circle((x, y), r, ec=color, fc='none', ls='-', 
                              alpha=alpha, lw=linewidth)
            ax.add_patch(circle)
            
            # Add confidence text
            ax.text(x, y, f'{acc:.2f}', ha='center', va='center', fontsize=8, 
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
            
            # Mark center
            ax.plot(x, y, 'o', color=color, markersize=2, alpha=alpha)
    
    def _create_well_tracking_subplot(self, ax, results: Dict, motor_data: MotorPosition, frame_number: int):
        """Create the main well tracking subplot"""
        ax.imshow(results['img'], cmap='gray')
        
        # Only show edge detection visualizations if edge condition is not satisfied yet
        edge_status = results.get('edge_detection_status', {})
        show_edge_detection = (
            self.config.enable_edge_detection and 
            not edge_status.get('edge_condition_satisfied', False)
        )
        
        if show_edge_detection:
            contour_coords = results.get('contour_coords')
            segments = results.get('segments')
            if contour_coords is not None and segments is not None:
                for i, segment in enumerate(segments):
                    if len(segment) > 1:
                        ax.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=3, 
                                alpha=0.8, label='Convex Hull Boundary' if i == 0 else "")
                ax.scatter(contour_coords[:, 0], contour_coords[:, 1], c='red', s=30, 
                           alpha=0.8, zorder=5, label='Hull Vertices')
            
            first_line = True
            for xline, yline in results['lines']:
                label = 'Detected Lines' if first_line else None
                ax.plot(xline, yline, 'cyan', linewidth=2, alpha=0.9, linestyle='--', label=label)
                first_line = False
        
        if results['circles']:
            self._draw_circle_detections(ax, results['circles'])
        
        if results.get('tracked_circles') or self.well_tracker:
            self.visualizer.add_well_tracking_visualization(
                ax,
                results['tracked_circles'],
                results['well_ids'],
                self.well_tracker
            )
        
        if self.well_tracker and len(self.well_tracker.detected_wells) > 1:
            self._draw_stagger_relationships(ax)
        
        title = self._generate_tracking_title(frame_number, motor_data, results)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        self._add_motor_position_box(ax, motor_data)
        self._add_calibration_info_box(ax, results)
        self._add_motor_suggestion_box(ax, results)
        
        legend_elements = self._get_well_tracking_legend_elements(results)
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
                  framealpha=0.9, edgecolor='white', ncol=2)
        
        ax.axis('off')
    
    def _draw_circle_detections(self, ax, circles: Tuple):
        """Draw detected circles with confidence information"""
        accum_values, cx, cy, radii = circles
        first_circle = True
        
        if len(accum_values) == 0:
            return
            
        max_accum = np.max(accum_values)
        min_accum = np.min(accum_values)
        accum_range = max_accum - min_accum if max_accum > min_accum else 1.0
        
        for i, (x, y, r, acc) in enumerate(zip(cx, cy, radii, accum_values)):
            label = 'Hough Circles' if first_circle else None
            
            normalized_conf = (acc - min_accum) / accum_range if accum_range > 0 else 1.0
            alpha = 0.5 + 0.5 * normalized_conf
            linewidth = 2 + 2 * normalized_conf
            
            circle = plt.Circle((x, y), r, ec='cyan', fc='none', ls='--', 
                              alpha=alpha, lw=linewidth, label=label)
            ax.add_patch(circle)
            
            info_text = f"({x:.0f}, {y:.0f})\nConf: {acc:.3f}\nR: {r:.0f}"
            
            if normalized_conf > 0.75:
                box_color = 'lime'
                text_color = 'lime'
            elif normalized_conf > 0.5:
                box_color = 'cyan'
                text_color = 'cyan'
            else:
                box_color = 'yellow'
                text_color = 'yellow'
            
            ax.text(x, y - r - 15, info_text, 
                    ha='center', va='bottom', fontsize=8, 
                    color=text_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                             edgecolor=box_color, alpha=0.9, linewidth=1))
            
            ax.plot(x, y, 'o', color='cyan', markersize=3, alpha=alpha)
            
            first_circle = False
    
    def _draw_stagger_relationships(self, ax):
        """Draw lines showing stagger relationships between rows"""
        if not self.well_tracker:
            return
            
        row1_wells = {wid: w for wid, w in self.well_tracker.detected_wells.items() if w.get('row') == 1}
        row2_wells = {wid: w for wid, w in self.well_tracker.detected_wells.items() if w.get('row') == 2}
        
        for well_id, well_info in row1_wells.items():
            row, col = well_id_to_row_col(well_id, self.config)
            row2_right_id = self.config.total_wells_row1 + col
            row2_left_id = self.config.total_wells_row1 + col + 1
            
            row2_right = row2_wells.get(row2_right_id)
            row2_left = row2_wells.get(row2_left_id)
            
            if row2_right:
                ax.plot([well_info['x'], row2_right['x']], 
                       [well_info['y'], row2_right['y']], 
                       'gray', alpha=0.3, linewidth=1, linestyle=':')
            if row2_left:
                ax.plot([well_info['x'], row2_left['x']], 
                       [well_info['y'], row2_left['y']], 
                       'gray', alpha=0.3, linewidth=1, linestyle=':')
    
    def _get_well_tracking_legend_elements(self, results: Dict) -> List[Line2D]:
        """Get legend elements for well tracking subplot"""
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                   markersize=10, lw=3, markeredgecolor='lime', label=f'Row 1 Wells (1,1)-(1,{self.config.total_wells_row1})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua', 
                   markersize=10, lw=3, markeredgecolor='aqua', label=f'Row 2 Wells (2,1)-(2,{self.config.total_wells_row2})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                   markersize=10, lw=2, markeredgecolor='yellow', linestyle=':', label='Predicted R1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                   markersize=10, lw=2, markeredgecolor='cyan', linestyle=':', label='Predicted R2'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                   markersize=10, lw=3, markeredgecolor='orange', label='Unassigned'),
            Line2D([0], [0], color='yellow', lw=3, ls='--', alpha=0.5, label='Row 1 Line'),
            Line2D([0], [0], color='cyan', lw=3, ls='--', alpha=0.5, label='Row 2 Line')
        ]
        
        # Only show edge detection elements if edge condition is not satisfied yet
        edge_status = results.get('edge_detection_status', {})
        show_edge_detection = (
            self.config.enable_edge_detection and 
            not edge_status.get('edge_condition_satisfied', False)
        )
        
        if show_edge_detection:
            legend_elements.insert(0, Line2D([0], [0], color='red', lw=3, label='Convex Hull'))
            legend_elements.insert(1, Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, lw=0, label='Hull Vertices'))
        
        if self.config.enable_edge_detection:
            if edge_status.get('edge_condition_satisfied'):
                legend_elements.append(Line2D([0], [0], color='green', lw=3, label='Edge Detection: Complete'))
            else:
                legend_elements.append(Line2D([0], [0], color='red', lw=3, label='Edge Detection: Waiting'))
        
        # Show background removal status based on whether it's currently active
        if self.config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                if edge_status.get('edge_condition_satisfied', False):
                    legend_elements.append(Line2D([0], [0], color='gray', lw=3, label='Background Removal: Disabled'))
                else:
                    legend_elements.append(Line2D([0], [0], color='purple', lw=3, label='Background Removal: Active'))
            else:
                legend_elements.append(Line2D([0], [0], color='orange', lw=3, label='Background Removal: Unavailable'))
        
        return legend_elements
    
    def _generate_tracking_title(self, frame_number: int, motor_data: MotorPosition, results: Dict) -> str:
        """Generate comprehensive title for tracking subplot"""
        title_parts = [
            f"Frame {frame_number}",
            f"φ={motor_data.phi:.1f}°",
            f"Circles: {results['num_circles']}"
        ]
        
        if self.well_tracker:
            row1_count = sum(1 for w in self.well_tracker.detected_wells.values() if w.get('row') == 1)
            row2_count = sum(1 for w in self.well_tracker.detected_wells.values() if w.get('row') == 2)
            title_parts.append(f"R1={row1_count}/{self.config.total_wells_row1}, R2={row2_count}/{self.config.total_wells_row2}")
        
        if self.well_tracker:
            prediction_count = len(self.well_tracker.predicted_positions) if self.well_tracker.predicted_positions else 0
            total_assigned = len(self.well_tracker.detected_wells)
            unassigned_count = len(self.well_tracker.unassigned_detections)
            
            if unassigned_count > 0:
                title_parts.append(f"Unassigned={unassigned_count}")
            elif prediction_count > 0:
                title_parts.append(f"Pred={prediction_count}")
            elif total_assigned == 0:
                title_parts.append("All Unassigned")
            elif (not self.well_tracker.established_spacing or 
                  not any(row_id in self.well_tracker.row_params for row_id in [1, 2])):
                if total_assigned < 2:
                    title_parts.append(f"Learning ({total_assigned}/2)")
                else:
                    title_parts.append("Learning Complete")
            else:
                title_parts.append("Ready (Established)")
        
        # Edge detection status
        edge_status = results.get('edge_detection_status', {})
        if self.config.enable_edge_detection:
            if edge_status.get('edge_condition_satisfied'):
                title_parts.append(f"Edge@F{edge_status.get('edge_detection_frame', '?')}")
            else:
                title_parts.append("Awaiting Edge")
        
        # Background removal status
        if self.config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                if edge_status.get('edge_condition_satisfied', False):
                    title_parts.append("BG-Disabled")
                else:
                    title_parts.append("BG-Active")
            else:
                title_parts.append("BG-Error")
        
        # Row flip status
        if edge_status.get('row_layout_flipped'):
            flip_count = len(edge_status.get('phi_flip_history', []))
            if flip_count > 0:
                title_parts.append(f"Layout Flipped ({flip_count}x)")
            else:
                title_parts.append("Layout Flipped")
        
        # Last successful frame
        last_successful = edge_status.get('last_successful_frame')
        if last_successful is not None:
            title_parts.append(f"LastOK@F{last_successful}")
        
        return " - ".join(title_parts)
    
    def _add_motor_position_box(self, ax, motor_data: MotorPosition):
        """Add motor position information box"""
        motor_text = (f"Motor Positions\n"
                     f"X: {motor_data.x:.3f}\n"
                     f"Y: {motor_data.y:.3f}\n"
                     f"Z: {motor_data.z:.3f}\n"
                     f"φ: {motor_data.phi:.3f}°")
        ax.text(0.02, 0.98, motor_text,
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, 
                         edgecolor='darkblue', linewidth=2))
    
    def _add_calibration_info_box(self, ax, results: Dict):
        """Add motor calibration information box"""
        calibration_info = results.get('motor_calibration_info', {})
        if not calibration_info:
            return
            
        if calibration_info.get('is_calibrated'):
            max_samples_text = calibration_info.get('max_samples', 'Unlimited')
            cal_text = (f"Inverse Calibration: {calibration_info['method']}\n"
                      f"Score: {calibration_info['avg_score']:.3f} | "
                      f"Samples: {calibration_info['samples_collected']} (Max: {max_samples_text})\n"
                      f"Wells: {calibration_info.get('avg_common_wells', 0):.1f} | "
                      f"Std: σx={calibration_info.get('avg_pixel_std', {}).get('x', 0):.2f}px "
                      f"σy={calibration_info.get('avg_pixel_std', {}).get('y', 0):.2f}px")
            cal_color = 'green' if calibration_info['avg_score'] > 0.8 else 'orange'
        else:
            max_samples_text = calibration_info.get('max_samples', 'Unlimited')
            cal_text = (f"Inverse Calibration: Learning...\n"
                      f"Samples: {calibration_info['samples_collected']}/{calibration_info['samples_needed']} "
                      f"(Max: {max_samples_text})\n"
                      f"Mode: {'Averaged' if calibration_info.get('averaging_enabled') else 'Individual'}")
            cal_color = 'gray'
        
        ax.text(0.98, 0.98, cal_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=cal_color, alpha=0.6))
    
    def _add_motor_suggestion_box(self, ax, results: Dict):
        """Add motor position suggestions box"""
        suggested_positions = results.get('suggested_motor_positions', {})
        if not suggested_positions or len(suggested_positions) == 0:
            return
            
        if 'average' in suggested_positions:
            suggested_motor = suggested_positions['average']
            suggestion_text = (f"To Center Wells (Avg):\n"
                             f"Move To X: {suggested_motor.x:.3f}\n"
                             f"Y: {suggested_motor.y:.3f}\n"
                             f"Z: {suggested_motor.z:.3f}")
            if self.motor_calibration and self.motor_calibration.use_phi:
                suggestion_text += f"\nφ: {suggested_motor.phi:.3f}°"
        else:
            first_well_id = min(suggested_positions.keys())
            suggested_motor = suggested_positions[first_well_id]
            well_label = format_well_label(first_well_id, self.config)
            suggestion_text = (f"To Center Well {well_label}:\n"
                             f"Move To X: {suggested_motor.x:.3f}\n"
                             f"Y: {suggested_motor.y:.3f}\n"
                             f"Z: {suggested_motor.z:.3f}")
            if self.motor_calibration and self.motor_calibration.use_phi:
                suggestion_text += f"\nφ: {suggested_motor.phi:.3f}°"
        
        ax.text(0.98, 0.02, suggestion_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def generate_frame_sequence(self) -> List[int]:
        """Generate frame sequence based on loop_count parameter"""
        frames = []
        
        forward_frames = list(range(self.config.min_frame, self.config.max_frame))
        reverse_frames = list(range(self.config.max_frame - 1, self.config.min_frame, -1))
        
        for i in range(self.config.loop_count + 1):
            if i == 0:
                frames.extend(forward_frames)
            elif i % 2 == 1:
                frames.extend(reverse_frames)
            else:
                frames.extend(forward_frames[1:])
        
        return frames
    
    def run(self):
        """Main processing loop"""
        try:
            from IPython.display import clear_output, display
            use_ipython = True
        except ImportError:
            use_ipython = False
            import os
        
        writer = None
        video_path = None
        
        if self.config.save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"well_tracking_debug_enhanced_{timestamp}.mp4"
            video_path = Path(self.config.output_dir) / video_filename
            writer = imageio.get_writer(video_path, fps=self.config.video_fps)
            logger.info(f"Recording Debug Video To: {video_path}")
        
        frame_sequence = self.generate_frame_sequence()
        total_frames_to_process = len(frame_sequence)
        
        logger.info(f"Processing {total_frames_to_process} Frames With Enhanced Debug Visualization...")
        logger.info(f"Debug Frames Include: Original, Circle Edges, Circle Detection, Background-Removed, Edge Detection, Contours+Lines, Main Tracking")
        
        initial_layout = "Row 2 Top, Row 1 Bottom" if self.config.initial_row_layout_flipped else "Row 1 Top, Row 2 Bottom"
        logger.info(f"Initial Row Layout: {initial_layout}")
        
        if self.config.enable_edge_detection:
            logger.info(f"Edge Detection Enabled: Waiting For Circle Within {self.config.edge_distance_multiplier}x Radius Of Non-Horizontal Line")
            logger.info(f"Non-Horizontal Threshold: Lines Beyond ±{self.config.max_line_angle_degrees}° From Horizontal")
            logger.info(f"Well Arrangement: (x,1) Has Largest X [Right] → (x,max) Has Smallest X [Left]")
            logger.info(f"Tracking Reference Well: (2,1) Instead Of (1,1)")
            logger.info(f"Phi-Based Row Flipping Enabled: Layout Will Flip When Phi Changes >90° Between Frames")
            logger.info(f"Edge Detection Will Be Disabled Once Condition Is Satisfied")
        else:
            logger.info(f"Edge Detection Disabled: Well Tracking Will Begin Immediately")
            logger.info(f"Line Detection And Convex Hull Processing Skipped For Better Performance")
        
        if self.config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                model_desc = self.config.rembg_model if self.config.rembg_model else "default"
                logger.info(f"Background Removal Enabled: Using {model_desc} Model For Edge Detection")
                logger.info(f"Background Removal Will Be Disabled Once Edge Condition Is Satisfied")
                logger.info(f"Circle Detection Remains Unaffected By Background Removal")
            else:
                missing_libs = []
                if not REMBG_AVAILABLE:
                    missing_libs.append("rembg")
                if not PIL_AVAILABLE:
                    missing_libs.append("PIL/Pillow")
                logger.warning(f"Background Removal Requested But Required Libraries Not Available: {', '.join(missing_libs)}")
        else:
            logger.info(f"Background Removal Disabled")
        
        logger.info(f"Enhanced Tracking Features:")
        logger.info(f"  - Temporal Matching Uses Last Successful Frame (0 Unassigned Wells)")
        logger.info(f"  - Re-Evaluation System Reassigns All Wells When Unassigned Wells Detected")
        logger.info(f"  - Comprehensive Debug Visualization With 7 Subplots")
        logger.info(f"  - Reorganized Layout: Background Removal, Edge Detection, And Contours & Lines In Row 2")
        
        try:
            frame_index = 0
            current_direction = "Forward"
            loop_iteration = 0
            
            for frame_number in frame_sequence:
                if frame_index > 0:
                    prev_frame = frame_sequence[frame_index - 1]
                    if frame_number < prev_frame and current_direction == "Forward":
                        current_direction = "Reverse"
                        loop_iteration += 1
                    elif frame_number > prev_frame and current_direction == "Reverse":
                        current_direction = "Forward"
                        loop_iteration += 1
                
                frame_index += 1
                if self.config.display_frames:
                    if use_ipython:
                        clear_output(wait=True)
                    else:
                        os.system('cls' if os.name == 'nt' else 'clear')
                
                motor_data, img = self.load_frame_data(frame_number)
                
                if motor_data is None or img is None:
                    continue
                
                if not (self.config.phi_min <= motor_data.phi <= self.config.phi_max):
                    continue
                
                detection_results = self.process_frame_detection(frame_number, img, motor_data)
                tracking_results = self.update_tracking(frame_number, detection_results, motor_data)
                
                results = {**detection_results, **tracking_results}
                
                self.frames_processed += 1
                if results['tracked_circles']:
                    self.frames_with_tracking += 1
                
                if frame_index % 10 == 0:
                    progress_pct = (frame_index / total_frames_to_process) * 100
                    edge_status = results.get('edge_detection_status', {})
                    edge_msg = ""
                    if self.config.enable_edge_detection:
                        if edge_status.get('edge_condition_satisfied'):
                            edge_msg = f" | Edge@F{edge_status.get('edge_detection_frame', '?')} (Complete)"
                        else:
                            edge_msg = " | Awaiting Edge"
                    else:
                        edge_msg = " | Edge Detection Disabled"
                    
                    bg_msg = ""
                    if self.config.use_background_removal:
                        if REMBG_AVAILABLE and PIL_AVAILABLE:
                            if edge_status.get('edge_condition_satisfied', False):
                                bg_msg = " | BG-Disabled"
                            else:
                                bg_msg = " | BG-Active"
                        else:
                            bg_msg = " | BG-Error"
                    
                    flip_msg = ""
                    if edge_status.get('row_layout_flipped'):
                        flip_count = len(edge_status.get('phi_flip_history', []))
                        flip_msg = f" | Layout Flipped ({flip_count}x)"
                    
                    successful_msg = ""
                    if edge_status.get('last_successful_frame') is not None:
                        successful_msg = f" | LastOK@F{edge_status.get('last_successful_frame')}"
                    
                    logger.info(f"Progress: {progress_pct:.1f}% - Frame {frame_number} - φ={motor_data.phi:.1f}°{edge_msg}{bg_msg}{flip_msg}{successful_msg}")
                
                fig = self.create_visualization(frame_number, results, motor_data)
                
                if self.config.loop_count > 0:
                    fig.suptitle(f"Debug Visualization - Loop Mode: Pass {loop_iteration + 1}/{self.config.loop_count + 1} "
                               f"({current_direction}) - Frame {frame_index}/{total_frames_to_process}", 
                               fontsize=16, y=0.99)
                else:
                    fig.suptitle(f"Debug Visualization - Frame {frame_number} - Enhanced Well Tracking System", 
                               fontsize=16, y=0.99)
                
                if self.config.save_individual_frames:
                    if self.config.loop_count > 0:
                        frame_path = Path(self.config.output_images_dir) / f"debug_frame_{frame_number}_loop{loop_iteration}_{current_direction.lower()}.png"
                    else:
                        frame_path = Path(self.config.output_images_dir) / f"debug_frame_{frame_number}.png"
                    plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                
                if writer:
                    fig.canvas.draw()
                    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    writer.append_data(frame[:, :, :3])
                
                if self.config.display_frames:
                    if use_ipython:
                        display(fig)
                    else:
                        plt.show()
                
                plt.close(fig)
                
        except KeyboardInterrupt:
            logger.info("Processing Interrupted By User")
        finally:
            if writer:
                writer.close()
                logger.info(f"Debug Video Saved: {video_path}")
            
            if self.config.save_json_output and self.well_center_tracker:
                saved_path = self.well_center_tracker.save_to_json(motor_calibration=self.motor_calibration, well_tracker=self.well_tracker)
                logger.info(f"Results Saved To: {saved_path}")
            elif not self.config.save_json_output:
                logger.info(f"JSON Output Disabled - Results Not Saved To File")
            
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final processing summary"""
        logger.info(f"Processing Complete:")
        logger.info(f"  Frames Processed: {self.frames_processed}")
        logger.info(f"  Frames Skipped: {self.frames_skipped}")
        logger.info(f"  Frames With Tracking: {self.frames_with_tracking}")
        
        if self.config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                model_desc = self.config.rembg_model if self.config.rembg_model else "default"
                logger.info(f"  Background Removal: Used {model_desc} Model For Edge Detection Until Edge Condition Met")
            else:
                missing_libs = []
                if not REMBG_AVAILABLE:
                    missing_libs.append("rembg")
                if not PIL_AVAILABLE:
                    missing_libs.append("PIL/Pillow")
                logger.info(f"  Background Removal: Failed - Required Libraries Not Available: {', '.join(missing_libs)}")
        
        if self.well_tracker and self.config.enable_edge_detection:
            edge_status = self.well_tracker.get_edge_detection_status()
            if edge_status['edge_condition_satisfied']:
                logger.info(f"  Edge Detection: Satisfied At Frame {edge_status['edge_detection_frame']}")
                if edge_status['edge_circle_info']:
                    circle_info = edge_status['edge_circle_info']
                    logger.info(f"    Edge Circle: ({circle_info['circle']['x']:.1f}, {circle_info['circle']['y']:.1f})")
                    logger.info(f"    Line Slope: {circle_info['line_slope']:.3f}")
                    logger.info(f"    Distance: {circle_info['distance_to_line']:.1f} Pixels")
                logger.info(f"    Edge Detection Logic Disabled After Frame {edge_status['edge_detection_frame']}")
                if self.config.use_background_removal:
                    logger.info(f"    Background Removal Also Disabled After Frame {edge_status['edge_detection_frame']}")
            else:
                logger.info(f"  Edge Detection: Not Satisfied (Waiting For Edge Condition)")
            
            initial_layout_desc = "Row 2 Top, Row 1 Bottom" if self.config.initial_row_layout_flipped else "Row 1 Top, Row 2 Bottom" 
            logger.info(f"  Initial Layout: {initial_layout_desc}")
            
            if edge_status.get('phi_flip_history'):
                flip_count = len(edge_status['phi_flip_history'])
                final_flipped = edge_status.get('row_layout_flipped')
                final_layout_desc = "Row 2 Top, Row 1 Bottom" if final_flipped else "Row 1 Top, Row 2 Bottom"
                
                logger.info(f"  Row Layout Flips: {flip_count} Flips Detected")
                logger.info(f"  Final Layout: {final_layout_desc}")
                if edge_status.get('last_perpendicular_phi') is not None:
                    logger.info(f"  Last Phi: {edge_status['last_perpendicular_phi']:.1f}°")
                
                for i, flip in enumerate(edge_status['phi_flip_history'][-3:]):
                    flip_layout = "Row 2 Top, Row 1 Bottom" if flip['new_layout_flipped'] else "Row 1 Top, Row 2 Bottom"
                    logger.info(f"    Flip {i+max(0,flip_count-3)+1}: Frame {flip['frame']} - "
                          f"φ {flip['previous_phi']:.1f}° → {flip['current_phi']:.1f}° "
                          f"(Δ{flip['phi_difference']:.1f}°) → {flip_layout}")
            else:
                logger.info(f"  Row Layout Flips: None Detected")
                current_layout_desc = "Row 2 Top, Row 1 Bottom" if edge_status.get('row_layout_flipped') else "Row 1 Top, Row 2 Bottom"
                logger.info(f"  Final Layout: {current_layout_desc} (Same As Initial)")
                if edge_status.get('last_perpendicular_phi') is not None:
                    logger.info(f"  Last Phi: {edge_status['last_perpendicular_phi']:.1f}°")
            
            if edge_status.get('last_successful_frame') is not None:
                logger.info(f"  Last Successful Frame: {edge_status['last_successful_frame']} ({edge_status.get('num_successful_frame_wells', 0)} Wells)")
            else:
                logger.info(f"  Last Successful Frame: None (No Frames With All Wells Assigned)")
        elif self.well_tracker and not self.config.enable_edge_detection:
            logger.info(f"  Edge Detection: Disabled - Well Tracking Started Immediately")
            edge_status = self.well_tracker.get_edge_detection_status()
            if edge_status.get('last_successful_frame') is not None:
                logger.info(f"  Last Successful Frame: {edge_status['last_successful_frame']} ({edge_status.get('num_successful_frame_wells', 0)} Wells)")
        
        if self.motor_calibration and self.motor_calibration.is_calibrated:
            self._print_calibration_models()
    
    def _print_calibration_models(self):
        """Print the trained calibration model coefficients and parameters"""
        cal = self.motor_calibration
        
        logger.info("="*80)
        logger.info("Motor Calibration Models")
        logger.info("="*80)
        
        model_type = 'Polynomial' if cal.use_polynomial else 'Linear'
        logger.info(f"Model Type: {model_type} Ridge Regression (Pixel → Motor)")
        logger.info(f"Regularization Alpha: {cal.alpha}")
        logger.info(f"Training Samples: {len(cal.motor_history)}")
        logger.info(f"Uses Phi: {'Yes' if cal.use_phi else 'No'}")
        if cal.use_phi:
            logger.info(f"Phi Weight: {cal.phi_weight}")
        
        logger.info(f"Input Features: pixel_dx, pixel_dy")
        if cal.use_polynomial:
            logger.info(f"Polynomial Features: pixel_dx, pixel_dy, pixel_dx², pixel_dx*pixel_dy, pixel_dy²")
        
        logger.info(f"Model Equations:")
        logger.info(f"Input: Δpixel = [dx, dy]")
        if cal.use_polynomial:
            logger.info(f"Features: X = [dx, dy, dx², dx*dy, dy²]")
        else:
            logger.info(f"Features: X = [dx, dy]")
        
        axes = ['X', 'Y', 'Z']
        models = [cal.model_motor_x, cal.model_motor_y, cal.model_motor_z]
        
        if cal.use_phi:
            axes.append('Phi')
            models.append(cal.model_motor_phi)
        
        logger.info("-"*80)
        logger.info("Model Coefficients")
        logger.info("-"*80)
        
        for axis, model in zip(axes, models):
            score = cal.calibration_scores.get(f'motor_{axis.lower()}', 0.0)
            logger.info(f"Motor {axis} Model (R² = {score:.4f}):")
            logger.info(f"  Intercept: {model.intercept_:.6f}")
            
            if cal.use_polynomial:
                feature_names = ['pixel_dx', 'pixel_dy', 'pixel_dx²', 'pixel_dx*pixel_dy', 'pixel_dy²']
            else:
                feature_names = ['pixel_dx', 'pixel_dy']
            
            logger.info(f"  Coefficients:")
            for name, coef in zip(feature_names, model.coef_):
                logger.info(f"    {name:12}: {coef:12.6f}")
            
            equation_parts = [f"{model.intercept_:.6f}"]
            for name, coef in zip(feature_names, model.coef_):
                sign = "+" if coef >= 0 else ""
                equation_parts.append(f"{sign}{coef:.6f}*{name}")
            
            equation = " ".join(equation_parts)
            logger.info(f"  Equation: Δmotor_{axis.lower()} = {equation}")
        
        logger.info("-"*80)
        logger.info("Training Data Statistics")
        logger.info("-"*80)
        
        if cal.pixel_history and cal.motor_history:
            pixel_array = np.array(cal.pixel_history)
            motor_array = np.array(cal.motor_history)
            
            logger.info(f"Pixel Delta Statistics:")
            logger.info(f"  Mean: [{pixel_array.mean(axis=0)[0]:8.3f}, {pixel_array.mean(axis=0)[1]:8.3f}]")
            logger.info(f"  Std:  [{pixel_array.std(axis=0)[0]:8.3f}, {pixel_array.std(axis=0)[1]:8.3f}]")
            logger.info(f"  Min:  [{pixel_array.min(axis=0)[0]:8.3f}, {pixel_array.min(axis=0)[1]:8.3f}]")
            logger.info(f"  Max:  [{pixel_array.max(axis=0)[0]:8.3f}, {pixel_array.max(axis=0)[1]:8.3f}]")
            
            logger.info(f"Motor Delta Statistics:")
            motor_labels = ['X', 'Y', 'Z'] + (['Phi'] if cal.use_phi else [])
            for i, label in enumerate(motor_labels):
                logger.info(f"  Motor {label}:")
                logger.info(f"    Mean: {motor_array.mean(axis=0)[i]:10.6f}")
                logger.info(f"    Std:  {motor_array.std(axis=0)[i]:10.6f}")
                logger.info(f"    Min:  {motor_array.min(axis=0)[i]:10.6f}")
                logger.info(f"    Max:  {motor_array.max(axis=0)[i]:10.6f}")
        
        logger.info("-"*80)
        logger.info("Training Configuration")
        logger.info("-"*80)
        logger.info(f"Averaging Mode: {'Enabled' if cal.config.calibration_use_average_movement else 'Individual Wells'}")
        logger.info(f"Multi-Frame: {'Enabled' if cal.config.calibration_use_multi_frame else 'Sequential Only'}")
        if cal.config.calibration_use_multi_frame:
            logger.info(f"Pairing Strategy: {cal.config.calibration_pairing_strategy}")
        logger.info(f"Min Common Wells: {cal.config.calibration_min_common_wells}")
        
        stats = cal.training_stats
        logger.info(f"Training Data Generation:")
        logger.info(f"  Total Pairs Generated: {stats['total_pairs_generated']}")
        logger.info(f"  Unique Frame Pairs: {len(stats['unique_frame_pairs'])}")
        logger.info(f"  Avg Frame Gap: {stats['average_frame_gap']:.1f}")
        logger.info(f"  Avg Common Wells: {stats['average_common_wells']:.1f}")
        
        if stats['pixel_delta_std']['x']:
            avg_std_x = np.mean(stats['pixel_delta_std']['x'])
            avg_std_y = np.mean(stats['pixel_delta_std']['y'])
            logger.info(f"  Avg Pixel Delta Std: [{avg_std_x:.3f}, {avg_std_y:.3f}]")
        
        logger.info("="*80)


def main():
    """Main entry point with centralized configuration"""
    config = Config()
    system = WellTrackingSystem(config)
    system.run()


if __name__ == "__main__":
    main()