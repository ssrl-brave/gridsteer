#!/usr/bin/env python3
"""
Non-Persistent Line Analyzer - Handles analysis and processing without state persistence
"""

import json
import logging
import math
import os
import sys
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import linregress

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from skimage.feature import canny
from skimage.measure import ransac, EllipseModel
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks

import matplotlib
matplotlib.use('Agg')
plt.ioff()

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration parameters for the line analyzer"""
    output_dir: str = "output_videos"
    output_images_dir: str = "output_images"
    
    target_radius: int = 110
    radius_range: int = 10
    min_x_distance: int = 225
    min_y_distance: int = 225
    hough_num_peaks: int = 10
    hough_threshold: float = 0.15
    
    roundness_threshold: float = 0.92
    min_circles_required: int = 2
    use_average_mode: bool = True
    
    edge_sigma: float = 15.0
    edge_low_threshold: float = 0.15
    edge_high_threshold: float = 0.7
    
    line_hough_threshold: int = 80
    line_min_distance: int = 20
    line_min_angle: int = 80
    line_num_peaks: int = 10
    
    horizontal_tolerance_degrees: float = 15.0
    check_line_intersection: bool = True
    intersection_margin: int = 50
    
    backup_edge_sigma: float = 15.0
    backup_edge_low_threshold: float = 0.2
    backup_edge_high_threshold: float = 0.7
    
    hull_min_area: int = 200
    border_buffer: int = 5
    
    video_fps: int = 10
    save_video: bool = False
    save_individual_frames: bool = True
    display_frames: bool = False
    
    ellipse_margin_factor: float = 1.0
    inner_factor: float = 0.0
    use_annular_mask: bool = True
    min_edge_points: int = 0
    ransac_residual_threshold: float = 5.0
    ransac_max_trials: int = 100
    
    def get_radius_range(self) -> Tuple[int, int]:
        """Get the minimum and maximum radius for circle detection"""
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
class HorizontalLinePair:
    """Data structure for horizontal line pairs"""
    distance: float
    line1_y: float
    line2_y: float
    frame_number: int
    phi: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class AnalyzerState:
    """State maintained across analyzer calls"""
    frames_processed: int = 0
    min_distance_found: float = float('inf')
    best_frame_info: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'frames_processed': self.frames_processed,
            'min_distance_found': self.min_distance_found,
            'best_frame_info': self.best_frame_info
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnalyzerState':
        """Create from dictionary"""
        return cls(
            frames_processed=data.get('frames_processed', 0),
            min_distance_found=data.get('min_distance_found', float('inf')),
            best_frame_info=data.get('best_frame_info', None)
        )


def create_circular_mask(shape: Tuple[int, int], center: Tuple[float, float], 
                        outer_radius: float, inner_radius: float = 0) -> np.ndarray:
    """Create circular or annular mask"""
    y_indices, x_indices = np.ogrid[:shape[0], :shape[1]]
    distances_sq = (x_indices - center[0])**2 + (y_indices - center[1])**2
    
    if inner_radius > 0:
        return (distances_sq <= outer_radius**2) & (distances_sq >= inner_radius**2)
    else:
        return distances_sq <= outer_radius**2


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
            backup_lines_filt = self._filter_border_lines(backup_lines, edge.shape, border_buffer)
            return edge, ph, backup_lines_filt
        
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
                lines.append((xline[sel], yline[sel], ang, dist))
        
        return lines
    
    def _filter_border_lines(self, lines: List, shape: Tuple[int, int], border_buffer: int) -> List:
        """Filter out lines that are too close to image borders"""
        height, width = shape
        filtered_lines = []
        
        for line_data in lines:
            x_coords, y_coords, _, _ = line_data
            l = linregress( x_coords, y_coords)
            yi = l.intercept
            # check the y intercept to find border line
            if not np.isinf(yi) and abs(yi-height) <= border_buffer:
                continue

            # check the x intercept
            l_inv = linregress( y_coords, x_coords)
            xi = l_inv.intercept
            if not np.isinf(xi) and (abs(xi-width) <= border_buffer):
                continue
            #is_border_line = (
            #    np.any(x_coords <= border_buffer) or 
            #    np.any(x_coords >= width - border_buffer) or
            #    np.any(y_coords <= border_buffer) or 
            #    np.any(y_coords >= height - border_buffer)
            #)
            
            #if not is_border_line:
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


class HorizontalLineDetector:
    """Specialized detector for horizontal lines using convex hull approach"""
    
    def __init__(self, config: Config):
        self.config = config
        self.config.edge_sigma = 40
        self.circle_detector = CircleDetector(config)
        self.line_detector = LineDetector(config)
        self.contour_processor = ContourProcessor(config)
    
    def detect_horizontal_lines(self, img: np.ndarray) -> Tuple[List[Dict], float, Dict]:
        """Detect horizontal lines and return line parameters, minimum distance, and processing info"""
        # TODO separate edge finding and circle detection... 
        edge, circles = self.circle_detector.detect_circles(img)
        
        contour_coords, segments = self.contour_processor.extract_contour_coordinates(
            edge, remove_border_points=False)
        
        #if contour_coords is not None and segments is not None:
        #    contour_img, ph, lines = self.line_detector.detect_lines(
        #        contour_coords, img.shape, segments, backup_img=img)
        #    detection_method = "Convex Hull"
        #else:
        self.config
        contour_img, ph, lines = self.line_detector.detect_lines(
                None, img.shape, None, backup_img=img)
        detection_method = "Direct Edge Detection"
        
        horizontal_lines = self._extract_horizontal_line_params(lines, img.shape)
        min_distance = self._find_minimum_parallel_distance(horizontal_lines, img.shape)
        
        processing_info = {
            'edge_image': edge,
            'contour_coords': contour_coords,
            'segments': segments,
            'contour_img': contour_img,
            'detection_method': detection_method,
            'circles': circles,
            'lines_detected': len(lines)
        }
        
        return horizontal_lines, min_distance, processing_info
    
    def _extract_horizontal_line_params(self, lines: List, img_shape: Tuple) -> List[Dict]:
        """Extract parameters for horizontal lines from detected lines"""
        horizontal_threshold = np.deg2rad(self.config.horizontal_tolerance_degrees)
        horizontal_lines = []
        
        for line_data in lines:
            #if len(line_data) >= 4:
            #    x_coords, y_coords, angle, distance = line_data[0], line_data[1], line_data[2], line_data[3]
            #else:
            x_coords, y_coords = line_data[0], line_data[1]
            if len(x_coords) > 1 and len(y_coords) > 1:
                dx = x_coords[-1] - x_coords[0]
                dy = y_coords[-1] - y_coords[0]
                angle = np.arctan2(dy, dx)
                x_center, y_center = img_shape[1]/2, img_shape[0]/2
                distance = abs(np.mean(y_coords) - y_center)
            else:
                continue
            
            # Normalize angle to be between -pi/2 and pi/2
            normalized_angle = angle
            while normalized_angle > np.pi/2:
                normalized_angle -= np.pi
            while normalized_angle < -np.pi/2:
                normalized_angle += np.pi
            
            # Check if line is approximately horizontal
            abs_angle = abs(normalized_angle)
            if abs_angle <= horizontal_threshold: #or abs_angle >= (np.pi/2 - horizontal_threshold):
                y_intercept = np.mean(y_coords)
                
                horizontal_lines.append({
                    'angle': angle,
                    'distance': distance,
                    'y_intercept': y_intercept,
                    'normalized_angle': abs_angle,
                    'x_coords': x_coords,
                    'y_coords': y_coords
                })
        
        return horizontal_lines
    
    def _find_minimum_parallel_distance(self, horizontal_lines: List[Dict], img_shape: Tuple) -> float:
        """Find minimum distance between horizontal parallel lines that don't intersect in frame"""
        min_distance = float('inf')
        
        if len(horizontal_lines) >= 2:
            horizontal_lines.sort(key=lambda x: x['y_intercept'])
            
            for i in range(len(horizontal_lines)):
                for j in range(i + 1, len(horizontal_lines)):
                    line1 = horizontal_lines[i]
                    line2 = horizontal_lines[j]
                    
                    # Check if lines are approximately parallel
                    angle_diff = abs(line1['normalized_angle'] - line2['normalized_angle'])
                    if angle_diff <= np.deg2rad(10):
                        
                        lines_are_parallel = True
                        if self.config.check_line_intersection:
                            lines_are_parallel = not self._lines_intersect_in_frame_alt(line1, line2, img_shape)
                        
                        if lines_are_parallel:
                            distance = abs(line1['y_intercept'] - line2['y_intercept'])
                            if distance < min_distance:
                                min_distance = distance
        
        return min_distance if min_distance != float('inf') else float('inf')
    
    def _lines_intersect_in_frame_alt(self, line1: Dict, line2: Dict, img_shape: Tuple) -> bool:
        """Check if two lines intersect within the frame boundaries"""
        x_coords1 = line1.get('x_coords')
        y_coords1 = line1.get('y_coords') 
        x_coords2 = line2.get('x_coords')
        y_coords2 = line2.get('y_coords')
        
        if (x_coords1 is not None and y_coords1 is not None and 
            x_coords2 is not None and y_coords2 is not None and
            len(x_coords1) >= 2 and len(y_coords1) >= 2 and
            len(x_coords2) >= 2 and len(y_coords2) >= 2):
            
            try:
                # Fit lines to coordinates: y = m*x + b
                m1 = (y_coords1[-1] - y_coords1[0]) / (x_coords1[-1] - x_coords1[0]) if x_coords1[-1] != x_coords1[0] else 0
                b1 = y_coords1[0] - m1 * x_coords1[0]
                
                m2 = (y_coords2[-1] - y_coords2[0]) / (x_coords2[-1] - x_coords2[0]) if x_coords2[-1] != x_coords2[0] else 0
                b2 = y_coords2[0] - m2 * x_coords2[0]
                
                # Check if slopes are too similar (parallel)
                if abs(m1 - m2) < 0.01:
                    return False
                
                # Find intersection point
                x_intersect = (b2 - b1) / (m1 - m2)
                y_intersect = m1 * x_intersect + b1
                
                # Check if intersection is within frame
                height, width = img_shape
                margin = self.config.intersection_margin
                within_frame = (margin <= x_intersect <= width - margin and 
                               margin <= y_intersect <= height - margin)
                
                return within_frame
                
            except (ZeroDivisionError, IndexError):
                pass
        
        return False


class CircleDetector:
    """Handles circle detection using Hough transform"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect_circles(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Detect circles and return edge image and circle parameters"""
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


class LineAnalyzer:
    """Main non-persistent analyzer class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.horizontal_detector = HorizontalLineDetector(config)
        
        for dir_path in [config.output_dir, config.output_images_dir]:
            Path(dir_path).mkdir(exist_ok=True)
    
    def load_frame_data(self, frame_number: int, data_path: str) -> Tuple[Optional[MotorPosition], Optional[np.ndarray]]:
        """Load frame data from .npz files"""
        try:
            data = np.load(f"{data_path}test{frame_number}.npz")
            motor_pos = MotorPosition(
                x=float(data['x']),
                y=float(data['y']),
                z=float(data['z']),
                phi=float(data['phi'])
            )
            return motor_pos, data['sample']
        except Exception as e:
            return None, None
    
    def analyze_frame(self, frame_number: int, data_path: str, input_state: AnalyzerState) -> Tuple[dict, AnalyzerState]:
        """Analyze a single frame and return results with updated state"""
        motor_data, img = self.load_frame_data(frame_number, data_path)
        
        if motor_data is None or img is None:
            return {
                'success': False,
                'reason': f'Could Not Load Frame Data for Frame {frame_number}',
                'has_lines': False,
                'is_best': False
            }, input_state
        
        horizontal_lines, min_distance, processing_info = self.horizontal_detector.detect_horizontal_lines(img)
        
        updated_state = AnalyzerState(
            frames_processed=input_state.frames_processed + 1,
            min_distance_found=input_state.min_distance_found,
            best_frame_info=input_state.best_frame_info
        )
        
        # Create line pair info if valid lines found
        line_pair = None
        if min_distance != float('inf') and len(horizontal_lines) >= 2:
            horizontal_lines.sort(key=lambda x: x['y_intercept'])
            
            for i in range(len(horizontal_lines)):
                for j in range(i + 1, len(horizontal_lines)):
                    line1 = horizontal_lines[i]
                    line2 = horizontal_lines[j]
                    
                    distance = abs(line1['y_intercept'] - line2['y_intercept'])
                    if abs(distance - min_distance) < 1e-6:
                        line_pair = HorizontalLinePair(
                            distance=distance,
                            line1_y=line1['y_intercept'],
                            line2_y=line2['y_intercept'],
                            frame_number=frame_number,
                            phi=motor_data.phi
                        )
                        break
                if line_pair:
                    break
        
        # Check if this is the best frame
        is_best_frame = False
        if line_pair and line_pair.distance < updated_state.min_distance_found:
            updated_state.min_distance_found = line_pair.distance
            updated_state.best_frame_info = line_pair.to_dict()
            is_best_frame = True
        
        if self.config.save_individual_frames:
            try:
                self._save_frame_visualization(frame_number, img, motor_data, 
                                             horizontal_lines, processing_info, 
                                             line_pair, is_best_frame)
            except Exception as e:
                print(f"Warning: Could Not Save Frame Visualization: {e}", file=sys.stderr)
        
        result = {
            'success': True,
            'frame_number': frame_number,
            'phi': motor_data.phi,
            'has_lines': line_pair is not None,
            'distance': line_pair.distance if line_pair else float('inf'),
            'is_best': is_best_frame,
            'horizontal_lines_count': len(horizontal_lines),
            'detection_method': processing_info.get('detection_method', 'Unknown')
        }
        
        return result, updated_state
    
    def _save_frame_visualization(self, frame_number: int, img: np.ndarray, 
                                motor_data: MotorPosition, horizontal_lines: List[Dict], 
                                processing_info: Dict, line_pair: Optional[HorizontalLinePair],
                                is_best_frame: bool = False):
        """Save frame visualization with 2x2 subplot layout"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Frame {frame_number} Analysis (φ={motor_data.phi:.3f}°)', fontsize=16)
        
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        edge_img = processing_info.get('edge_image')
        if edge_img is not None:
            ax2.imshow(edge_img, cmap='gray')
        ax2.set_title('Edge Detection')
        ax2.axis('off')
        
        ax3.imshow(img, cmap='gray')
        contour_coords = processing_info.get('contour_coords')
        if contour_coords is not None:
            ax3.scatter(contour_coords[:, 0], contour_coords[:, 1], c='red', s=10)
        ax3.set_title(f'Hull ({processing_info.get("detection_method", "Unknown")})')
        ax3.axis('off')
        
        ax4.imshow(img, cmap='gray')
        for i, line_info in enumerate(horizontal_lines):
            y_intercept = line_info['y_intercept']
            ax4.axhline(y=y_intercept, color='lime' if i < 2 else 'cyan', 
                       linewidth=2, alpha=0.8)
            ax4.text(10, y_intercept, f'L{i+1}: {y_intercept:.1f}', 
                    color='white', fontweight='bold')
        
        if line_pair:
            mid_x = img.shape[1] / 2
            ax4.plot([mid_x, mid_x], [line_pair.line1_y, line_pair.line2_y], 
                    'yellow', linewidth=3)
            ax4.text(mid_x + 10, (line_pair.line1_y + line_pair.line2_y) / 2, 
                    f'{line_pair.distance:.1f}px', color='yellow', fontweight='bold')
        
        ax4.set_title(f'Horizontal Lines (Count: {len(horizontal_lines)})')
        ax4.axis('off')
        
        prefix = "BEST_" if is_best_frame else ""
        frame_path = Path(self.config.output_images_dir) / f"{prefix}frame_{frame_number}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Non-persistent line analyzer')
    parser.add_argument('frame_number', type=int, help='Frame number to analyze')
    parser.add_argument('data_path', type=str, help='Path to data directory')
    parser.add_argument('--state', type=str, help='JSON state data from persistent processor')
    return parser.parse_args()


def main():
    """Main entry point for non-persistent analyzer"""
    if len(sys.argv) == 1 or '--help' in sys.argv:
        print("Usage: python 1_non_persistent_processor.py <frame_number> <data_path> [--state <json_state>]")
        sys.exit(0)
    
    try:
        args = parse_arguments()
        
        if args.state:
            try:
                state_data = json.loads(args.state)
                input_state = AnalyzerState.from_dict(state_data)
            except json.JSONDecodeError:
                input_state = AnalyzerState()
        else:
            input_state = AnalyzerState()
        
        data_path = args.data_path
        if not data_path.endswith('/'):
            data_path += '/'
        
        config = Config()
        analyzer = LineAnalyzer(config)
        
        result, updated_state = analyzer.analyze_frame(args.frame_number, data_path, input_state)
        
        output = {
            'result': result,
            'state': updated_state.to_dict()
        }
        
        print(json.dumps(output))
        
    except ValueError:
        error_output = {
            'result': {
                'success': False, 
                'reason': 'Invalid Frame Number'
            },
            'state': AnalyzerState().to_dict()
        }
        print(json.dumps(error_output))
        sys.exit(1)
    except Exception as e:
        error_output = {
            'result': {
                'success': False, 
                'reason': f'Unexpected Error: {str(e)}'
            },
            'state': AnalyzerState().to_dict()
        }
        print(json.dumps(error_output), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
