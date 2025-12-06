#!/usr/bin/env python3
"""
Non-Persistent Line Analyzer - Handles analysis and processing without state persistence
"""

import json
import logging
import sys
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from rembg import remove
from PIL import Image

import matplotlib
matplotlib.use('Agg')
plt.ioff()

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration parameters for the line analyzer"""
    output_images_dir: str = "output_images"
    
    # Background removal settings
    use_background_removal: bool = True
    rembg_model: Optional[str] = None  # "birefnet-general-lite"  # None  # Model for rembg (e.g., 'u2net', 'silueta', etc.). If None, uses default

    # Outline tracing parameters
    outline_min_area: int = 500
    outline_approximation_epsilon: float = 2.0
    border_buffer: int = 5
    
    # Line detection parameters
    line_hough_threshold: int = 150  # 80
    line_min_distance: int = 20
    line_min_angle: int = 80
    line_num_peaks: int = 10
    
    horizontal_tolerance_degrees: float = 45.0
    check_line_intersection: bool = True
    intersection_margin: int = 0
    
    # Backup edge detection
    backup_edge_sigma: float = 15.0
    backup_edge_low_threshold: float = 0.2
    backup_edge_high_threshold: float = 0.7
    
    # Output settings
    save_individual_frames: bool = True
    
    # Logging settings
    verbose: bool = False
    log_dir: str = "logs"
    log_file: Optional[str] = None


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
        return asdict(self)


@dataclass
class AnalyzerState:
    """State maintained across analyzer calls"""
    frames_processed: int = 0
    min_distance_found: float = float('inf')
    best_frame_info: Optional[dict] = None
    
    def to_dict(self) -> dict:
        return {
            'frames_processed': self.frames_processed,
            'min_distance_found': self.min_distance_found,
            'best_frame_info': self.best_frame_info
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AnalyzerState':
        return cls(
            frames_processed=data.get('frames_processed', 0),
            min_distance_found=data.get('min_distance_found', float('inf')),
            best_frame_info=data.get('best_frame_info', None)
        )


class BackgroundRemover:
    """Handles background removal using rembg"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = None 
        
        # Create session once during initialization if model is specified
        if self.config.rembg_model is not None:
            try:
                from rembg import new_session
                self.session = new_session(model_name=self.config.rembg_model)
                logger.debug(f"Created rembg Session with Model: {self.config.rembg_model}")
            except Exception as e:
                logger.warning(f"Failed to Create rembg Session: {e} -- Will Use Default")
                self.session = None

    def remove_background(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove background from image using rembg and return processed image and mask"""
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
            
            # Use session if available, otherwise use default
            if self.session is not None:
                output = remove(pil_image, session=self.session)
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


class OutlineTracer:
    """Handles direct outline tracing from background-removed images"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def trace_external_outline(self, background_removed_img: np.ndarray, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List], bool]:
        """Trace the external outline of the largest object from background-removed image
        
        Returns:
            - outline_coords: The original unfiltered coordinates
            - segments: List of coordinate segments
            - is_closed: Whether the contour is closed
        """
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.debug("No Contours Found In Background-Removed Image")
                return None, None, False
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            logger.debug(f"Largest Contour Area: {largest_area}, Min Required: {self.config.outline_min_area}")
            
            if largest_area < self.config.outline_min_area:
                logger.debug("Largest Contour Too Small")
                return None, None, False
            
            # Approximate the contour to reduce noise
            epsilon = self.config.outline_approximation_epsilon
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            outline_coords = approx_contour.reshape(-1, 2)
            
            segments = [outline_coords]
            is_closed = True  # Contour from findContours is always closed
            
            logger.debug(f"Successfully Traced Outline With {len(outline_coords)} Points")
            logger.debug("Contour Will Be Drawn As Closed")
            
            return outline_coords, segments, is_closed
            
        except Exception as e:
            logger.warning(f"Outline Tracing Failed: {e}")
            return None, None, False


class LineDetector:
    """Handles line detection using Hough transform"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect_lines(self, contour_coords: Optional[np.ndarray], img_shape: Tuple, 
                    segments: Optional[List] = None,
                    backup_img: Optional[np.ndarray] = None,
                    is_closed: bool = True) -> Tuple:
        """Detect lines using Hough transform"""
        threshold = self.config.line_hough_threshold
        min_distance = self.config.line_min_distance
        min_angle = self.config.line_min_angle
        num_peaks = self.config.line_num_peaks
        border_buffer = self.config.border_buffer
        
        primary_lines = []
        contour_img = None
        ph = None
        
        logger.debug(f"Line Detection - Threshold: {threshold}, Min Distance: {min_distance}")
        
        # Try outline coordinates first
        if contour_coords is not None and len(contour_coords) > 0 and img_shape is not None:
            height, width = img_shape
            contour_img = np.zeros((height, width), dtype=np.uint8)
            
            if segments is not None:
                for segment in segments:
                    if len(segment) > 1:
                        pts = segment.astype(np.int32)
                        cv2.polylines(contour_img, [pts], isClosed=is_closed, color=255, thickness=2)
            
            # Apply border filtering to the drawn image
            filtered_contour_img = self._apply_border_filtering(contour_img, border_buffer)
            
            primary_lines = self._extract_lines_from_image(filtered_contour_img, threshold, 
                                                         min_distance, min_angle, num_peaks)
            if primary_lines:
                logger.debug(f"Found {len(primary_lines)} Lines From Outline After Border Filtering")
                return filtered_contour_img, ph, primary_lines
        
        # Fallback to edge detection
        if backup_img is not None:
            logger.debug("Using Backup Edge Detection For Line Finding")
            edge = canny(backup_img, sigma=self.config.backup_edge_sigma, 
                        low_threshold=self.config.backup_edge_low_threshold, 
                        high_threshold=self.config.backup_edge_high_threshold, 
                        use_quantiles=True)
            
            filtered_edge = self._apply_border_filtering(edge, border_buffer)
            
            backup_lines = self._extract_lines_from_image(filtered_edge, threshold, 
                                                        min_distance, min_angle, num_peaks)
            logger.debug(f"Found {len(backup_lines)} Lines From Backup Edge Detection After Border Filtering")
            return filtered_edge, ph, backup_lines
        
        return None, None, []
    
    def _apply_border_filtering(self, img: np.ndarray, border_buffer: int) -> np.ndarray:
        """Apply border filtering by masking out border regions of the image"""
        if border_buffer <= 0:
            return img
        
        height, width = img.shape
        filtered_img = img.copy()
        
        # Mask out border regions
        filtered_img[:border_buffer, :] = 0  # Top
        filtered_img[height-border_buffer:, :] = 0  # Bottom
        filtered_img[:, :border_buffer] = 0  # Left
        filtered_img[:, width-border_buffer:] = 0  # Right
        
        logger.debug(f"Applied Border Filtering With Buffer: {border_buffer}px")
        return filtered_img
    
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


class HorizontalLineDetector:
    """Specialized detector for horizontal lines"""
    
    def __init__(self, config: Config):
        self.config = config
        self.line_detector = LineDetector(config)
        self.outline_tracer = OutlineTracer(config)
    
    def detect_horizontal_lines(self, img: np.ndarray, background_removed_img: Optional[np.ndarray] = None, 
                              mask: Optional[np.ndarray] = None) -> Tuple[List[Dict], float, Dict]:
        """Detect horizontal lines and return line parameters, minimum distance, and processing info"""
        logger.debug("Starting Horizontal Line Detection")
        
        detection_method = "Unknown"
        contour_coords = None
        segments = None
        contour_img = None
        is_closed = True
        
        # Use direct outline tracing if background removal was successful
        if (self.config.use_background_removal and background_removed_img is not None and 
            mask is not None):
            logger.debug("Using Direct Outline Tracing From Background-Removed Image")
            
            contour_coords, segments, is_closed = self.outline_tracer.trace_external_outline(
                background_removed_img, mask)
            
            if contour_coords is not None and segments is not None:
                contour_img, ph, lines = self.line_detector.detect_lines(
                    contour_coords, img.shape, segments, backup_img=img, is_closed=is_closed)
                detection_method = "Direct Outline Tracing (Closed Contour)"
            else:
                logger.debug("Outline Tracing Failed - Falling Back To Edge Detection")
                contour_img, ph, lines = self.line_detector.detect_lines(
                    None, img.shape, None, backup_img=img)
                detection_method = "Edge Detection (Outline Fallback)"
        else:
            logger.debug("Using Traditional Edge Detection Method")
            contour_img, ph, lines = self.line_detector.detect_lines(
                None, img.shape, None, backup_img=img)
            detection_method = "Traditional Edge Detection"
        
        logger.debug(f"Detection Method: {detection_method}, Lines Found: {len(lines)}")
        
        all_horizontal_lines = self._extract_horizontal_line_params(lines, img.shape)
        horizontal_lines = self._filter_external_horizontal_lines(all_horizontal_lines)
        min_distance = self._find_minimum_parallel_distance(horizontal_lines, img.shape)
        
        logger.debug(f"All Horizontal Lines: {len(all_horizontal_lines)}, External Lines: {len(horizontal_lines)}, Min Distance: {min_distance}")
        
        processing_info = {
            'edge_image': contour_img,
            'contour_coords': contour_coords,
            'segments': segments,
            'contour_img': contour_img,
            'detection_method': detection_method,
            'lines_detected': len(lines),
            'all_horizontal_lines': len(all_horizontal_lines),
            'external_horizontal_lines': len(horizontal_lines),
            'is_closed_contour': is_closed
        }
        
        return horizontal_lines, min_distance, processing_info
    
    def _filter_external_horizontal_lines(self, horizontal_lines: List[Dict]) -> List[Dict]:
        """Filter horizontal lines to keep only the external-most ones (smallest and largest y-intercepts)"""
        if len(horizontal_lines) <= 2:
            return horizontal_lines
        
        sorted_lines = sorted(horizontal_lines, key=lambda x: x['y_intercept'])
        external_lines = [sorted_lines[0], sorted_lines[-1]]
        
        logger.debug(f"Filtered {len(horizontal_lines)} Horizontal Lines Down To {len(external_lines)} External Lines")
        logger.debug(f"Top Line Y-Intercept: {external_lines[0]['y_intercept']:.1f}, Bottom Line Y-Intercept: {external_lines[1]['y_intercept']:.1f}")
        
        return external_lines
    
    def _extract_horizontal_line_params(self, lines: List, img_shape: Tuple) -> List[Dict]:
        """Extract parameters for horizontal lines from detected lines"""
        horizontal_threshold = np.deg2rad(self.config.horizontal_tolerance_degrees)
        horizontal_lines = []
        
        for line_data in lines:
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
            if abs_angle <= horizontal_threshold:
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
                    if angle_diff <= np.deg2rad(self.config.horizontal_tolerance_degrees):
                        
                        lines_are_parallel = True
                        if self.config.check_line_intersection:
                            lines_are_parallel = not self._lines_intersect_in_frame(line1, line2, img_shape)
                        
                        if lines_are_parallel:
                            distance = abs(line1['y_intercept'] - line2['y_intercept'])
                            if distance < min_distance:
                                min_distance = distance
        
        return min_distance if min_distance != float('inf') else float('inf')
    
    def _lines_intersect_in_frame(self, line1: Dict, line2: Dict, img_shape: Tuple) -> bool:
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


class LineAnalyzer:
    """Main non-persistent analyzer class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.horizontal_detector = HorizontalLineDetector(config)
        self.background_remover = BackgroundRemover(config) if config.use_background_removal else None
        
        for dir_path in [config.output_images_dir]:
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
            logger.error(f"Failed To Load Frame {frame_number}: {e}")
            return None, None
    
    def analyze_frame(self, frame_number: int, data_path: str, input_state: AnalyzerState) -> Tuple[dict, AnalyzerState]:
        """Analyze a single frame and return results with updated state"""
        logger.debug(f"Analyzing Frame {frame_number}")
        motor_data, img = self.load_frame_data(frame_number, data_path)
        
        if motor_data is None or img is None:
            return {
                'success': False,
                'reason': f'Could Not Load Frame Data For Frame {frame_number}',
                'has_lines': False,
                'is_best': False
            }, input_state
        
        # Apply background removal if enabled
        processed_img = img
        mask = None
        background_removed = False
        if self.config.use_background_removal and self.background_remover is not None:
            logger.debug("Applying Background Removal")
            processed_img, mask = self.background_remover.remove_background(img)
            background_removed = True
        
        horizontal_lines, min_distance, processing_info = self.horizontal_detector.detect_horizontal_lines(
            img, processed_img if background_removed else None, mask)
        
        processing_info['background_removed'] = background_removed
        processing_info['original_image'] = img
        processing_info['processed_image'] = processed_img
        processing_info['mask'] = mask
        
        updated_state = AnalyzerState(
            frames_processed=input_state.frames_processed + 1,
            min_distance_found=input_state.min_distance_found,
            best_frame_info=input_state.best_frame_info
        )
        
        # Create line pair if valid lines found
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
            logger.info(f"New Best Frame: {frame_number} With Distance {line_pair.distance:.2f}px At Phi={motor_data.phi:.6f}")
        
        if self.config.save_individual_frames:
            try:
                self._save_frame_visualization(frame_number, img, processed_img, motor_data, 
                                             horizontal_lines, processing_info, 
                                             line_pair, is_best_frame, background_removed)
            except Exception as e:
                logger.warning(f"Could Not Save Frame Visualization: {e}")
        
        result = {
            'success': True,
            'frame_number': frame_number,
            'phi': motor_data.phi,
            'has_lines': line_pair is not None,
            'distance': line_pair.distance if line_pair else float('inf'),
            'is_best': is_best_frame,
            'all_horizontal_lines_count': processing_info.get('all_horizontal_lines', 0),
            'external_horizontal_lines_count': len(horizontal_lines),
            'detection_method': processing_info.get('detection_method', 'Unknown'),
            'background_removed': background_removed
        }
        
        return result, updated_state
    
    def _save_frame_visualization(self, frame_number: int, original_img: np.ndarray, 
                                    processed_img: np.ndarray, motor_data: MotorPosition, 
                                    horizontal_lines: List[Dict], processing_info: Dict, 
                                    line_pair: Optional[HorizontalLinePair],
                                    is_best_frame: bool = False, background_removed: bool = False):
        """Save frame visualization with 2x3 subplot layout"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        fig.suptitle(f'Frame {frame_number} Analysis (φ={motor_data.phi:.3f}°)', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(original_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(processed_img, cmap='gray')
        axes[0, 1].set_title('Processed Image (After Background Removal)' if background_removed else 'Processed Image (No Background Removal)')
        axes[0, 1].axis('off')
        
        # Mask visualization
        mask = processing_info.get('mask')
        if mask is not None:
            axes[0, 2].imshow(mask, cmap='gray')
            axes[0, 2].set_title('Background Removal Mask')
        else:
            axes[0, 2].imshow(processed_img, cmap='gray')
            axes[0, 2].set_title('No Mask Available')
        axes[0, 2].axis('off')
        
        # Outline visualization
        axes[1, 0].imshow(processed_img, cmap='gray')
        contour_coords = processing_info.get('contour_coords')
        if contour_coords is not None:
            axes[1, 0].scatter(contour_coords[:, 0], contour_coords[:, 1], c='red', s=10, alpha=0.7)
            if len(contour_coords) > 2:
                is_closed_contour = processing_info.get('is_closed_contour', True)
                if is_closed_contour:
                    closed_coords = np.vstack([contour_coords, contour_coords[0]])
                    axes[1, 0].plot(closed_coords[:, 0], closed_coords[:, 1], 'red', linewidth=2, alpha=0.8)
                else:
                    axes[1, 0].plot(contour_coords[:, 0], contour_coords[:, 1], 'red', linewidth=2, alpha=0.8)
        axes[1, 0].set_title(f'{processing_info.get("detection_method", "Unknown")}')
        axes[1, 0].axis('off')
        
        # Processing visualization
        edge_img = processing_info.get('contour_img')
        if edge_img is not None:
            axes[1, 1].imshow(edge_img, cmap='gray')
            axes[1, 1].set_title('Detection Processing Image (After Border Filtering)')
        else:
            axes[1, 1].imshow(processed_img, cmap='gray')
            axes[1, 1].set_title('No Processing Image Available')
        axes[1, 1].axis('off')
        
        # Line detection results
        axes[1, 2].imshow(processed_img, cmap='gray')
        
        all_lines_count = processing_info.get('all_horizontal_lines', 0)
        if all_lines_count > len(horizontal_lines):
            axes[1, 2].text(10, 30, f'Total Detected: {all_lines_count}, Showing: {len(horizontal_lines)} External', 
                        color='white', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        
        # Draw the detected lines
        for i, line_info in enumerate(horizontal_lines):
            x_coords = line_info.get('x_coords')
            y_coords = line_info.get('y_coords')
            y_intercept = line_info['y_intercept']
            
            color = 'lime' if i == 0 else 'red'
            label = 'Top' if i == 0 else 'Bottom'
            
            if x_coords is not None and y_coords is not None and len(x_coords) > 0 and len(y_coords) > 0:
                axes[1, 2].plot(x_coords, y_coords, color=color, linewidth=3, alpha=0.9, label=label)
                mid_idx = len(x_coords) // 2
                axes[1, 2].text(x_coords[mid_idx], y_coords[mid_idx] + 15, 
                            f'{label}: y≈{y_intercept:.1f}px', 
                            color='white', fontweight='bold', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        
        if line_pair:
            mid_x = processed_img.shape[1] / 2
            axes[1, 2].plot([mid_x, mid_x], [line_pair.line1_y, line_pair.line2_y], 
                        'yellow', linewidth=4, alpha=0.9)
            axes[1, 2].text(mid_x + 10, (line_pair.line1_y + line_pair.line2_y) / 2, 
                        f'Distance:\n{line_pair.distance:.1f}px', color='yellow', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        
        axes[1, 2].set_title(f'External Horizontal Lines (Showing: {len(horizontal_lines)} of {all_lines_count} Detected)')
        axes[1, 2].axis('off')
        
        prefix = "CURRENT_BEST_" if is_best_frame else ""
        frame_path = Path(self.config.output_images_dir) / f"{prefix}frame_{frame_number}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved Visualization To {frame_path}")


def setup_logging(config: Config):
    """Setup logging configuration"""
    if config.verbose:
        if config.log_file:
            log_file = Path(config.log_file)
            
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='a'),
                    logging.NullHandler()
                ]
            )
            logger.info(f"Non-Persistent Analyzer - Appending To Existing Log: {log_file}")
        else:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"line_analyzer_{timestamp}.log"
            
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.NullHandler()
                ]
            )
            logger.info(f"Non-Persistent Analyzer - Created New Log: {log_file}")
    else:
        logging.basicConfig(level=logging.CRITICAL)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Non-Persistent Line Analyzer With Background Removal, Direct Outline Tracing, And External Horizontal Line Filtering')
    parser.add_argument('frame_number', type=int, help='Frame Number To Analyze')
    parser.add_argument('data_path', type=str, help='Path To Data Directory')
    parser.add_argument('--state', type=str, help='JSON State Data From Persistent Processor')
    parser.add_argument('--verbose', action='store_true', help='Enable Verbose Logging')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory For Log Files')
    parser.add_argument('--log-file', type=str, help='Specific Log File Path To Append To')
    parser.add_argument('--output-dir', type=str, default='output_images', help='Directory For Output Visualization Images')
    return parser.parse_args()


def main():
    """Main entry point for non-persistent analyzer"""
    if len(sys.argv) == 1 or '--help' in sys.argv:
        print("Usage: python 1_non_persistent_processor.py <frame_number> <data_path> [--state <json_state>] [--verbose] [--log-dir <dir>] [--log-file <path>] [--output-dir <dir>]")
        sys.exit(0)
    
    try:
        args = parse_arguments()
        
        config = Config()
        config.verbose = args.verbose
        config.log_dir = args.log_dir
        config.log_file = args.log_file
        config.output_images_dir = args.output_dir
        
        setup_logging(config)
        
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
