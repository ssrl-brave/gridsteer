#!/usr/bin/env python3
"""
Non-Persistent Tray Width Analyzer - Handles analysis and processing without state persistence
"""

import json
import logging
import sys
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d

plt.ioff()

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration parameters for the tray width analyzer"""
    output_images_dir: str = "output_images_1"

    # Output settings
    save_individual_frames: bool = True

    # Detection parameters
    dark_percentile: float = 25.0
    num_perpendicular_lines: int = 5

    # Smoothing parameters
    smoothing_sigma: float = 5.0  # Gaussian filter sigma (larger = smoother curves)

    # Curvature detection parameters
    curvature_percentile: float = 85.0  # Percentile for adaptive curvature threshold
    min_curvature_threshold: float = 0.0001  # Minimum curvature threshold to avoid noise sensitivity

    # Logging settings
    verbose: bool = False
    log_dir: str = "logs_1"
    log_file: Optional[str] = None


@dataclass
class MotorPosition:
    """Motor position data"""
    x: float
    y: float
    z: float
    phi: float


@dataclass
class TrayWidthMeasurement:
    """Data structure for tray width measurements"""
    avg_width: float
    widths: List[float]
    frame_number: int
    phi: float

    def to_dict(self) -> dict:
        """Convert to dictionary with JSON-serializable types"""
        return {
            'avg_width': float(self.avg_width),
            'widths': [float(w) for w in self.widths],
            'frame_number': int(self.frame_number),
            'phi': float(self.phi)
        }


@dataclass
class AnalyzerState:
    """State maintained across analyzer calls"""
    frames_processed: int = 0
    min_width_found: float = float('inf')
    best_frame_info: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary with JSON-serializable types"""
        return {
            'frames_processed': int(self.frames_processed),
            'min_width_found': float(self.min_width_found),
            'best_frame_info': self.best_frame_info
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AnalyzerState':
        return cls(
            frames_processed=data.get('frames_processed', 0),
            min_width_found=data.get('min_width_found', float('inf')),
            best_frame_info=data.get('best_frame_info', None)
        )


class TrayWidthDetector:
    """Detector for tray width using DBSCAN clustering + PCA and curvature-based trough analysis"""

    def __init__(self, config: Config):
        self.config = config

    def smooth_intensity_profile(self, intensities: np.ndarray, sigma: float = 10.0) -> np.ndarray:
        """
        Smooth intensity profile using Gaussian filter.

        Args:
            intensities: Raw intensity values
            sigma: Standard deviation of Gaussian kernel (larger = smoother)

        Returns:
            Smoothed intensity values
        """
        if len(intensities) < 3:
            return intensities

        try:
            smoothed = gaussian_filter1d(intensities, sigma=sigma, mode='nearest')
            return smoothed
        except Exception as e:
            logger.warning(f"Gaussian Smoothing Failed: {e}, Returning Original Intensities")
            return intensities

    def detect_tray_width(self, img: np.ndarray, dark_percentile: float = 30.0,
                          num_perpendicular_lines: int = 5) -> Tuple[float, Dict]:
        """
        Detect tray width using DBSCAN clustering, PCA, and curvature analysis.

        Args:
            img: Input grayscale image
            dark_percentile: Percentile threshold for dark point detection (0-100)
            num_perpendicular_lines: Number of perpendicular lines to sample

        Returns:
            avg_width: Average tray width across perpendicular lines
            debug_info: Dictionary with debug information
        """
        # Normalize image to 0-1 range
        if img.dtype == np.uint8:
            img_normalized = img.astype(float) / 255.0
        else:
            img_normalized = (img - img.min()) / (img.max() - img.min())

        height, width = img_normalized.shape

        # Calculate adaptive threshold based on percentile
        intensity_threshold = np.percentile(img_normalized, dark_percentile)
        logger.debug(f"Adaptive Threshold at {dark_percentile}th Percentile: {intensity_threshold:.3f}")

        # Get dark points below threshold
        dark_mask = img_normalized < intensity_threshold
        dark_points_yx = np.column_stack(np.where(dark_mask))

        if len(dark_points_yx) < 10:
            logger.warning(f"Not Enough Dark Points Found (Only {len(dark_points_yx)}) With Percentile {dark_percentile}")
            return float('inf'), {'error': 'Not enough dark points found'}

        # Use DBSCAN clustering to find the largest cluster
        y_coords = dark_points_yx[:, 0]
        x_coords = dark_points_yx[:, 1]

        eps = 15  # Maximum distance between points in a cluster
        min_samples = 50  # Minimum points to form a cluster

        logger.debug(f"Running DBSCAN Clustering With eps={eps}px, min_samples={min_samples}")

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(dark_points_yx)
        labels = clustering.labels_

        # Find the largest cluster (excluding noise)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if len(unique_labels) == 0:
            logger.warning("DBSCAN Found No Clusters, Using All Points")
            cluster_mask = np.ones(len(dark_points_yx), dtype=bool)
            largest_cluster_size = len(dark_points_yx)
        else:
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
            largest_cluster_size = cluster_sizes[largest_cluster_label]

            cluster_mask = labels == largest_cluster_label

            logger.debug(f"DBSCAN Found {len(unique_labels)} Clusters, Keeping Largest: {largest_cluster_size}/{len(dark_points_yx)} ({100*largest_cluster_size/len(dark_points_yx):.1f}%)")

        # Use largest cluster for PCA
        clustered_points = dark_points_yx[cluster_mask]
        y_coords = clustered_points[:, 0]
        x_coords = clustered_points[:, 1]

        # Fit PCA to find tray orientation
        logger.debug(f"Fitting PCA to {len(clustered_points)} Points From Largest Cluster")

        mean_x = np.mean(x_coords)
        mean_y = np.mean(y_coords)
        centered_points = np.column_stack([x_coords - mean_x, y_coords - mean_y])

        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        principal_axis_idx = np.argmax(eigenvalues)
        principal_vector = eigenvectors[:, principal_axis_idx]

        vx, vy = principal_vector
        if abs(vx) < 1e-6:
            m_parallel = 1e6
        else:
            m_parallel = vy / vx

        b_parallel = mean_y - m_parallel * mean_x

        logger.debug(f"PCA Fitted Parallel Line: y = {m_parallel:.4f}*x + {b_parallel:.4f}")

        # Calculate perpendicular slope
        if abs(m_parallel) < 1e-6:
            m_perpendicular = 1e6
        else:
            m_perpendicular = -1.0 / m_parallel

        logger.debug(f"Perpendicular Slope: {m_perpendicular:.4f}")

        # Generate evenly spaced perpendicular lines
        x_min, x_max = x_coords.min(), x_coords.max()

        margin = 0.1
        x_positions = np.linspace(x_min + (x_max - x_min) * margin,
                                  x_max - (x_max - x_min) * margin,
                                  num_perpendicular_lines)

        logger.debug(f"Placing {num_perpendicular_lines} Perpendicular Lines at x Positions: {x_positions}")

        # Measure width along each perpendicular line
        widths = []
        perpendicular_lines = []

        for i, x_pos in enumerate(x_positions):
            y_pos = m_parallel * x_pos + b_parallel

            # Sample along the perpendicular line
            if abs(m_perpendicular) > 1:
                y_samples = np.arange(0, height, 0.5)
                x_samples = x_pos + (y_samples - y_pos) / m_perpendicular
            else:
                x_samples = np.arange(0, width, 0.5)
                y_samples = y_pos + m_perpendicular * (x_samples - x_pos)

            # Keep only points within image bounds
            valid_mask = (x_samples >= 0) & (x_samples < width) & (y_samples >= 0) & (y_samples < height)
            x_samples = x_samples[valid_mask]
            y_samples = y_samples[valid_mask]

            if len(x_samples) < 2:
                logger.warning(f"Perpendicular Line {i} Has Insufficient Valid Points")
                continue

            # Sample intensities using nearest-neighbor
            x_int = np.clip(x_samples.astype(int), 0, width - 1)
            y_int = np.clip(y_samples.astype(int), 0, height - 1)
            intensities_original = img_normalized[y_int, x_int]

            # Apply Gaussian smoothing
            intensities_smoothed = self.smooth_intensity_profile(intensities_original, sigma=self.config.smoothing_sigma)
            intensities = intensities_smoothed

            # Find deepest trough
            dark_regions = intensities < intensity_threshold

            # Find continuous dark regions
            transitions = np.diff(np.concatenate([[False], dark_regions, [False]]).astype(int))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]

            if len(starts) == 0:
                logger.debug(f"Perpendicular Line {i}: No Trough Found")
                continue

            # Find deepest trough
            deepest_trough_idx = None
            deepest_min_intensity = float('inf')

            for trough_idx, (start, end) in enumerate(zip(starts, ends)):
                trough_intensities = intensities[start:end]
                min_intensity = np.min(trough_intensities)
                if min_intensity < deepest_min_intensity:
                    deepest_min_intensity = min_intensity
                    deepest_trough_idx = trough_idx

            deepest_start = starts[deepest_trough_idx]
            deepest_end = ends[deepest_trough_idx]

            # Find minimum point within deepest trough
            trough_intensities = intensities[deepest_start:deepest_end]
            min_idx_relative = np.argmin(trough_intensities)
            min_idx = deepest_start + min_idx_relative

            # Calculate derivatives for curvature-based detection
            first_derivative = np.gradient(intensities)
            second_derivative = np.gradient(first_derivative)

            # Adaptive curvature threshold
            trough_curvature = np.abs(second_derivative[deepest_start:deepest_end])
            curvature_threshold = np.percentile(trough_curvature, self.config.curvature_percentile)
            curvature_threshold = max(curvature_threshold, self.config.min_curvature_threshold)

            # Search left from minimum
            width_start = min_idx
            for i in range(min_idx - 1, -1, -1):
                if intensities[i] >= intensity_threshold:
                    break
                if np.abs(second_derivative[i]) > curvature_threshold:
                    width_start = i + 1
                    break
                width_start = i

            # Search right from minimum
            width_end = min_idx
            for i in range(min_idx + 1, len(intensities)):
                if intensities[i] >= intensity_threshold:
                    break
                if np.abs(second_derivative[i]) > curvature_threshold:
                    width_end = i
                    break
                width_end = i + 1

            width_along_line = width_end - width_start

            widths.append(width_along_line)
            perpendicular_lines.append({
                'x_samples': x_samples,
                'y_samples': y_samples,
                'intensities': intensities_smoothed,
                'intensities_original': intensities_original,
                'first_derivative': first_derivative,
                'second_derivative': second_derivative,
                'width': width_along_line,
                'x_pos': x_pos,
                'y_pos': y_pos,
                'trough_start': deepest_start,
                'trough_end': deepest_end,
                'width_start': width_start,
                'width_end': width_end,
                'min_idx': min_idx,
                'min_intensity': deepest_min_intensity,
                'curvature_threshold': curvature_threshold,
                'num_troughs': len(starts)
            })

            logger.debug(f"Perpendicular Line {i}: Deepest Trough Width = {width_along_line:.1f} Pixels "
                        f"(Min Intensity: {deepest_min_intensity:.3f}, Curvature Threshold: {curvature_threshold:.6f}, {len(starts)} Trough(s) Found)")

        if len(widths) == 0:
            logger.warning("Could Not Measure Any Widths")
            return float('inf'), {'error': 'Could not measure widths'}

        avg_width = np.mean(widths)
        logger.info(f"Average Tray Width: {avg_width:.2f} Pixels (Widths: {widths})")

        debug_info = {
            'parallel_line_slope': m_parallel,
            'parallel_line_intercept': b_parallel,
            'perpendicular_slope': m_perpendicular,
            'widths': widths,
            'avg_width': avg_width,
            'perpendicular_lines': perpendicular_lines,
            'dark_points_count': len(dark_points_yx),
            'dark_points_yx': dark_points_yx,
            'cluster_mask': cluster_mask,
            'cluster_labels': labels,
            'largest_cluster_size': largest_cluster_size,
            'clustered_points': clustered_points,
            'dark_percentile': dark_percentile,
            'intensity_threshold': intensity_threshold,
            'img_normalized': img_normalized
        }

        return avg_width, debug_info


class LineAnalyzer:
    """Main non-persistent analyzer class"""

    def __init__(self, config: Config):
        self.config = config
        self.tray_width_detector = TrayWidthDetector(config)

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
            logger.error(f"Failed to Load Frame {frame_number}: {e}")
            return None, None
    
    def analyze_frame(self, frame_number: int, data_path: str, input_state: AnalyzerState) -> Tuple[dict, AnalyzerState]:
        """Analyze a single frame and return results with updated state"""
        logger.debug(f"Analyzing Frame {frame_number}")
        motor_data, img = self.load_frame_data(frame_number, data_path)

        if motor_data is None or img is None:
            return {
                'success': False,
                'reason': f'Could not Load Frame Data for Frame {frame_number}',
                'has_measurement': False,
                'is_best': False
            }, input_state

        # Detect tray width
        avg_width, detection_info = self.tray_width_detector.detect_tray_width(
            img, dark_percentile=self.config.dark_percentile,
            num_perpendicular_lines=self.config.num_perpendicular_lines)

        updated_state = AnalyzerState(
            frames_processed=input_state.frames_processed + 1,
            min_width_found=input_state.min_width_found,
            best_frame_info=input_state.best_frame_info
        )

        # Create width measurement if valid
        width_measurement = None
        if avg_width != float('inf') and 'error' not in detection_info:
            width_measurement = TrayWidthMeasurement(
                avg_width=avg_width,
                widths=detection_info.get('widths', []),
                frame_number=frame_number,
                phi=motor_data.phi
            )

        # Check if this is the best frame
        is_best_frame = False
        if width_measurement and width_measurement.avg_width < updated_state.min_width_found:
            updated_state.min_width_found = width_measurement.avg_width
            updated_state.best_frame_info = width_measurement.to_dict()
            is_best_frame = True
            logger.info(f"New Best Frame: {frame_number} With Average Width {width_measurement.avg_width:.2f}px at Phi={motor_data.phi:.6f}")

        if self.config.save_individual_frames:
            try:
                self._save_frame_visualization(frame_number, img, motor_data,
                                             detection_info, width_measurement, is_best_frame)
            except Exception as e:
                logger.warning(f"Could Not Save Frame Visualization: {e}")

        result = {
            'success': True,
            'frame_number': frame_number,
            'phi': motor_data.phi,
            'has_measurement': width_measurement is not None,
            'width': width_measurement.avg_width if width_measurement else float('inf'),
            'is_best': is_best_frame,
            'detection_method': 'DBSCAN+PCA + Curvature-Based Width Detection'
        }

        return result, updated_state
    
    def _save_frame_visualization(self, frame_number: int, original_img: np.ndarray,
                                    motor_data: MotorPosition, detection_info: Dict,
                                    width_measurement: Optional[TrayWidthMeasurement],
                                    is_best_frame: bool = False):
        """Save frame visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        fig.suptitle(f'Frame {frame_number} - DBSCAN+PCA + Curvature-Based Width Detection (φ={motor_data.phi:.3f}°)', fontsize=16)

        img_normalized = detection_info.get('img_normalized', original_img)
        dark_points_yx = detection_info.get('dark_points_yx')
        cluster_mask = detection_info.get('cluster_mask')
        clustered_points = detection_info.get('clustered_points')
        largest_cluster_size = detection_info.get('largest_cluster_size', 0)
        m_parallel = detection_info.get('parallel_line_slope')
        b_parallel = detection_info.get('parallel_line_intercept')
        perpendicular_lines = detection_info.get('perpendicular_lines', [])
        widths = detection_info.get('widths', [])

        # Plot 1: Original image
        axes[0, 0].imshow(original_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Plot 2: Dark points showing clustering results
        axes[0, 1].imshow(img_normalized, cmap='gray', alpha=0.5)
        if dark_points_yx is not None and len(dark_points_yx) > 0:
            # Show two categories: DBSCAN noise (gray) and clustered points (green)
            if cluster_mask is not None and clustered_points is not None:
                # Non-clustered points (DBSCAN noise)
                non_clustered_points = dark_points_yx[~cluster_mask]

                # Subsample for visualization
                if len(non_clustered_points) > 1000:
                    indices = np.random.choice(len(non_clustered_points), 1000, replace=False)
                    non_clustered_display = non_clustered_points[indices]
                else:
                    non_clustered_display = non_clustered_points

                if len(clustered_points) > 5000:
                    indices = np.random.choice(len(clustered_points), 5000, replace=False)
                    clustered_display = clustered_points[indices]
                else:
                    clustered_display = clustered_points

                # Plot in layers: noise (bottom), clustered points (top)
                if len(non_clustered_display) > 0:
                    axes[0, 1].scatter(non_clustered_display[:, 1], non_clustered_display[:, 0],
                                     c='gray', s=1, alpha=0.2, label=f'DBSCAN Noise ({len(non_clustered_points)})')

                axes[0, 1].scatter(clustered_display[:, 1], clustered_display[:, 0],
                                 c='green', s=1, alpha=0.6, label=f'Main Tray Cluster ({largest_cluster_size})')
            else:
                # Fallback
                if len(dark_points_yx) > 5000:
                    indices = np.random.choice(len(dark_points_yx), 5000, replace=False)
                    dark_points_display = dark_points_yx[indices]
                else:
                    dark_points_display = dark_points_yx
                axes[0, 1].scatter(dark_points_display[:, 1], dark_points_display[:, 0],
                                 c='red', s=1, alpha=0.3, label='Dark Points')

        if m_parallel is not None and b_parallel is not None:
            x_line = np.array([0, img_normalized.shape[1]])
            y_line = m_parallel * x_line + b_parallel
            axes[0, 1].plot(x_line, y_line, 'cyan', linewidth=3, label=f'PCA Tray Axis (y={m_parallel:.3f}x+{b_parallel:.1f})')
            axes[0, 1].legend(loc='upper right', fontsize=8)

        dark_pct = detection_info.get('dark_percentile', 'N/A')
        threshold_val = detection_info.get('intensity_threshold', 'N/A')
        if isinstance(threshold_val, float):
            threshold_str = f'{threshold_val:.3f}'
        else:
            threshold_str = str(threshold_val)
        total_pts = detection_info.get("dark_points_count", 0)
        cluster_pct = (largest_cluster_size / total_pts * 100) if total_pts > 0 else 0
        axes[0, 1].set_title(f'DBSCAN Clustering: {largest_cluster_size}/{total_pts} points ({cluster_pct:.1f}%)\n{dark_pct}th Percentile, Threshold={threshold_str}')
        axes[0, 1].set_xlim(0, img_normalized.shape[1])
        axes[0, 1].set_ylim(img_normalized.shape[0], 0)

        # Plot 3: Perpendicular lines overlay
        axes[0, 2].imshow(original_img, cmap='gray')
        colors = ['red', 'orange', 'yellow', 'green', 'cyan']
        for i, perp_line in enumerate(perpendicular_lines):
            x_samples = perp_line.get('x_samples')
            y_samples = perp_line.get('y_samples')
            width = perp_line.get('width', 0)
            if x_samples is not None and y_samples is not None:
                color = colors[i % len(colors)]
                axes[0, 2].plot(x_samples, y_samples, color=color, linewidth=2, alpha=0.8,
                              label=f'Line {i+1}: {width:.0f}px')

        if len(perpendicular_lines) > 0:
            axes[0, 2].legend(loc='upper right', fontsize=8)
        axes[0, 2].set_title(f'Perpendicular Lines (n={len(perpendicular_lines)})')
        axes[0, 2].axis('off')

        # Plot 4: Width bar chart
        if len(widths) > 0:
            axes[1, 0].bar(range(len(widths)), widths, color=colors[:len(widths)])
            axes[1, 0].axhline(y=np.mean(widths), color='red', linestyle='--', linewidth=2, label=f'Avg: {np.mean(widths):.1f}px')
            axes[1, 0].set_xlabel('Perpendicular Line Index')
            axes[1, 0].set_ylabel('Width (pixels)')
            axes[1, 0].set_title('Width Measurements Along Each Line')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Width Data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Width Measurements')

        # Plot 5: Intensity profiles along perpendicular lines with trough highlighting
        axes[1, 1].set_title('Intensity Profiles with Width Measurement\n(Curvature-Based Detection: Stops at Flat -> Curved Transition)')
        threshold = detection_info.get('intensity_threshold', 0.3)
        for i, perp_line in enumerate(perpendicular_lines):
            intensities_smoothed = perp_line.get('intensities')
            intensities_original = perp_line.get('intensities_original')
            width_start = perp_line.get('width_start')
            width_end = perp_line.get('width_end')
            min_idx = perp_line.get('min_idx')

            if intensities_smoothed is not None:
                color = colors[i % len(colors)]

                # Plot original intensities as faint dotted line
                if intensities_original is not None:
                    axes[1, 1].plot(intensities_original, color=color, alpha=0.3, linestyle=':',
                                   linewidth=1, label=f'Line {i+1} (Original)')

                # Plot smoothed intensities as solid line
                axes[1, 1].plot(intensities_smoothed, color=color, alpha=0.7, linewidth=2,
                               label=f'Line {i+1} (Smoothed)')

                # Highlight the measured width region
                if width_start is not None and width_end is not None:
                    axes[1, 1].axvspan(width_start, width_end, color=color, alpha=0.2)

                    # Mark the start and stop points with vertical lines
                    axes[1, 1].axvline(x=width_start, color=color, linestyle='--',
                                      linewidth=1.5, alpha=0.8)
                    axes[1, 1].axvline(x=width_end, color=color, linestyle='--',
                                      linewidth=1.5, alpha=0.8)

                    # Mark the minimum (deepest) point with a circle
                    if min_idx is not None:
                        axes[1, 1].plot(min_idx, intensities_smoothed[min_idx], 'o',
                                       color=color, markersize=8, markeredgecolor='black',
                                       markeredgewidth=1.5)

        if isinstance(threshold, float):
            threshold_label = f'Threshold: {threshold:.3f}'
        else:
            threshold_label = f'Threshold: {threshold}'
        axes[1, 1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=threshold_label)
        axes[1, 1].set_xlabel('Position Along Line')
        axes[1, 1].set_ylabel('Normalized Intensity')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Summary information
        axes[1, 2].axis('off')
        summary_text = f"Frame: {frame_number}\n"
        summary_text += f"φ: {motor_data.phi:.6f}°\n\n"
        if width_measurement:
            summary_text += f"Average Width: {width_measurement.avg_width:.2f} px\n"
            summary_text += f"Individual Widths:\n"
            for i, w in enumerate(width_measurement.widths):
                summary_text += f"  Line {i+1}: {w:.1f} px\n"
            summary_text += f"\nMin Width: {min(width_measurement.widths):.1f} px\n"
            summary_text += f"Max Width: {max(width_measurement.widths):.1f} px\n"
            summary_text += f"Std Dev: {np.std(width_measurement.widths):.2f} px\n"
        else:
            summary_text += "No valid measurement\n"

        if is_best_frame:
            summary_text = "*** BEST FRAME (SO FAR) ***\n\n" + summary_text
            axes[1, 2].text(0.5, 0.5, summary_text, transform=axes[1, 2].transAxes,
                          fontsize=12, verticalalignment='center', horizontalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            axes[1, 2].text(0.5, 0.5, summary_text, transform=axes[1, 2].transAxes,
                          fontsize=12, verticalalignment='center', horizontalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        axes[1, 2].set_title('Summary')

        prefix = "CURRENT_BEST_" if is_best_frame else ""
        frame_path = Path(self.config.output_images_dir) / f"{prefix}frame_{frame_number}.png"
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        logger.debug(f"Saved Visualization to {frame_path}")


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
            logger.info(f"Non-Persistent Analyzer - Appending to Existing Log: {log_file}")
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
    parser = argparse.ArgumentParser(description='Non-Persistent Tray Width Analyzer Using DBSCAN Clustering, PCA and Curvature-Based Detection')
    parser.add_argument('frame_number', type=int, help='Frame Number to Analyze')
    parser.add_argument('data_path', type=str, help='Path to Data Directory')
    parser.add_argument('--state', type=str, help='JSON State Data From Persistent Processor')
    parser.add_argument('--verbose', action='store_true', help='Enable Verbose Logging')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for Log Files')
    parser.add_argument('--log-file', type=str, help='Specific Log File Path to Append to')
    parser.add_argument('--output-dir', type=str, default='output_images', help='Directory for Output Visualization Images')
    return parser.parse_args()


def main():
    """Main entry point for non-persistent analyzer"""
    if len(sys.argv) == 1 or '--help' in sys.argv:
        print("Usage: python optimize_phi_transient.py <frame_number> <data_path> [--state <json_state>] [--verbose] [--log-dir <dir>] [--log-file <path>] [--output-dir <dir>]")
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
