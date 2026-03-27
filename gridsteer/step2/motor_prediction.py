"""
Motor Prediction and Calibration Module
Handles motor position prediction and calibration.
"""

import logging
import numpy as np
import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from gridsteer.step2.main import Config
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import RBFInterpolator

logger = logging.getLogger(__name__)

@dataclass
class MotorPosition:
    """Motor position data."""
    x: float
    y: float
    z: float
    phi: float


@dataclass
class FrameObservation:
    """Single frame observation for calibration."""
    frame_number: int
    motor_position: np.ndarray
    pixel_positions: Dict[int, np.ndarray]
    detected_well_ids: Set[int]


class InverseMotorCalibration:
    """Learns inverse transformation from pixel to motor coordinates.
    Predicts X, Y, Z motor shifts while phi remains constant."""

    def __init__(self, config: "Config"):
        self.config = config

        self.min_samples = config.calibration_min_samples
        self.max_samples = config.calibration_max_samples
        self.calibration_method = config.calibration_method.lower()
        self.alpha = config.calibration_alpha
        self.spline_smoothing = config.calibration_spline_smoothing

        if self.calibration_method not in ["linear", "spline", "polynomial"]:
            logger.warning(f"Invalid Calibration Method '{self.calibration_method}', Defaulting to 'Linear'")
            self.calibration_method = "linear"

        self.frame_observations: Dict[int, FrameObservation] = {}

        maxlen = self.max_samples if self.max_samples is not None else None
        self.motor_shifts = deque(maxlen=maxlen)
        self.pixel_shifts = deque(maxlen=maxlen)
        self.frame_pair_history = deque(maxlen=maxlen)
        self.common_wells_history = deque(maxlen=maxlen)

        if self.calibration_method in ["linear", "polynomial"]:
            self.model_motor_x = Ridge(alpha=self.alpha)
            self.model_motor_y = Ridge(alpha=self.alpha)
            self.model_motor_z = Ridge(alpha=self.alpha)
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False) if self.calibration_method == "polynomial" else None
        else:
            self.model_motor_x = None
            self.model_motor_y = None
            self.model_motor_z = None
            self.poly_features = None

        self.is_calibrated = False
        self.calibration_scores = {}
        self.score_history = deque(maxlen=50)  # Track recent average scores
        self.score_deltas = deque(maxlen=50)   # Track score changes when adding pairs

        self.training_stats = {
            'total_pairs_generated': 0,
            'unique_frame_pairs': set(),
            'average_frame_gap': 0,
            'min_frame_gap': float('inf'),
            'max_frame_gap': 0,
            'average_common_wells': 0,
            'min_common_wells': float('inf'),
            'max_common_wells': 0,
            'pixel_delta_std': {'x': [], 'y': []},
            'outliers_detected': 0,
            'outliers_by_method': {}
        }

    def _detect_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
        Returns boolean mask where True indicates inlier."""
        
        if len(data) < 4:
            return np.ones(len(data), dtype=bool)

        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        inliers = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        return inliers

    def _filter_outliers(self, pixel_deltas: list, well_ids: list) -> Tuple[np.ndarray, list, int]:
        """Filter outliers from pixel deltas using Tukey's IQR method.

        Uses the standard IQR outlier detection (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
        Can be disabled by setting calibration_outlier_detection to "none".

        Returns (filtered_deltas, filtered_ids, num_outliers_removed)."""
        method = self.config.calibration_outlier_detection.lower()

        if method == "none":
            return np.array(pixel_deltas), well_ids, 0

        data = np.array(pixel_deltas)

        if method == "iqr":
            inlier_mask = self._detect_outliers_iqr(data)
        else:
            logger.warning(f"Unknown Outlier Detection Method '{method}', Defaulting to IQR")
            inlier_mask = self._detect_outliers_iqr(data)

        filtered_deltas = data[inlier_mask]
        filtered_ids = [wid for wid, is_inlier in zip(well_ids, inlier_mask) if is_inlier]
        num_outliers = len(pixel_deltas) - len(filtered_deltas)

        if num_outliers > 0:
            outlier_ids = [wid for wid, is_inlier in zip(well_ids, inlier_mask) if not is_inlier]
            logger.info(f"Filtered {num_outliers} Outlier(s) Using IQR Method. Outlier Well IDs: {outlier_ids}")

        return filtered_deltas, filtered_ids, num_outliers

    def _is_score_drop_significant(self, new_avg_score: float, old_avg_score: Optional[float] = None) -> bool:
        """Detect if score drop is significant using adaptive threshold based on historical variance."""
        
        # Need enough history to establish baseline variation
        if len(self.score_history) < 5:
            return False

        if old_avg_score is None:
            if len(self.score_history) == 0:
                return False
            old_avg_score = self.score_history[-1]

        score_drop = old_avg_score - new_avg_score

        # Only consider drops (not improvements)
        if score_drop <= 0:
            return False

        # Calculate adaptive threshold based on historical variation
        scores_array = np.array(self.score_history)
        std_dev = np.std(scores_array)

        # Also consider typical magnitude of changes
        if len(self.score_deltas) > 0:
            deltas_array = np.array(self.score_deltas)
            median_abs_change = np.median(np.abs(deltas_array))
        else:
            median_abs_change = 0

        # Threshold adapts to data's natural scale
        # Use 3-sigma rule (99.7% confidence for normal distributions)
        threshold = max(3 * std_dev, median_abs_change)

        is_outlier_drop = score_drop > threshold

        if is_outlier_drop:
            logger.warning(
                f"Significant Score Drop Detected: {old_avg_score:.4f} -> {new_avg_score:.4f} "
                f"(Drop={score_drop:.4f}, Threshold={threshold:.6f}, StdDev={std_dev:.6f})"
            )

        return is_outlier_drop

    def _save_training_state(self) -> dict:
        """Save current training state for potential rollback."""
        return {
            'motor_shifts': list(self.motor_shifts),
            'pixel_shifts': list(self.pixel_shifts),
            'frame_pair_history': list(self.frame_pair_history),
            'common_wells_history': list(self.common_wells_history),
            'is_calibrated': self.is_calibrated,
            'calibration_scores': dict(self.calibration_scores),
        }

    def _restore_training_state(self, state: dict):
        """Restore training state from saved snapshot."""
        self.motor_shifts.clear()
        self.pixel_shifts.clear()
        self.frame_pair_history.clear()
        self.common_wells_history.clear()

        for item in state['motor_shifts']:
            self.motor_shifts.append(item)
        for item in state['pixel_shifts']:
            self.pixel_shifts.append(item)
        for item in state['frame_pair_history']:
            self.frame_pair_history.append(item)
        for item in state['common_wells_history']:
            self.common_wells_history.append(item)

        self.is_calibrated = state['is_calibrated']
        self.calibration_scores = state['calibration_scores']

    def add_observation(self, motor_data: MotorPosition, detected_wells: Dict[int, Dict],
                       frame_number: int):
        """Add observation of motor and pixel positions."""
        if not detected_wells:
            return

        motor_array = np.array([motor_data.x, motor_data.y, motor_data.z, motor_data.phi])
        
        pixel_positions = {}
        for well_id, well_info in detected_wells.items():
            pixel_pos = np.array([well_info['x'], well_info['y']])
            pixel_positions[well_id] = pixel_pos
        
        observation = FrameObservation(
            frame_number=frame_number,
            motor_position=motor_array.copy(),
            pixel_positions=pixel_positions.copy(),
            detected_well_ids=set(detected_wells.keys())
        )

        # Save state before adding this observation's training pairs
        saved_state = self._save_training_state()
        old_avg_score = None
        if self.is_calibrated and len(self.calibration_scores) > 0:
            old_avg_score = np.mean(list(self.calibration_scores.values()))

        self.frame_observations[frame_number] = observation

        if self.config.calibration_use_average_movement:
            self._generate_training_pairs_averaged()
        else:
            self._generate_training_pairs_individual()

        if len(self.motor_shifts) >= self.min_samples:
            self._train_models()

            if self.is_calibrated:
                new_avg_score = np.mean(list(self.calibration_scores.values()))
                if self._is_score_drop_significant(new_avg_score, old_avg_score):
                    logger.warning(f"Rejecting Observation From Frame {frame_number} Due to Significant Score Drop")
                    self._restore_training_state(saved_state)
                    del self.frame_observations[frame_number]
                    self.training_stats['outliers_detected'] += 1
                    if 'score_drop' not in self.training_stats['outliers_by_method']:
                        self.training_stats['outliers_by_method']['score_drop'] = 0
                    self.training_stats['outliers_by_method']['score_drop'] += 1
                    return
    
    def _generate_training_pairs_averaged(self):
        """Generate training pairs using average movement across wells."""
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
            well_ids = []
            for well_id in common_wells:
                delta = obs2.pixel_positions[well_id] - obs1.pixel_positions[well_id]
                pixel_deltas.append(delta)
                well_ids.append(well_id)

            filtered_deltas, filtered_ids, num_outliers = self._filter_outliers(pixel_deltas, well_ids)

            if len(filtered_deltas) < self.config.calibration_min_common_wells:
                continue

            avg_pixel_delta = np.mean(filtered_deltas, axis=0)
            std_pixel_delta = np.std(filtered_deltas, axis=0) if len(filtered_deltas) > 1 else np.array([0, 0])

            motor_delta_xyz = obs2.motor_position[:3] - obs1.motor_position[:3]

            if np.linalg.norm(motor_delta_xyz) > 1e-6:
                self.motor_shifts.append(motor_delta_xyz)
                self.pixel_shifts.append(avg_pixel_delta)
                self.frame_pair_history.append((frame1, frame2))
                self.common_wells_history.append(len(filtered_deltas))

                self._update_training_statistics(frame1, frame2, len(filtered_deltas), std_pixel_delta, num_outliers)
    
    def _generate_training_pairs_individual(self):
        """Generate training pairs from individual well movements."""
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
                motor_delta_xyz = obs2.motor_position[:3] - obs1.motor_position[:3]

                if np.linalg.norm(motor_delta_xyz) > 1e-6:
                    self.motor_shifts.append(motor_delta_xyz)
                    self.pixel_shifts.append(pixel_delta)
                    self.frame_pair_history.append((frame1, frame2))
                    self.common_wells_history.append(1)

                    self._update_training_statistics(frame1, frame2, 1, np.array([0, 0]), 0)

    def _update_training_statistics(self, frame1: int, frame2: int,
                                   num_common_wells: int, std_pixel_delta: np.ndarray,
                                   num_outliers: int = 0):
        """Update training statistics."""
        frame_gap = abs(frame2 - frame1)

        self.training_stats['total_pairs_generated'] += 1
        self.training_stats['unique_frame_pairs'].add((frame1, frame2))

        self.training_stats['min_frame_gap'] = min(self.training_stats['min_frame_gap'], frame_gap)
        self.training_stats['max_frame_gap'] = max(self.training_stats['max_frame_gap'], frame_gap)

        self.training_stats['min_common_wells'] = min(self.training_stats['min_common_wells'], num_common_wells)
        self.training_stats['max_common_wells'] = max(self.training_stats['max_common_wells'], num_common_wells)

        self.training_stats['pixel_delta_std']['x'].append(std_pixel_delta[0])
        self.training_stats['pixel_delta_std']['y'].append(std_pixel_delta[1])

        if num_outliers > 0:
            self.training_stats['outliers_detected'] += num_outliers
            method = self.config.calibration_outlier_detection
            if method not in self.training_stats['outliers_by_method']:
                self.training_stats['outliers_by_method'][method] = 0
            self.training_stats['outliers_by_method'][method] += num_outliers
        
        if self.frame_pair_history:
            gaps = [abs(p[1] - p[0]) for p in self.frame_pair_history]
            self.training_stats['average_frame_gap'] = np.mean(gaps)
        
        if self.common_wells_history:
            self.training_stats['average_common_wells'] = np.mean(list(self.common_wells_history))
    
    def _train_models(self):
        """Train regression models to predict motor shifts."""
        if len(self.pixel_shifts) < self.min_samples:
            return

        old_avg_score = None
        if self.is_calibrated and len(self.calibration_scores) > 0:
            old_avg_score = np.mean(list(self.calibration_scores.values()))

        X = np.array(self.pixel_shifts)
        motor_deltas = np.array(self.motor_shifts)

        if self.calibration_method == "spline":
            try:
                self.model_motor_x = RBFInterpolator(
                    X, motor_deltas[:, 0],
                    smoothing=self.spline_smoothing,
                    kernel='thin_plate_spline'
                )
                self.model_motor_y = RBFInterpolator(
                    X, motor_deltas[:, 1],
                    smoothing=self.spline_smoothing,
                    kernel='thin_plate_spline'
                )
                self.model_motor_z = RBFInterpolator(
                    X, motor_deltas[:, 2],
                    smoothing=self.spline_smoothing,
                    kernel='thin_plate_spline'
                )

                predictions_x = self.model_motor_x(X)
                predictions_y = self.model_motor_y(X)
                predictions_z = self.model_motor_z(X)

                self.calibration_scores['motor_x'] = 1 - np.sum((motor_deltas[:, 0] - predictions_x)**2) / np.sum((motor_deltas[:, 0] - np.mean(motor_deltas[:, 0]))**2)
                self.calibration_scores['motor_y'] = 1 - np.sum((motor_deltas[:, 1] - predictions_y)**2) / np.sum((motor_deltas[:, 1] - np.mean(motor_deltas[:, 1]))**2)
                self.calibration_scores['motor_z'] = 1 - np.sum((motor_deltas[:, 2] - predictions_z)**2) / np.sum((motor_deltas[:, 2] - np.mean(motor_deltas[:, 2]))**2)

            except Exception as e:
                logger.error(f"Failed to Train Spline Models: {e}. Falling Back to Linear.")
                self.calibration_method = "linear"
                self.model_motor_x = Ridge(alpha=self.alpha)
                self.model_motor_y = Ridge(alpha=self.alpha)
                self.model_motor_z = Ridge(alpha=self.alpha)
                self._train_models()
                return

        elif self.calibration_method in ["linear", "polynomial"]:
            X_transformed = X
            if self.calibration_method == "polynomial" and self.poly_features:
                X_transformed = self.poly_features.fit_transform(X)

            self.model_motor_x.fit(X_transformed, motor_deltas[:, 0])
            self.model_motor_y.fit(X_transformed, motor_deltas[:, 1])
            self.model_motor_z.fit(X_transformed, motor_deltas[:, 2])

            self.calibration_scores['motor_x'] = self.model_motor_x.score(X_transformed, motor_deltas[:, 0])
            self.calibration_scores['motor_y'] = self.model_motor_y.score(X_transformed, motor_deltas[:, 1])
            self.calibration_scores['motor_z'] = self.model_motor_z.score(X_transformed, motor_deltas[:, 2])

        self.is_calibrated = True

        new_avg_score = np.mean(list(self.calibration_scores.values()))
        self.score_history.append(new_avg_score)

        if old_avg_score is not None:
            score_delta = new_avg_score - old_avg_score
            self.score_deltas.append(score_delta)
    
    def predict_motor_shifts(self, pixel_delta: np.ndarray) -> Optional[np.ndarray]:
        """Predict motor shifts for X, Y, Z based on pixel movement."""
        if not self.is_calibrated:
            return None

        if len(pixel_delta) != 2:
            raise ValueError("pixel_delta must be 2D (dx, dy)")

        X = pixel_delta.reshape(1, -1)

        if self.calibration_method == "spline":
            motor_dx = float(self.model_motor_x(X)[0]) if hasattr(self.model_motor_x(X), '__len__') else float(self.model_motor_x(X))
            motor_dy = float(self.model_motor_y(X)[0]) if hasattr(self.model_motor_y(X), '__len__') else float(self.model_motor_y(X))
            motor_dz = float(self.model_motor_z(X)[0]) if hasattr(self.model_motor_z(X), '__len__') else float(self.model_motor_z(X))
        else:
            X_transformed = X
            if self.calibration_method == "polynomial" and self.poly_features:
                X_transformed = self.poly_features.transform(X)

            motor_dx = self.model_motor_x.predict(X_transformed)[0]
            motor_dy = self.model_motor_y.predict(X_transformed)[0]
            motor_dz = self.model_motor_z.predict(X_transformed)[0]

        return np.array([motor_dx, motor_dy, motor_dz])
    
    def predict_motor_position_for_pixel_shift(self, current_motor: MotorPosition,
                                              current_pixel: Tuple[float, float],
                                              target_pixel: Tuple[float, float]) -> Optional[MotorPosition]:
        """Predict motor position for pixel shift."""
        if not self.is_calibrated:
            return None

        pixel_delta = np.array([
            target_pixel[0] - current_pixel[0],
            target_pixel[1] - current_pixel[1]
        ])

        motor_shifts = self.predict_motor_shifts(pixel_delta)
        if motor_shifts is None:
            return None

        return MotorPosition(
            x=current_motor.x + motor_shifts[0],
            y=current_motor.y + motor_shifts[1],
            z=current_motor.z + motor_shifts[2],
            phi=current_motor.phi
        )
    
    def estimate_motor_for_well_centering(self, current_motor: MotorPosition,
                                         well_pixel_position: Tuple[float, float],
                                         frame_center: Tuple[float, float]) -> Optional[MotorPosition]:
        """Estimate motor position to center a well."""
        return self.predict_motor_position_for_pixel_shift(
            current_motor, well_pixel_position, frame_center
        )

    def get_calibration_info(self) -> Dict:
        """Get calibration status and quality metrics."""
        if not self.is_calibrated:
            return {
                'is_calibrated': False,
                'samples_collected': len(self.motor_shifts),
                'samples_needed': self.min_samples,
                'max_samples': self.max_samples if self.max_samples else 'Unlimited',
                'mapping_direction': 'Pixel Shift → Motor Shift (X,Y,Z)',
                'averaging_enabled': self.config.calibration_use_average_movement,
                'multi_frame_enabled': self.config.calibration_use_multi_frame,
                'pairing_strategy': self.config.calibration_pairing_strategy,
                'total_frames': len(self.frame_observations)
            }
        
        avg_score = np.mean(list(self.calibration_scores.values()))

        if self.calibration_method == "spline":
            method_desc = f'Spline (RBF, smoothing={self.spline_smoothing})'
        elif self.calibration_method == "polynomial":
            method_desc = 'Polynomial Ridge (degree=2)'
        else:
            method_desc = 'Linear Ridge'

        method_desc += ' (Shift-Based X,Y,Z)'
        if self.config.calibration_use_average_movement:
            method_desc += ' [Averaged]'
        if self.config.calibration_use_multi_frame:
            method_desc += f' [{self.config.calibration_pairing_strategy}]'
        
        avg_std_x = np.mean(self.training_stats['pixel_delta_std']['x']) if self.training_stats['pixel_delta_std']['x'] else 0
        avg_std_y = np.mean(self.training_stats['pixel_delta_std']['y']) if self.training_stats['pixel_delta_std']['y'] else 0
        
        return {
            'is_calibrated': True,
            'samples_collected': len(self.motor_shifts),
            'max_samples': self.max_samples if self.max_samples else 'Unlimited',
            'calibration_scores': self.calibration_scores,
            'avg_score': avg_score,
            'method': method_desc,
            'mapping_direction': 'Pixel Shift → Motor Shift (X,Y,Z; Phi constant)',
            'averaging_enabled': self.config.calibration_use_average_movement,
            'multi_frame_enabled': self.config.calibration_use_multi_frame,
            'pairing_strategy': self.config.calibration_pairing_strategy,
            'training_stats': dict(self.training_stats),
            'unique_frame_pairs': len(self.training_stats['unique_frame_pairs']),
            'avg_frame_gap': self.training_stats['average_frame_gap'],
            'avg_common_wells': self.training_stats['average_common_wells'],
            'avg_pixel_std': {'x': avg_std_x, 'y': avg_std_y},
            'outlier_detection_method': self.config.calibration_outlier_detection,
            'outliers_detected': self.training_stats['outliers_detected'],
            'outliers_by_method': self.training_stats['outliers_by_method']
        }

    def save_model(self, filepath: str):
        """Save the calibration model to a file."""
        if not self.is_calibrated:
            logger.warning("Cannot Save Uncalibrated Model")
            return

        model_data = {
            'calibration_method': self.calibration_method,
            'model_motor_x': self.model_motor_x,
            'model_motor_y': self.model_motor_y,
            'model_motor_z': self.model_motor_z,
            'poly_features': self.poly_features,
            'is_calibrated': self.is_calibrated,
            'calibration_scores': self.calibration_scores,
            'min_samples': self.min_samples,
            'max_samples': self.max_samples,
            'alpha': self.alpha,
            'spline_smoothing': self.spline_smoothing
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Calibration Model Saved to {filepath}")