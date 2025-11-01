"""
Motor Prediction and Calibration Module
Handles motor position prediction and calibration.
"""

import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from step2.main import Config
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
            logger.warning(f"Invalid Calibration Method '{self.calibration_method}', Defaulting To 'Linear'")
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
        
        self.frame_observations[frame_number] = observation
        
        if self.config.calibration_use_average_movement:
            self._generate_training_pairs_averaged()
        else:
            self._generate_training_pairs_individual()
        
        if len(self.motor_shifts) >= self.min_samples:
            self._train_models()
    
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
            for well_id in common_wells:
                delta = obs2.pixel_positions[well_id] - obs1.pixel_positions[well_id]
                pixel_deltas.append(delta)
            
            avg_pixel_delta = np.mean(pixel_deltas, axis=0)
            std_pixel_delta = np.std(pixel_deltas, axis=0) if len(pixel_deltas) > 1 else np.array([0, 0])

            motor_delta_xyz = obs2.motor_position[:3] - obs1.motor_position[:3]

            if np.linalg.norm(motor_delta_xyz) > 1e-6:
                self.motor_shifts.append(motor_delta_xyz)
                self.pixel_shifts.append(avg_pixel_delta)
                self.frame_pair_history.append((frame1, frame2))
                self.common_wells_history.append(len(common_wells))

                self._update_training_statistics(frame1, frame2, len(common_wells), std_pixel_delta)
    
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

                    self._update_training_statistics(frame1, frame2, 1, np.array([0, 0]))
    
    def _update_training_statistics(self, frame1: int, frame2: int,
                                   num_common_wells: int, std_pixel_delta: np.ndarray):
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
        
        if self.frame_pair_history:
            gaps = [abs(p[1] - p[0]) for p in self.frame_pair_history]
            self.training_stats['average_frame_gap'] = np.mean(gaps)
        
        if self.common_wells_history:
            self.training_stats['average_common_wells'] = np.mean(list(self.common_wells_history))
    
    def _train_models(self):
        """Train regression models to predict motor shifts."""
        if len(self.pixel_shifts) < self.min_samples:
            return

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
                logger.error(f"Failed To Train Spline Models: {e}. Falling Back To Linear.")
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
            'avg_pixel_std': {'x': avg_std_x, 'y': avg_std_y}
        }
