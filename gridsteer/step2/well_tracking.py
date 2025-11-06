"""
Well Tracking Module
Handles well identification and tracking across frames.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from gridsteer.step2.main import Config

from gridsteer.step2.motor_prediction import MotorPosition

from gridsteer.step2.well_detection import (
    ImageProcessor,
    GeometryUtils,
    REMBG_AVAILABLE,
    PIL_AVAILABLE
)

from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

def calculate_angle_difference(phi1: float, phi2: float) -> float:
    """Calculate minimum angle difference between two phi values."""
    diff = abs(phi1 - phi2)
    return min(diff, 360 - diff) if diff > 180 else diff


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def well_id_to_row_col(well_id: int, config: "Config") -> Tuple[int, int]:
    """Convert well ID to (row, column)."""
    if well_id <= config.total_wells_row1:
        return (1, well_id)
    else:
        return (2, well_id - config.total_wells_row1)


def row_col_to_well_id(row: int, col: int, config: "Config") -> int:
    """Convert (row, column) to well ID."""
    if row == 1:
        return col
    else:
        return config.total_wells_row1 + col


def format_well_label(well_id: int, config: "Config") -> str:
    """Format well ID as (row, column) string."""
    row, col = well_id_to_row_col(well_id, config)
    return f"({row},{col})"


class WellCenterTracker:
    """Tracks when each well is closest to frame center."""

    def __init__(self, frame_shape: Optional[Tuple[int, int]] = None, config: Optional["Config"] = None):
        self.frame_shape = frame_shape
        self.frame_center = None
        self.config = config or Config()
        if frame_shape:
            self.set_frame_shape(frame_shape)

        self.reference_frame_number: Optional[int] = None
        self.reference_motor: Optional[MotorPosition] = None
        self.pixel_offsets_to_center: Dict[int, Tuple[float, float]] = {}
        self.well_tracker_ref: Optional['WellTracker'] = None

    def set_frame_shape(self, frame_shape: Tuple[int, int]):
        self.frame_shape = frame_shape
        height, width = frame_shape[:2]
        self.frame_center = (width / 2, height / 2)

    def update(self, frame_number: int, detected_wells: Dict, motor_data: MotorPosition):
        """Update tracking with new frame data."""
        if not self.frame_center:
            return

        if self.reference_frame_number is None and self.well_tracker_ref is not None:
            if len(self.well_tracker_ref.row_params) >= 2:
                self.reference_frame_number = frame_number
                self.reference_motor = motor_data

                logger.info(f"Reference Frame Established At Frame {frame_number}")

                for well_id, well_info in detected_wells.items():
                    if well_id is None:
                        continue

                    well_x, well_y = well_info['x'], well_info['y']
                    dx = self.frame_center[0] - well_x
                    dy = self.frame_center[1] - well_y
                    self.pixel_offsets_to_center[well_id] = (dx, dy)

                if self.well_tracker_ref.predicted_positions:
                    for well_id, pred_info in self.well_tracker_ref.predicted_positions.items():
                        if well_id not in self.pixel_offsets_to_center:
                            pred_x, pred_y = pred_info['x'], pred_info['y']
                            dx = self.frame_center[0] - pred_x
                            dy = self.frame_center[1] - pred_y
                            self.pixel_offsets_to_center[well_id] = (dx, dy)


    def _estimate_missing_well_offsets(self, well_tracker: Optional['WellTracker']):
        """Estimate pixel offsets for undetected wells."""
        if not well_tracker or not well_tracker.established_spacing:
            return

        for well_id in range(1, self.config.total_wells + 1):
            if well_id in self.pixel_offsets_to_center:
                continue

            row, col = well_id_to_row_col(well_id, self.config)

            if row not in well_tracker.row_params:
                continue

            slope, intercept = well_tracker.row_params[row]
            spacing = well_tracker.established_spacing

            row_wells_with_offsets = {
                wid: self.pixel_offsets_to_center[wid]
                for wid in self.pixel_offsets_to_center.keys()
                if well_id_to_row_col(wid, self.config)[0] == row
            }

            if not row_wells_with_offsets:
                continue

            ref_well_id = min(row_wells_with_offsets.keys(),
                            key=lambda wid: abs(well_id_to_row_col(wid, self.config)[1] - col))
            ref_col = well_id_to_row_col(ref_well_id, self.config)[1]
            ref_dx, ref_dy = self.pixel_offsets_to_center[ref_well_id]

            ref_x = self.frame_center[0] - ref_dx
            ref_y = self.frame_center[1] - ref_dy

            col_diff = col - ref_col
            est_x = ref_x - col_diff * spacing
            est_y = slope * est_x + intercept

            dx = self.frame_center[0] - est_x
            dy = self.frame_center[1] - est_y
            self.pixel_offsets_to_center[well_id] = (dx, dy)
    
    def save_to_json(self, filename: Optional[str] = None, motor_calibration: Optional['InverseMotorCalibration'] = None,
                     well_tracker: Optional['WellTracker'] = None) -> str:
        """Save predicted motor positions for centering wells."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"well_centering_positions_{timestamp}.json"

        if not motor_calibration or not motor_calibration.is_calibrated:
            logger.warning("Motor Calibration Not Available - Cannot Generate Centering Positions")
            return ""

        if not self.frame_center:
            logger.warning("Frame Center Not Available - Cannot Generate Centering Positions")
            return ""

        if self.reference_motor is None or self.reference_frame_number is None:
            logger.warning("No Reference Frame Established - Cannot Generate Centering Positions")
            return ""

        self._estimate_missing_well_offsets(well_tracker)

        json_data = {
            'frame_center': list(self.frame_center),
            'reference_frame_number': self.reference_frame_number,
            'reference_motor_position': {
                'x': float(self.reference_motor.x),
                'y': float(self.reference_motor.y),
                'z': float(self.reference_motor.z),
                'phi': float(self.reference_motor.phi)
            },
            'well_centering_positions': {}
        }

        successful_predictions = 0
        failed_predictions = 0

        for well_id in range(1, self.config.total_wells + 1):
            try:
                if well_id not in self.pixel_offsets_to_center:
                    failed_predictions += 1
                    logger.warning(f"No Pixel Offset Available For Well {format_well_label(well_id, self.config)}")
                    continue

                pixel_delta = np.array(self.pixel_offsets_to_center[well_id])

                motor_shift = motor_calibration.predict_motor_shifts(pixel_delta)

                if motor_shift is None:
                    failed_predictions += 1
                    logger.warning(f"Could Not Predict Motor Shift For Well {format_well_label(well_id, self.config)}")
                    continue

                predicted_motor = MotorPosition(
                    x=self.reference_motor.x + motor_shift[0],
                    y=self.reference_motor.y + motor_shift[1],
                    z=self.reference_motor.z + motor_shift[2],
                    phi=self.reference_motor.phi
                )

                row, col = well_id_to_row_col(well_id, self.config)
                well_key = f"({row},{col})"

                json_data['well_centering_positions'][well_key] = {
                    'well_id': int(well_id),
                    'row': row,
                    'column': col,
                    'pixel_offset_to_center': {
                        'dx': float(pixel_delta[0]),
                        'dy': float(pixel_delta[1])
                    },
                    'predicted_motor_shift': {
                        'dx': float(motor_shift[0]),
                        'dy': float(motor_shift[1]),
                        'dz': float(motor_shift[2])
                    },
                    'motor_position': {
                        'x': float(predicted_motor.x),
                        'y': float(predicted_motor.y),
                        'z': float(predicted_motor.z),
                        'phi': float(predicted_motor.phi)
                    }
                }
                successful_predictions += 1

            except Exception as e:
                failed_predictions += 1
                logger.error(f"Error Predicting Motor Position For Well {well_id}: {e}")

        output_dir = Path(self.config.output_json_dir)
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        keys = list(json_data["well_centering_positions"].keys())
        new_d = {}
        for k in keys:
          new_d[k] = json_data["well_centering_positions"][k]["motor_position"]
        # Crucial print to stdout so TCL can capture it
        print(json.dumps(new_d))

        logger.info(f"Saved Motor Centering Positions: {successful_predictions}/{self.config.total_wells} Wells")
        logger.info(f"  Reference Frame: {self.reference_frame_number}")

        return str(filepath)




class WellIdentifier:
    """Handles well identification logic for staggered layout."""

    def __init__(self, config: "Config"):
        self.config = config

    def identify_well_using_stagger(self, x: float, y: float, row_id: int,
                                   other_row_wells: Dict[int, Dict], spacing: float) -> Optional[int]:
        """Identify well ID using staggered layout."""
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
        """Determine well ID based on spacing from reference wells."""
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



class WellTrackingSystem:
    """Main system for well tracking with motor calibration."""

    def __init__(self, config: "Config"):
        from gridsteer.step2.main import setup_logging
        from gridsteer.step2.motor_prediction import InverseMotorCalibration
        from gridsteer.step2.visualization import Visualizer

        self.config = config

        setup_logging(config.verbose_mode, config.log_directory)

        self.well_tracker = WellTracker(config) if config.enable_well_tracking else None
        self.well_center_tracker = (WellCenterTracker(config=config)
                                   if config.track_well_centers and config.enable_well_tracking
                                   else None)

        if self.well_tracker and self.well_center_tracker:
            self.well_tracker.well_center_tracker = self.well_center_tracker
            self.well_center_tracker.well_tracker_ref = self.well_tracker

        self.motor_calibration = (InverseMotorCalibration(config)
                                 if config.enable_motor_calibration else None)
        self.image_processor = ImageProcessor(config)
        self.visualizer = Visualizer(config)

        self.frames_processed = 0
        self.frames_skipped = 0
        self.frames_with_tracking = 0

        for dir_path in [config.output_dir, config.output_images_dir, config.output_json_dir]:
            Path(dir_path).mkdir(exist_ok=True)

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

        logger.info("="*80)
        logger.info("Motor Calibration: X,Y Shift Prediction Enabled")
        logger.info("Row Assignment: DBSCAN Initially, Then Closest-Row After Establishment")
        logger.info("="*80)

    def load_frame_data(self, frame_number: int) -> Tuple[Optional[MotorPosition], Optional[np.ndarray]]:
        """Load frame data from .npz files."""
        try:
            data_f = os.path.join( self.config.data_path, f"test{frame_number}.npz")
            #data = np.load(f"{self.config.data_path}/test{frame_number}.npz")
            data = np.load(data_f)
            motor_pos = MotorPosition(
                x=float(data['x']),
                y=float(data['y']),
                z=float(data['z']),
                phi=float(data['phi'])
            )
            return motor_pos, data['sample']
        except Exception as e:
            #import traceback
            #traceback.print_exc()
            return None, None

    def process_frame_detection(self, frame_number: int, img: np.ndarray,
                               motor_data: MotorPosition) -> Dict:
        """Process frame for detection."""
        if self.well_center_tracker and self.well_center_tracker.frame_shape is None:
            self.well_center_tracker.set_frame_shape(img.shape)

        original_img = img.copy()

        img_bg_removed = None
        should_use_background_removal = (
            self.config.use_background_removal and
            (not self.well_tracker or not self.well_tracker.edge_condition_satisfied)
        )
        if should_use_background_removal:
            img_bg_removed, _ = self.image_processor.remove_background(img)

        edge_for_circles, circles = self.image_processor.find_circles(img)

        should_perform_edge_detection = (
            self.config.enable_edge_detection and
            (not self.well_tracker or not self.well_tracker.edge_condition_satisfied)
        )

        contour_coords = None
        segments = None
        lines = []
        edge_for_contours = None

        if should_perform_edge_detection:
            img_for_edge = img_bg_removed if img_bg_removed is not None else img

            edge_for_contours = self.image_processor.generate_edge_image(img_for_edge)

            contour_coords, segments = self.image_processor.extract_contour_coordinates(
                edge_for_contours, remove_border_points=True)

            if contour_coords is not None and segments is not None:
                _, lines = self.image_processor.find_lines(
                    contour_coords, img_for_edge.shape, segments, backup_img=img_for_edge)
            else:
                _, lines = self.image_processor.find_lines(
                    None, img_for_edge.shape, None, backup_img=img_for_edge)

        num_circles = len(circles[1]) if circles else 0

        return {
            'img': original_img,
            'img_bg_removed': img_bg_removed,
            'edge_for_circles': edge_for_circles,
            'edge_for_contours': edge_for_contours,
            'circles': circles,
            'lines': lines,
            'contour_coords': contour_coords,
            'segments': segments,
            'num_circles': num_circles
        }

    def update_tracking(self, frame_number: int, detection_results: Dict,
                       motor_data: MotorPosition) -> Dict:
        """Update tracking based on detection results."""
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

        return tracking_results

    def create_visualization(self, frame_number: int, results: Dict,
                           motor_data: MotorPosition):
        """Create visualization figure."""
        return self.visualizer.create_visualization(
            frame_number, results, motor_data, self.well_tracker,
            self.config, REMBG_AVAILABLE, PIL_AVAILABLE
        )

    def run(self):
        """Main processing loop."""
        import imageio
        import matplotlib.pyplot as plt

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
            video_filename = f"annotated_well_tracking_video_{timestamp}.mp4"
            video_path = Path(self.config.output_dir) / video_filename
            writer = imageio.get_writer(video_path, fps=self.config.video_fps)
            logger.info(f"Recording Video To: {video_path}")

        forward_sequence = list(range(self.config.min_frame, self.config.max_frame + 1))
        reverse_sequence = list(range(self.config.max_frame - 1, self.config.min_frame - 1, -1))
        frame_sequence = [(f, False) for f in forward_sequence] + [(f, True) for f in reverse_sequence]
        total_frames_to_process = len(frame_sequence)

        logger.info(f"Processing {total_frames_to_process} Frames (Including Reverse Sequence)")

        initial_layout = "Row 2 Top, Row 1 Bottom" if self.config.initial_row_layout_flipped else "Row 1 Top, Row 2 Bottom"
        logger.info(f"Initial Row Layout: {initial_layout}")

        if self.config.enable_edge_detection:
            logger.info(f"Edge Detection Enabled")
        else:
            logger.info(f"Edge Detection Disabled")

        try:
            for frame_index, (frame_number, is_reverse) in enumerate(frame_sequence, 1):
                print(frame_index)
                if self.config.display_frames:
                    if use_ipython:
                        clear_output(wait=True)
                    else:
                        os.system('cls' if os.name == 'nt' else 'clear')

                motor_data, img = self.load_frame_data(frame_number)
                print(motor_data)

                if motor_data is None or img is None:
                    self.frames_skipped += 1
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

                    rows_msg = ""
                    if edge_status.get('rows_established'):
                        rows_msg = " | Rows: Established (ClosestRow)"
                    else:
                        rows_msg = " | Rows: Learning (DBSCAN)"

                    logger.info(f"Progress: {progress_pct:.1f}% - Frame {frame_number} - φ={motor_data.phi:.1f}°{edge_msg}{rows_msg}")

                fig = self.create_visualization(frame_number, results, motor_data)
                direction_label = " (Reverse)" if is_reverse else ""
                fig.suptitle(f"Frame {frame_number}{direction_label}", fontsize=16, y=0.99)

                if self.config.save_individual_frames:
                    reverse_suffix = "_reverse" if is_reverse else ""
                    frame_path = Path(self.config.output_images_dir) / f"frame_{frame_number}{reverse_suffix}.png"
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
                logger.info(f"Video Saved: {video_path}")

            if self.config.save_json_output and self.well_center_tracker:
                saved_path = self.well_center_tracker.save_to_json(filename=self.config.json_filename, motor_calibration=self.motor_calibration, well_tracker=self.well_tracker)
                logger.info(f"Results Saved To: {saved_path}")

            self._print_final_summary()

    def _print_final_summary(self):
        """Print final processing summary."""
        logger.info(f"Processing Complete:")
        logger.info(f"  Frames Processed: {self.frames_processed}")
        logger.info(f"  Frames Skipped: {self.frames_skipped}")
        logger.info(f"  Frames With Tracking: {self.frames_with_tracking}")

        if self.motor_calibration and self.motor_calibration.is_calibrated:
            logger.info(f"  Calibration: Successfully Trained")
            self._print_calibration_models()

    def _print_calibration_models(self):
        """Print calibration model information."""
        cal = self.motor_calibration

        logger.info("="*80)
        logger.info("Motor Calibration Models")
        logger.info("="*80)

        method_display = {
            'linear': 'Linear Ridge Regression',
            'polynomial': 'Polynomial Ridge Regression (Degree=2)',
            'spline': 'Spline Interpolation (RBF Thin-Plate)'
        }
        model_type = method_display.get(cal.calibration_method, cal.calibration_method.capitalize())

        logger.info(f"Model Type: {model_type}")
        if cal.calibration_method != "spline":
            logger.info(f"Regularization Alpha: {cal.alpha}")
        else:
            logger.info(f"Spline Smoothing: {cal.spline_smoothing}")
        logger.info(f"Training Samples: {len(cal.motor_shifts)}")

        if cal.calibration_method in ["linear", "polynomial"]:
            axes = ['X', 'Y', 'Z']
            models = [cal.model_motor_x, cal.model_motor_y, cal.model_motor_z]

            logger.info("-"*80)
            logger.info("Model Coefficients")
            logger.info("-"*80)

            for axis, model in zip(axes, models):
                score = cal.calibration_scores.get(f'motor_{axis.lower()}', 0.0)
                logger.info(f"Motor {axis} (R² = {score:.4f}):")
                logger.info(f"  Intercept: {model.intercept_:.6f}")

                if cal.calibration_method == "polynomial":
                    feature_names = ['Δpixel_x', 'Δpixel_y', 'Δpixel_x²', 'Δpixel_x*Δpixel_y', 'Δpixel_y²']
                else:
                    feature_names = ['Δpixel_x', 'Δpixel_y']

                logger.info(f"  Coefficients:")
                for name, coef in zip(feature_names, model.coef_):
                    logger.info(f"    {name:15}: {coef:12.6f}")
        else:
            logger.info("-"*80)
            logger.info("Model Performance")
            logger.info("-"*80)
            for axis in ['X', 'Y', 'Z']:
                score = cal.calibration_scores.get(f'motor_{axis.lower()}', 0.0)
                logger.info(f"Motor {axis} (R² = {score:.4f})")

        logger.info("="*80)


class WellTracker:
    """Adaptive well tracking for two-row configuration."""

    def __init__(self, config: "Config"):
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

        self.last_successful_row_wells: Dict[int, Dict[int, Dict]] = {1: {}, 2: {}}
        self.last_successful_row_frame: Dict[int, int] = {}

        self.average_radius = None

        self.predicted_positions = {}
        self.previous_detected_wells = {}

        self.reference_frame_wells = {}
        self.reference_frame_number = None
        self.established_spacing = None

        self.edge_condition_satisfied = False
        self.edge_detection_frame = None
        self.edge_circle_info = None

        self.last_perpendicular_phi = None
        self.row_layout_flipped = config.initial_row_layout_flipped
        self.phi_flip_history = []

        self.rows_established = False
        self.well_center_tracker = None
    
    def _check_and_handle_phi_flip(self, current_phi: float) -> bool:
        """Check if phi has changed by >90° and handle row flipping."""
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

                self.established_spacing = None
                self.reference_frame_wells = {}
                self.reference_frame_number = None
                self.last_successful_frame_wells = {}
                self.last_successful_frame_number = None
                self.rows_established = False
                logger.info("  Cleared Established Spacing And Reference Frame Due To Layout Flip")
        
        self.last_perpendicular_phi = current_phi
        
        return flip_occurred
    
    def _check_edge_condition(self, detections: List[Tuple], lines: List) -> Tuple[bool, Optional[Dict]]:
        """Check if circle is at edge of row."""
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
    
    def _assign_to_established_rows(self, detections: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Assign detections to established rows based on y-coordinate proximity."""
        if not self.row_params:
            return {}
        
        rows = {1: [], 2: []}

        row_y_positions = {}
        
        for row_id in [1, 2]:
            if row_id in self.row_params:
                slope, intercept = self.row_params[row_id]
                # Use middle of image as x reference
                if self.well_center_tracker and self.well_center_tracker.frame_center:
                    ref_x = self.well_center_tracker.frame_center[0]
                else:
                    # Fallback: use average x from detections
                    ref_x = np.mean([d[0] for d in detections]) if detections else 0
                row_y_positions[row_id] = slope * ref_x + intercept
            elif self.last_successful_frame_wells:
                # Use last successful frame to estimate y position
                row_wells = [w for wid, w in self.last_successful_frame_wells.items() 
                            if w.get('row') == row_id]
                if row_wells:
                    row_y_positions[row_id] = np.mean([w['y'] for w in row_wells])
        
        # Assign each detection to closest row
        for det in detections:
            det_x, det_y, det_r = det
            
            if len(row_y_positions) == 0:
                # No row information available
                continue
            
            # Find closest row
            min_dist = float('inf')
            closest_row = None
            
            for row_id, row_y in row_y_positions.items():
                dist = abs(det_y - row_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_row = row_id
            
            # Only assign if within reasonable distance (more lenient than initial clustering)
            if closest_row is not None and min_dist < self.config.row_y_tolerance * 3:
                rows[closest_row].append(det)
        
        # Apply row flipping logic if both rows have detections
        if rows[1] and rows[2]:
            avg_y1 = np.mean([d[1] for d in rows[1]])
            avg_y2 = np.mean([d[1] for d in rows[2]])
            
            if not self.row_layout_flipped:
                # Normal: row with smaller y should be row 1
                if avg_y1 > avg_y2:
                    rows[1], rows[2] = rows[2], rows[1]
            else:
                # Flipped: row with smaller y should be row 2
                if avg_y1 < avg_y2:
                    rows[1], rows[2] = rows[2], rows[1]
        
        # Remove empty rows
        return {k: v for k, v in rows.items() if v}
    
    def _detect_rows(self, detections: List[Tuple], current_phi: float = None) -> Dict[int, List[Tuple]]:
        """Detect and cluster detections into two rows with phi-based row flipping."""
        if len(detections) < 1:
            return {}
        
        if current_phi is not None:
            self._check_and_handle_phi_flip(current_phi)
        
        if self.rows_established and len(self.row_params) >= 1:
            logger.debug(f"Frame {self.frame_number}: Using Established Row Assignment")
            return self._assign_to_established_rows(detections)

        logger.debug(f"Frame {self.frame_number}: Using DBSCAN Clustering For Row Detection")
        
        if len(detections) < 2:
            # Not enough detections to cluster
            return {}
        
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
        """Update tracking of the last successful frame (no unassigned wells)
        UPDATED: Also tracks last successful detection for each row separately
        UPDATED: Also updates reference frame to be the most recent fully-assigned frame"""
        if len(self.unassigned_detections) == 0 and len(self.detected_wells) > 0:
            self.last_successful_frame_wells = self.detected_wells.copy()
            self.last_successful_frame_number = self.frame_number
            logger.debug(f"Updated Last Successful Frame To Frame {self.frame_number} With {len(self.detected_wells)} Wells")

            # Update reference frame to be the most recent fully-assigned frame
            self.reference_frame_wells = self.detected_wells.copy()
            self.reference_frame_number = self.frame_number

            # Calculate spacing for reference frame if not already established
            if self.established_spacing is None:
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

            logger.debug(f"Updated Reference Frame To Frame {self.frame_number}")

        # Update row-specific successful tracking
        for row_id in [1, 2]:
            row_wells = {wid: info for wid, info in self.detected_wells.items() if info.get('row') == row_id}
            if row_wells:
                self.last_successful_row_wells[row_id] = row_wells.copy()
                self.last_successful_row_frame[row_id] = self.frame_number
                logger.debug(f"Updated Row {row_id} Last Successful Frame To {self.frame_number} With {len(row_wells)} Wells")
    
    def _find_best_temporal_match(self, x: float, y: float, row_id: int) -> Optional[int]:
        """Find the best matching well ID from the last successful frame using row-specific history"""
        
        # Priority 1: Use row-specific last successful wells (best for handling row disappearance)
        if row_id in self.last_successful_row_wells and self.last_successful_row_wells[row_id]:
            ref_row_wells = self.last_successful_row_wells[row_id]
            logger.debug(f"Frame {self.frame_number}: Using row-specific reference for Row {row_id} "
                        f"from frame {self.last_successful_row_frame.get(row_id, '?')}")
        # Priority 2: Use overall last successful frame
        elif self.last_successful_frame_wells:
            ref_row_wells = {well_id: well_info for well_id, well_info in self.last_successful_frame_wells.items()
                            if well_info.get('row') == row_id}
        # Priority 3: Use previous frame
        elif self.previous_detected_wells:
            ref_row_wells = {well_id: well_info for well_id, well_info in self.previous_detected_wells.items()
                            if well_info.get('row') == row_id}
        else:
            return None
        
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
        
        if best_id:
            logger.debug(f"Frame {self.frame_number}: Temporal match for ({x:.1f}, {y:.1f}) -> Well {format_well_label(best_id, self.config)} (dist={min_dist:.1f})")
        
        return best_id
    
    def _reevaluate_all_assignments(self, rows: Dict[int, List[Tuple]]) -> bool:
        """Re-evaluate all assignments when unassigned wells are detected
        UPDATED: Uses row-specific references for better handling of row disappearance"""
        logger.debug(f"Re-Evaluating All Assignments At Frame {self.frame_number}")
        
        self.detected_wells = {}
        all_detections = []
        for row_id, row_detections in rows.items():
            for det in row_detections:
                all_detections.append((det[0], det[1], det[2], row_id))
        
        all_detections.sort(key=lambda d: d[0], reverse=True)
        
        # Build reference wells dict, preferring row-specific references
        reference_wells = {}
        for row_id in [1, 2]:
            if row_id in self.last_successful_row_wells and self.last_successful_row_wells[row_id]:
                # Use row-specific reference (best for row that disappeared and reappeared)
                reference_wells.update(self.last_successful_row_wells[row_id])
                logger.debug(f"  Using Row {row_id}-Specific Reference From Frame {self.last_successful_row_frame.get(row_id, '?')}")
        
        # Fallback to overall references if row-specific not available
        if not reference_wells:
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
        """Process detections for a single row
        UPDATED: Uses row-specific references for better handling of row disappearance"""
        sorted_detections = sorted(row_detections, key=lambda d: d[0], reverse=True)
        
        other_row_id = 2 if row_id == 1 else 1
        other_row_wells = {wid: info for wid, info in self.detected_wells.items()
                          if info.get('row') == other_row_id}
        
        for det_x, det_y, det_r in sorted_detections:
            best_id = None
            spacing = self.established_spacing or self.row_spacing.get(row_id)

            # Temporal matching with previous frames
            best_id = self._find_best_temporal_match(det_x, det_y, row_id)

            # Spacing-based matching with reference wells
            if best_id is None and spacing:
                # Build reference wells, preferring row-specific history
                reference_wells = {}
                if row_id in self.last_successful_row_wells and self.last_successful_row_wells[row_id]:
                    reference_wells = self.last_successful_row_wells[row_id]
                elif self.last_successful_frame_wells:
                    reference_wells = self.last_successful_frame_wells
                elif self.reference_frame_wells:
                    reference_wells = self.reference_frame_wells
                elif self.previous_detected_wells:
                    reference_wells = self.previous_detected_wells
                
                if reference_wells:
                    best_id = self.well_identifier.determine_well_id_from_spacing(
                        det_x, det_y, row_id, reference_wells, spacing)

            # Stagger relationship matching with other row
            if best_id is None and other_row_wells and spacing:
                best_id = self.well_identifier.identify_well_using_stagger(det_x, det_y, row_id, other_row_wells, spacing)

            # Spatial consistency within current frame
            if best_id is None and self.detected_wells:
                best_id = self._determine_id_from_current_frame(det_x, det_y, row_id)

            # Initial assignment fallback
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
        """Process detections as two rows with temporal consistency
        MODIFIED: Handles cases where one or both rows may be missing
        UPDATED: Re-fits lines after assignment to track shifting rows"""
        self.detected_wells = {}
        self.predicted_positions = {}
        self.unassigned_detections = []
        
        # Process each row sequentially (assign wells to IDs)
        for row_id in [1, 2]:
            row_detections = rows.get(row_id, [])
            if row_detections:
                self._process_single_row(row_id, row_detections)
        
        # NOW re-fit line parameters based on the assigned detections
        # This allows the lines to track shifting row positions
        for row_id in [1, 2]:
            # Get all detected wells for this row
            row_wells = [(wid, w) for wid, w in self.detected_wells.items() if w.get('row') == row_id]
            
            if len(row_wells) >= self.config.min_circles_per_row:
                # Extract points from detected wells
                points = np.array([[w['x'], w['y']] for _, w in row_wells])
                
                if len(points) > 2:
                    slope, intercept, _ = self.geometry_utils.fit_line_ransac(points, self.config)
                else:
                    slope = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0]) if points[1, 0] != points[0, 0] else 0
                    intercept = points[0, 1] - slope * points[0, 0]
                
                if self.geometry_utils.is_line_horizontal(slope, self.config.max_line_angle_degrees):
                    self.row_params[row_id] = (slope, intercept)
                    logger.debug(f"Frame {self.frame_number}: Updated Row {row_id} line params: slope={slope:.4f}, intercept={intercept:.2f}")
                    
                    # Update spacing
                    positions = sorted([w['x'] for _, w in row_wells])
                    if len(positions) >= 2:
                        spacings = np.diff(positions)
                        self.row_spacing[row_id] = np.median(np.abs(spacings))
                        
                        if self.established_spacing is None:
                            self.established_spacing = self.row_spacing[row_id]
            
            elif len(row_wells) == 1 and row_id in self.row_params:
                # Single detection: update intercept but keep slope to track vertical shift
                well_id, well_info = row_wells[0]
                det_x, det_y = well_info['x'], well_info['y']
                old_slope, old_intercept = self.row_params[row_id]
                
                new_intercept = det_y - old_slope * det_x
                self.row_params[row_id] = (old_slope, new_intercept)
                logger.debug(f"Frame {self.frame_number}: Updated Row {row_id} intercept: {new_intercept:.2f} (single detection)")
                
                if self.established_spacing:
                    self.row_spacing[row_id] = self.established_spacing
            
            # If row has no detections, keep existing parameters but log it
            elif len(row_wells) == 0 and row_id in self.row_params:
                logger.debug(f"Frame {self.frame_number}: Row {row_id} has no detections, keeping previous line parameters")
        
        # Check if rows should be marked as established
        if not self.rows_established and len(self.row_params) >= 2:
            self.rows_established = True
            logger.info(f"Rows Established At Frame {self.frame_number} - Switching To Closest-Row Assignment")
        
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
    
    def update_tracks(self, detected_circles: Optional[Tuple], lines: Optional[List] = None, 
                     current_phi: Optional[float] = None) -> Tuple[Optional[Tuple], Optional[List]]:
        """Update tracking with new detections
        MODIFIED: Handles cases where rows may be missing after establishment"""
        self.frame_number += 1
        self.unassigned_detections = []
        
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
            
            # If rows are established, we can work with 0, 1, or 2 rows
            # If rows are not established, we need exactly 2 rows
            if not self.rows_established and len(rows) != 2:
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
        """Get current edge detection status including row flipping and establishment information
        UPDATED: Includes row-specific tracking information"""
        return {
            'edge_condition_satisfied': self.edge_condition_satisfied,
            'edge_detection_frame': self.edge_detection_frame,
            'edge_circle_info': self.edge_circle_info,
            'current_frame': self.frame_number,
            'row_layout_flipped': self.row_layout_flipped,
            'last_perpendicular_phi': self.last_perpendicular_phi,
            'phi_flip_history': self.phi_flip_history,
            'last_successful_frame': self.last_successful_frame_number,
            'num_successful_frame_wells': len(self.last_successful_frame_wells),
            'rows_established': self.rows_established,
            'row_params_available': list(self.row_params.keys()),
            'established_spacing': self.established_spacing,
            'row1_last_seen': self.last_successful_row_frame.get(1),
            'row2_last_seen': self.last_successful_row_frame.get(2),
            'row1_wells_count': len(self.last_successful_row_wells.get(1, {})),
            'row2_wells_count': len(self.last_successful_row_wells.get(2, {}))
        }
