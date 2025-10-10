#!/usr/bin/env python3
"""
Persistent Frame Processor - Handles frame iteration and calls non-persistent analyzer

python optimize_phi.py "/qfs/projects/bioprep/data/automation/new_grid_center_db.2/" 99 --verbose
"""

import subprocess
import glob
import sys
import os
import json
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from gridsteer import  optimize_phi_transient


@dataclass
class PersistentConfig:
    """Configuration for persistent frame processing"""
    data_path: str = "" 
    min_frame: int = 0
    max_frame: int = 99
    phi_min: float = 0
    phi_max: float = 360
    max_frames_to_process: int = 100
    analyzer_script: str = optimize_phi_transient.__file__
    verbose: bool = False
    log_dir: str = "logs"
    log_file_path: str = ""
    output_images_dir: str = "output_images_1"


@dataclass
class AnalyzerState:
    """State maintained across analyzer calls"""
    frames_processed: int = 0
    min_distance_found: float = float('inf')
    best_frame_info: dict = None
    
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


class FrameProcessor:
    """Persistent frame processor that calls non-persistent analyzer"""
    
    def __init__(self, config: PersistentConfig):
        self.config = config
        self.analyzer_path = Path(config.analyzer_script)
        self.state = AnalyzerState()
        self.logger = logging.getLogger(__name__)
        
        if not self.analyzer_path.exists():
            raise FileNotFoundError(f"Analyzer Script Not Found: {self.analyzer_path}")
    
    def process_frames(self):
        """Process frames by calling non-persistent analyzer"""
        self.logger.info("Starting Frame Processing...")
        self.logger.info(f"Data Path: {self.config.data_path}")
        self.logger.info(f"Frame Range: {self.config.min_frame} To {self.config.max_frame}")
        self.logger.info(f"Max Frames To Process: {self.config.max_frames_to_process}")
        self.logger.info(f"Analyzer Script: {self.analyzer_path}")
        self.logger.info(f"Shared Log File: {self.config.log_file_path}")
        self.logger.info(f"Output Images Directory: {self.config.output_images_dir}")
        self.logger.info("-" * 60)
        
        frame_range = range(
            self.config.min_frame, 
            min(self.config.min_frame + self.config.max_frames_to_process, self.config.max_frame + 1)
        )
        
        try:
            for frame_number in frame_range:
                self.logger.debug(f"Processing Frame {frame_number}...")
                
                result = self._call_analyzer(frame_number)
                
                if result['success']:
                    if result['is_best']:
                        self.logger.info(f"New Best Frame: {frame_number} With Distance {result['distance']:.2f}px")
                    else:
                        if result['has_lines']:
                            self.logger.debug(f"Frame {frame_number}: Lines Found, Distance: {result['distance']:.2f}px")
                        else:
                            self.logger.debug(f"Frame {frame_number}: No Parallel Lines Detected")
                else:
                    self.logger.warning(f"Frame {frame_number} Skipped: {result.get('reason', 'Unknown Error')}")                
        except KeyboardInterrupt:
            self.logger.info("Processing Interrupted By User")
        except Exception as e:
            self.logger.error(f"Error During Processing: {e}")
        
        self._output_result()
    
    def _call_analyzer(self, frame_number: int) -> dict:
        """Call the non-persistent analyzer for a single frame"""
        try:
            state_json = json.dumps(self.state.to_dict())
            self.logger.debug(f"State JSON For Frame {frame_number}: {state_json}")

            cmd = [
                sys.executable, 
                str(self.analyzer_path),
                str(frame_number),
                self.config.data_path,
                '--state', state_json,
                '--output-dir', self.config.output_images_dir
            ]
            
            if self.config.verbose:
                cmd.extend(['--verbose', '--log-file', self.config.log_file_path])

            self.logger.debug(f"Calling Analyzer: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    try:
                        output_data = json.loads(lines[-1])
                        frame_result = output_data.get('result', {})
                        new_state_data = output_data.get('state', {})
                        
                        # Update persistent state
                        self.state = AnalyzerState.from_dict(new_state_data)
                        
                        return frame_result
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON From Analyzer: {e}")
                        return {
                            'success': False, 
                            'reason': f'Invalid JSON Output From Analyzer: {e}'
                        }
                else:
                    return {
                        'success': False, 
                        'reason': 'No Output From Analyzer'
                    }
            else:
                error_msg = result.stderr.strip() if result.stderr else 'Unknown Error'
                self.logger.error(f"Analyzer Failed: {error_msg}")
                return {
                    'success': False, 
                    'reason': f'Analyzer Failed: {error_msg}'
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error("Analyzer Timed Out")
            return {
                'success': False, 
                'reason': 'Analyzer Timed Out'
            }
        except Exception as e:
            self.logger.error(f"Exception Calling Analyzer: {e}")
            return {
                'success': False, 
                'reason': f'Exception Calling Analyzer: {str(e)}'
            }
    
    def _output_result(self):
        """Output the final result"""
        if self.config.verbose:
            self.logger.info("=" * 60)
            self.logger.info("Processing Complete")
            self.logger.info("=" * 60)
            self.logger.info(f"Total Frames Processed: {self.state.frames_processed}")
            
            if self.state.best_frame_info:
                best = self.state.best_frame_info
                self.logger.info("Best Frame Found:")
                self.logger.info(f" Frame Number: {best['frame_number']}")
                self.logger.info(f" Phi Value: {best['phi']:.6f}°")
                self.logger.info(f" Corrected Phi: {(best['phi'] + 90):.6f}°")
                self.logger.info(f" Distance Between Lines: {best['distance']:.2f} Pixels")
            else:
                self.logger.info("No Horizontal Parallel Lines Found In Analyzed Frames")
            self.logger.info("=" * 60)
        
        # Output phi value to stdout
        if self.state.best_frame_info:
            best = self.state.best_frame_info
            corrected_phi = best['phi'] + 90
            print(f"{corrected_phi:.6f}")
        else:
            print("NaN")


def setup_logging(config: PersistentConfig):
    """Setup logging configuration"""
    if config.verbose:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"line_analysis_session_{timestamp}.log"
        
        config.log_file_path = str(log_file)
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.NullHandler()
            ]
        )
        logging.getLogger().info(f"Persistent Processor - Session Log Created: {log_file}")
    else:
        logging.basicConfig(level=logging.CRITICAL)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Persistent Frame Processor For Line Analysis')
    parser.add_argument('data_path', type=str, help='Path To Data Directory Containing .npz Files')
    parser.add_argument('--imgs_to_proc', type=int, help='Number Of Images To Process', default=None)
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable Verbose Logging Mode')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory For Log Files (Used With --verbose)')
    parser.add_argument('--output-dir', type=str, default='output_images_1', help='Directory For Output Visualization Images')
    return parser.parse_args()


def main():
    """Main entry point for persistent frame processor"""
    args = parse_arguments()
    
    config = PersistentConfig()
    config.data_path = args.data_path
    config.verbose = args.verbose
    config.log_dir = args.log_dir
    config.output_images_dir = args.output_dir
    
    setup_logging(config)
    
    # Determine frame range from available files
    all_frames = glob.glob(f"{args.data_path}/*npz")
    frames_total = len(all_frames)
    config.max_frames_to_process = args.imgs_to_proc if args.imgs_to_proc is not None else frames_total
    config.min_frame = 0
    config.max_frame = frames_total
    
    processor = FrameProcessor(config)
    processor.process_frames()


if __name__ == "__main__":
    main()
