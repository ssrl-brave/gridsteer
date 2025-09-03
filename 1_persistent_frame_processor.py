#!/usr/bin/env python3
"""
Persistent Frame Processor - Handles frame iteration and calls non-persistent analyzer
"""

import subprocess
import sys
import time
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PersistentConfig:
    """Configuration for persistent frame processing"""
    data_path: str = "/qfs/projects/bioprep/data/automation/new_grid_center_db.2/"
    min_frame: int = 0
    max_frame: int = 99
    phi_min: float = 0
    phi_max: float = 360
    max_frames_to_process: int = 100
    analyzer_script: str = "1_non_persistent_processor.py"


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
        
        if not self.analyzer_path.exists():
            raise FileNotFoundError(f"Analyzer Script Not Found: {self.analyzer_path}")
    
    def process_frames(self):
        """Process frames by calling non-persistent analyzer"""
        print("Starting Frame Processing...")
        print(f"Data Path: {self.config.data_path}")
        print(f"Frame Range: {self.config.min_frame} To {self.config.max_frame}")
        print(f"Max Frames To Process: {self.config.max_frames_to_process}")
        print(f"Analyzer Script: {self.analyzer_path}")
        print("-" * 60)
        
        frame_range = range(
            self.config.min_frame, 
            min(self.config.min_frame + self.config.max_frames_to_process, self.config.max_frame + 1)
        )
        
        try:
            for frame_number in frame_range:
                print(f"\nProcessing Frame {frame_number}...")
                
                result = self._call_analyzer(frame_number)
                
                if result['success']:
                    if result['is_best']:
                        print(f"  *** New Best Frame: {frame_number} With Distance {result['distance']:.2f}px ***")
                    else:
                        if result['has_lines']:
                            print(f"  Lines Found, Distance: {result['distance']:.2f}px")
                        else:
                            print("  No Parallel Lines Detected")
                else:
                    print(f"  Skipped: {result.get('reason', 'Unknown Error')}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nProcessing Interrupted By User")
        except Exception as e:
            print(f"\nError During Processing: {e}")
        
        self._print_final_summary()
    
    def _call_analyzer(self, frame_number: int) -> dict:
        """Call the non-persistent analyzer for a single frame"""
        try:
            # Pass current state as JSON argument
            state_json = json.dumps(self.state.to_dict())
            cmd = [
                sys.executable, 
                str(self.analyzer_path),
                str(frame_number),
                self.config.data_path,
                '--state', state_json
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    try:
                        # Parse JSON output from analyzer
                        output_data = json.loads(lines[-1])
                        frame_result = output_data.get('result', {})
                        new_state_data = output_data.get('state', {})
                        
                        # Update persistent state
                        self.state = AnalyzerState.from_dict(new_state_data)
                        
                        return frame_result
                    except json.JSONDecodeError as e:
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
                return {
                    'success': False, 
                    'reason': f'Analyzer Failed: {error_msg}'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False, 
                'reason': 'Analyzer Timed Out'
            }
        except Exception as e:
            return {
                'success': False, 
                'reason': f'Exception Calling Analyzer: {str(e)}'
            }
    
    def _print_final_summary(self):
        """Print final processing summary"""
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total Frames Processed: {self.state.frames_processed}")
        
        if self.state.best_frame_info:
            best = self.state.best_frame_info
            print("\nBest Frame Found:")
            print(f"  Frame Number: {best['frame_number']}")
            # print(f"  Phi Value: {best['phi']:.6f}°")
            print(f"  Phi Value: {(best['phi'] + 90) if best['phi'] <= 180 else (best['phi'] - 90):.6f}°")
            print(f"  Distance Between Lines: {best['distance']:.2f} Pixels")
        else:
            print("\nNo Horizontal Parallel Lines Found In Analyzed Frames.")
        
        print("=" * 60)


def main():
    """Main entry point for persistent frame processor"""
    config = PersistentConfig()
    
    # Override configuration defaults if needed
    # config.data_path = "/path/to/your/data/"
    # config.max_frames_to_process = 50
    
    processor = FrameProcessor(config)
    processor.process_frames()


if __name__ == "__main__":
    main()