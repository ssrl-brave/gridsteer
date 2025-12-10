#!/usr/bin/env python3
"""
Hough Circle Radius Detector - Detects circles using Hough Transform with 2-line constraint
"""

import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional


@dataclass
class HoughDetectorConfig:
    """Configuration for Hough circle detection"""
    npz_path: str = ""
    visualize_all: bool = False
    visualize_best: bool = True
    min_radius: int = 10
    max_radius: int = 200
    radius_step: int = 5
    verbose: bool = False
    log_dir: str = "logs"
    log_file_path: str = ""
    output_dir: str = "output_images_1_5"
    canny_sigma: float = 15.0
    canny_low: float = 0.3
    canny_high: float = 0.7
    num_peaks: int = 20
    max_lines: int = 2
    line_tolerance_factor: float = 0.5


class HoughCircleDetector:
    """Circle detector using Hough Transform with 2-line constraint"""
    
    def __init__(self, config: HoughDetectorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _count_rows(self, y_coords: np.ndarray, tolerance: float) -> int:
        """Count number of distinct horizontal rows based on y-coordinates."""
        if len(y_coords) == 0:
            return 0
        
        sorted_y = np.sort(y_coords)
        rows = 1
        last_y = sorted_y[0]
        
        for y in sorted_y[1:]:
            if abs(y - last_y) > tolerance:
                rows += 1
                last_y = y
        
        return rows
    
    def _load_image(self) -> Optional[np.ndarray]:
        """Load image from npz file"""
        try:
            if not Path(self.config.npz_path).exists():
                raise FileNotFoundError(f"NPZ File Not Found: {self.config.npz_path}")

            self.logger.info(f"Loading NPZ File: {self.config.npz_path}")
            data = np.load(self.config.npz_path)
            img = data['sample']

            self.original_img = img.copy()

            # Normalize to 0-1 range if needed
            if img.dtype == np.uint8:
                img = img.astype(float) / 255.0
            else:
                img = (img - img.min()) / (img.max() - img.min())

            self.logger.info(f"Image Shape: {img.shape}, Dtype: {img.dtype}")
            return img

        except Exception as e:
            self.logger.error(f"Failed To Load NPZ File: {e}")
            self.logger.exception("Full Traceback:")
            return None
    
    def _visualize_radius(self, test_radius: int, circles: List, num_rows: int, 
                        fits_two_lines: bool, confidence_score: float, num_circles: int):
        """Visualize detected circles for a specific radius"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(self.original_img, cmap='gray')
            
            for cx_i, cy_i, r_i, acc_i in circles:
                circle = Circle((cx_i, cy_i), r_i, fill=False, color='lime', linewidth=2)
                ax.add_patch(circle)
                ax.plot(cx_i, cy_i, 'r.', markersize=8)
                ax.text(cx_i, cy_i - r_i - 5, f'{r_i:.0f}', color='red', 
                    fontsize=10, ha='center', weight='bold')
            
            title = (f"Test Radius: {test_radius}px | Found: {num_circles} circles | "
                    f"Rows: {num_rows} | Score: {confidence_score:.2f}")
            if fits_two_lines:
                title += f" Successfully Fits in {num_rows} Line{'s' if num_rows != 1 else ''}"
            
            ax.set_title(title, fontsize=14, weight='bold')
            ax.axis('off')
            plt.tight_layout()
            
            if self.config.visualize_all:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"radius_{test_radius:03d}.png"
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                self.logger.debug(f"Saved Visualization: {output_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error During Visualization: {e}")
            plt.close()
    
    def _visualize_best(self, best_result: dict, median_radius: float, all_radii: List[float]):
        """Visualize the best result"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.imshow(self.original_img, cmap='gray')
            
            for cx, cy, r, acc in best_result['circles']:
                circle = Circle((cx, cy), r, fill=False, color='lime', linewidth=2)
                ax.add_patch(circle)
                ax.plot(cx, cy, 'r.', markersize=8)
                ax.text(cx, cy - r - 5, f'{r:.0f}', color='red', 
                       fontsize=10, ha='center', weight='bold')
            
            ax.set_title(f"Median Radius: {median_radius:.1f}px ({best_result['num_circles']} Circles, "
                        f"{best_result['num_rows']} Row(s))", 
                        fontsize=14, weight='bold')
            ax.axis('off')
            plt.tight_layout()
            
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(exist_ok=True)

            npz_name = Path(self.config.npz_path).stem
            output_path = output_dir / f"{npz_name}_best_result.png"
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved Best Result To: {output_path}")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error During Best Visualization: {e}")
            plt.close()
    
    def get_all_circles(self) -> List[dict]:
        """
        Detect all circles across all tested radii using Hough Transform.

        Returns:
            List of detection results, each containing:
                - test_radius: The radius value tested
                - num_circles: Number of circles detected
                - avg_radius: Average radius of detected circles
                - std_radius: Standard deviation of radii
                - avg_accumulator: Average confidence from Hough transform
                - confidence_score: Combined score (circles * confidence / (1 + std))
                - num_rows: Number of horizontal rows circles fall into
                - fits_two_lines: Whether circles fit within max_lines constraint
                - circles: List of (cx, cy, radius, accumulator) tuples
        """
        try:
            self.logger.info("Starting Hough Circle Detection...")
            self.logger.info(f"Radius Range: {self.config.min_radius}-{self.config.max_radius} (Step {self.config.radius_step}), Max Lines: {self.config.max_lines}")

            img = self._load_image()
            if img is None:
                return []

            self.logger.info(f"Running Canny Edge Detection (Sigma={self.config.canny_sigma})...")
            edge = canny(img, sigma=self.config.canny_sigma,
                        low_threshold=self.config.canny_low,
                        high_threshold=self.config.canny_high,
                        use_quantiles=True)

            results = []

            self.logger.info("Sweeping Through Radii (Largest To Smallest)...")
            for test_radius in range(self.config.max_radius, self.config.min_radius - 1, -self.config.radius_step):
                radius_min = max(5, test_radius - 5)
                radius_max = test_radius + 5
                rads = np.arange(radius_min, radius_max + 1)

                hough_res = hough_circle(edge, rads)

                accum, cx, cy, radii = hough_circle_peaks(
                    hough_res, rads,
                    min_xdistance=int(2 * test_radius),
                    min_ydistance=int(2 * test_radius),
                    num_peaks=self.config.num_peaks,
                    threshold=None
                )

                if len(accum) > 0:
                    num_circles = len(accum)
                    avg_radius = np.mean(radii)
                    std_radius = np.std(radii) if len(radii) > 1 else 0
                    avg_confidence = np.mean(accum)
                    confidence_score = (num_circles * avg_confidence) / (1 + std_radius)

                    tolerance = test_radius * self.config.line_tolerance_factor
                    num_rows = self._count_rows(cy, tolerance)
                    fits_max_lines = num_rows <= self.config.max_lines

                    circles_list = list(zip(cx, cy, radii, accum))

                    results.append({
                        'test_radius': test_radius,
                        'num_circles': num_circles,
                        'avg_radius': avg_radius,
                        'std_radius': std_radius,
                        'avg_accumulator': avg_confidence,
                        'confidence_score': confidence_score,
                        'num_rows': num_rows,
                        'fits_two_lines': fits_max_lines,
                        'circles': circles_list
                    })

                    log_msg = (f"r={test_radius:3d}: {num_circles} Circles, "
                              f"Avg_r={avg_radius:.1f}, Std={std_radius:.2f}, "
                              f"Accum={avg_confidence:.2f}, Score={confidence_score:.2f}, "
                              f"Rows={num_rows}")
                    if fits_max_lines:
                        log_msg += f" ✓ Fits {self.config.max_lines} Lines"

                    self.logger.info(log_msg)

                    if self.config.visualize_all:
                        self._visualize_radius(test_radius, circles_list, num_rows,
                                              fits_max_lines, confidence_score, num_circles)

            if not results:
                self.logger.warning("No Circles Detected")

            return results

        except Exception as e:
            self.logger.error(f"Error During Circle Detection: {e}")
            self.logger.exception("Full Traceback:")
            return []

    def get_circles_on_lines(self, all_circle_results: List[dict]) -> Optional[dict]:
        """
        Filter circle detection results to find the best set that fits on the specified number of lines.

        Args:
            all_circle_results: List of circle detection results from get_all_circles()

        Returns:
            Best result dictionary that fits the line constraint, or best overall if none fit.
            Returns None if no circles were detected at all.
        """
        if not all_circle_results:
            self.logger.warning("No Circle Results To Filter")
            return None

        max_line_results = [r for r in all_circle_results if r['fits_two_lines']]

        if not max_line_results:
            self.logger.warning(f"No Results Fit In {self.config.max_lines} Line(s). Using Best Overall Result.")
            best = max(all_circle_results, key=lambda x: x['confidence_score'])
        else:
            best = max(max_line_results, key=lambda x: x['confidence_score'])

        self.logger.info("=" * 60)
        self.logger.info(f"Best Match: {best['num_circles']} Circles In {best['num_rows']} Row(s)")
        self.logger.info(f"  Radius: {best['avg_radius']:.2f}±{best['std_radius']:.2f}px (Test: {best['test_radius']}px)")
        self.logger.info(f"  Confidence Score: {best['confidence_score']:.2f}")
        self.logger.info("=" * 60)

        return best

    def detect_circle_radius(self) -> Tuple[Optional[float], List[float]]:
        """
        Detect circles using Hough Transform with 2-line constraint.
        Sweeps from largest to smallest radius and selects best result.

        Returns:
            (median_radius, all_radii): Median radius and list of all radii, or (None, [])
        """
        try:
            all_circle_results = self.get_all_circles()

            if not all_circle_results:
                return None, []

            best = self.get_circles_on_lines(all_circle_results)

            if best is None:
                return None, []

            all_radii = [r for _, _, r, _ in best['circles']]
            median_radius = np.median(all_radii)

            self.logger.info(f"Median Radius: {median_radius:.2f} Pixels")
            self.logger.info(f"All Radii: {sorted([round(r, 1) for r in all_radii])}")

            if self.config.visualize_best:
                self._visualize_best(best, median_radius, all_radii)

            return median_radius, all_radii

        except Exception as e:
            self.logger.error(f"Error During Circle Detection: {e}")
            self.logger.exception("Full Traceback:")
            return None, []


def setup_logging(config: HoughDetectorConfig):
    """Configure logging based on verbose flag"""
    if config.verbose:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        npz_name = Path(config.npz_path).stem if config.npz_path else "unknown"
        log_file = log_dir / f"hough_detection_{npz_name}_{timestamp}.log"

        config.log_file_path = str(log_file)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file)
            ]
        )
        logging.getLogger().info(f"Hough Circle Detector - Log Created: {log_file}")
    else:
        # Only radius value will be printed to stdout
        logging.basicConfig(
            level=logging.CRITICAL,
            format='%(levelname)s: %(message)s',
            handlers=[
                logging.NullHandler()
            ]
        )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hough Circle Radius Detector')

    parser.add_argument('npz_path', type=str, help='Path To NPZ File')
    
    parser.add_argument('--min-radius', type=int, default=10,
                       help='Minimum Radius To Test (Default: 10)')
    parser.add_argument('--max-radius', type=int, default=200,
                       help='Maximum Radius To Test (Default: 200)')
    parser.add_argument('--radius-step', type=int, default=5,
                       help='Radius Step Size (Default: 5)')
    
    parser.add_argument('--max-lines', type=int, default=2,
                       help='Maximum Number Of Lines/Rows For Circle Arrangement (Default: 2)')
    parser.add_argument('--line-tolerance', type=float, default=0.5,
                       help='Line Grouping Tolerance As Fraction Of Radius (Default: 0.5)')
    
    parser.add_argument('--visualize-all', action='store_true',
                       help='Save Visualization For Every Radius Tested')
    parser.add_argument('--no-visualize-best', action='store_true',
                       help='Do Not Save Visualization Of Best Result')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable Verbose Logging Mode')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory For Log Files (Used With --verbose)')
    parser.add_argument('--output-dir', type=str, default='output_images',
                       help='Directory For Output Visualization Images')
    
    return parser.parse_args()


def main():
    """Main entry point for Hough circle detector"""
    args = parse_arguments()

    config = HoughDetectorConfig()
    config.npz_path = args.npz_path
    config.min_radius = args.min_radius
    config.max_radius = args.max_radius
    config.radius_step = args.radius_step
    config.max_lines = args.max_lines
    config.line_tolerance_factor = args.line_tolerance
    config.visualize_all = args.visualize_all
    config.visualize_best = not args.no_visualize_best
    config.verbose = args.verbose
    config.log_dir = args.log_dir
    config.output_dir = args.output_dir
    
    setup_logging(config)
    
    try:
        detector = HoughCircleDetector(config)
        median_radius, radii = detector.detect_circle_radius()
        
        if median_radius is not None:
            print(f"{median_radius:.2f}")
            sys.exit(0)
        else:
            print("NaN")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected Error In Main: {e}")
        logging.exception("Full Traceback:")
        print("NaN")
        sys.exit(1)


if __name__ == "__main__":
    main()