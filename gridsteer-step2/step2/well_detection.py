"""
Well Detection Module
Handles image processing for circle and line detection.
"""

import cv2
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from step2.main import Config

from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, hough_line, hough_line_peaks

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


class CircleDetector:
    """Handles circle detection using Hough transform."""

    def __init__(self, config: "Config"):
        self.config = config

    def detect_circles(self, img: np.ndarray) -> Tuple:
        """Detect circles using Hough transform."""
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
    """Handles line detection using Hough transform."""

    def __init__(self, config: "Config"):
        self.config = config

    def detect_lines(self, contour_coords: Optional[np.ndarray], img_shape: Tuple,
                    segments: Optional[List] = None,
                    backup_img: Optional[np.ndarray] = None) -> Tuple:
        """Detect lines using Hough transform."""
        threshold = self.config.line_hough_threshold
        min_distance = self.config.line_min_distance
        min_angle = self.config.line_min_angle
        num_peaks = self.config.line_num_peaks
        border_buffer = self.config.border_buffer

        primary_lines = []
        contour_img = None

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
                return contour_img, primary_lines

        if backup_img is not None:
            edge = canny(backup_img, sigma=self.config.backup_edge_sigma,
                        low_threshold=self.config.backup_edge_low_threshold,
                        high_threshold=self.config.backup_edge_high_threshold,
                        use_quantiles=True)

            backup_lines = self._extract_lines_from_image(edge, threshold,
                                                        min_distance, min_angle, num_peaks)
            backup_lines = self._filter_border_lines(backup_lines, edge.shape, border_buffer)
            return edge, backup_lines

        return None, []

    def _extract_lines_from_image(self, img: np.ndarray, threshold: int,
                                min_distance: int, min_angle: int, num_peaks: int) -> List:
        """Extract lines from image using Hough transform."""
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
        """Filter out lines along image borders."""
        height, width = shape
        filtered_lines = []

        for line_data in lines:
            x_coords, y_coords = line_data

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
    """Handles contour extraction and processing."""

    def __init__(self, config: "Config"):
        self.config = config

    def extract_contour_coordinates(self, edge_image: np.ndarray, min_area: Optional[int] = None,
                                  remove_border_points: bool = True,
                                  border_buffer: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """Extract and process contour coordinates."""
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
        """Create contiguous segments from hull coordinates."""
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


class ImageProcessor:
    """Image processing utilities."""

    def __init__(self, config: "Config"):
        self.config = config
        self.circle_detector = CircleDetector(config)
        if config.enable_edge_detection:
            self.line_detector = LineDetector(config)
            self.contour_processor = ContourProcessor(config)
        else:
            self.line_detector = None
            self.contour_processor = None

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
        """Remove background from image using rembg."""
        if not REMBG_AVAILABLE or not PIL_AVAILABLE or not self.config.use_background_removal:
            mask = np.ones_like(img, dtype=np.uint8) * 255
            return img, mask

        try:
            if img.dtype != np.uint8:
                img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                img_normalized = img

            if len(img_normalized.shape) == 2:
                img_rgb = np.stack([img_normalized] * 3, axis=-1)
            else:
                img_rgb = img_normalized

            pil_image = Image.fromarray(img_rgb)

            if self.rembg_session is not None:
                output = remove(pil_image, session=self.rembg_session)
            else:
                output = remove(pil_image)

            result_array = np.array(output)

            if result_array.shape[2] == 4:
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

            return result.astype(np.uint8), mask

        except Exception as e:
            logger.warning(f"Background Removal Failed: {e}. Using Original Image")
            mask = np.ones_like(img, dtype=np.uint8) * 255
            return img, mask

    def generate_edge_image(self, img: np.ndarray) -> np.ndarray:
        """Generate edge image using Canny edge detection."""
        return canny(img, sigma=self.config.edge_sigma,
                    low_threshold=self.config.edge_low_threshold,
                    high_threshold=self.config.edge_high_threshold,
                    use_quantiles=True)

    def find_circles(self, img: np.ndarray) -> Tuple:
        """Detect circles using Hough transform."""
        return self.circle_detector.detect_circles(img)

    def extract_contour_coordinates(self, edge_image: np.ndarray, min_area: Optional[int] = None,
                                   remove_border_points: bool = True,
                                   border_buffer: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """Extract and process contour coordinates."""
        if not self.config.enable_edge_detection or self.contour_processor is None:
            return None, None
        return self.contour_processor.extract_contour_coordinates(edge_image, min_area, remove_border_points, border_buffer)

    def find_lines(self, contour_coords: Optional[np.ndarray], img_shape: Tuple,
                  segments: Optional[List] = None,
                  backup_img: Optional[np.ndarray] = None) -> Tuple:
        """Detect lines using Hough transform."""
        if not self.config.enable_edge_detection or self.line_detector is None:
            return None, []
        return self.line_detector.detect_lines(contour_coords, img_shape, segments, backup_img)


class GeometryUtils:
    """Utility class for geometric operations."""

    @staticmethod
    def fit_line_ransac(points: np.ndarray, config: "Config") -> Tuple[float, float, np.ndarray]:
        """Fit line using RANSAC."""
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
        """Robust median-based line fitting."""
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
        """Check if line is approximately horizontal."""
        max_angle_rad = np.deg2rad(max_angle_degrees)
        max_allowed_slope = np.tan(max_angle_rad)
        return abs(slope) <= max_allowed_slope

    @staticmethod
    def cluster_points_by_y(points: List[Tuple], y_tolerance: float, min_samples: int = 2) -> Dict[int, List[Tuple]]:
        """Cluster points by y-coordinate using DBSCAN."""
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
