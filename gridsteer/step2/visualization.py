"""
Visualization Module for Well Tracking System
Contains all plotting and visualization functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple

from gridsteer.step2.well_tracking import format_well_label, well_id_to_row_col


class Visualizer:
    """Visualization utilities for well tracking."""

    def __init__(self, config):
        self.config = config

    def add_well_tracking_visualization(self, ax, tracked_circles: Optional[Tuple],
                                       well_ids: Optional[List],
                                       well_tracker):
        """Add well tracking visualization for two-row configuration."""
        if not well_tracker:
            return

        lines_info = well_tracker.get_line_endpoints()
        if lines_info:
            for line_info in lines_info:
                (x1, y1), (x2, y2) = line_info['endpoints']
                row_id = line_info['row_id']
                is_extrapolated = line_info['is_extrapolated']

                # Choose color based on row
                color = 'yellow' if row_id == 1 else 'cyan'

                # Choose style based on whether it's extrapolated
                if is_extrapolated:
                    linestyle = ':'
                    alpha = 0.4
                    linewidth = 2
                    label = f'Row {row_id} Line (Extrapolated)'
                else:
                    linestyle = '--'
                    alpha = 0.5
                    linewidth = 3
                    label = f'Row {row_id} Line'

                ax.plot([x1, x2], [y1, y2],
                       color=color, linewidth=linewidth, alpha=alpha,
                       linestyle=linestyle, label=label)

        predicted_positions = well_tracker.get_all_predicted_positions()
        
        if predicted_positions:
            for well_id, pred in predicted_positions.items():
                row = pred.get('row', 1)
                color = 'yellow' if row == 1 else 'cyan'
                circle = plt.Circle((pred['x'], pred['y']), pred['radius'],
                                  ec=color, fc='none', ls=':', alpha=0.3, lw=2)
                ax.add_patch(circle)
                
                label = format_well_label(well_id, self.config)
                ax.text(pred['x'], pred['y'], label,
                       ha='center', va='center', fontsize=10,
                       color=color, alpha=0.5,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.3))

        if tracked_circles and well_ids:
            accum, cx, cy, radii = tracked_circles
            
            for x, y, r, well_id, conf in zip(cx, cy, radii, well_ids, accum):
                if well_id:
                    well_info = well_tracker.detected_wells.get(well_id, {})
                    row = well_info.get('row', 1)
                    
                    color = 'lime' if row == 1 else 'aqua'
                    label = format_well_label(well_id, self.config)
                    
                    circle = plt.Circle((x, y), r, ec=color, fc='none',
                                      ls='-', alpha=0.8, lw=4)
                    ax.add_patch(circle)
                    
                    ax.text(x, y, label, ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
                else:
                    circle = plt.Circle((x, y), r, ec='orange', fc='none',
                                      ls='-', alpha=0.8, lw=4)
                    ax.add_patch(circle)
                    
                    ax.text(x, y, '?', ha='center', va='center',
                           fontsize=14, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
    
    def create_visualization(self, frame_number: int, results: Dict,
                           motor_data, well_tracker, config, REMBG_AVAILABLE, PIL_AVAILABLE) -> plt.Figure:
        """Create visualization figure."""
        fig = plt.figure(figsize=(28, 20))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.2)

        axes = [
            fig.add_subplot(gs[0, 0]),  # Original
            fig.add_subplot(gs[0, 1]),  # Circle detection edge  
            fig.add_subplot(gs[0, 2]),  # Circle detection
            fig.add_subplot(gs[1, 0]),  # Background removed
            fig.add_subplot(gs[1, 1]),  # Contour edge
            fig.add_subplot(gs[1, 2]),  # Contours and lines
            fig.add_subplot(gs[2, :])   # Main tracking (full width)
        ]
        
        self._create_debug_subplots(axes, results, motor_data, frame_number,
                                   well_tracker, config, REMBG_AVAILABLE, PIL_AVAILABLE)
        
        return fig
    
    def _create_debug_subplots(self, axes, results: Dict, motor_data, frame_number: int,
                               well_tracker, config, REMBG_AVAILABLE, PIL_AVAILABLE):
        """Create all debug subplots."""

        axes[0].imshow(results['img'], cmap='gray', aspect='equal')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        if results['edge_for_circles'] is not None:
            axes[1].imshow(results['edge_for_circles'], cmap='gray', aspect='equal')
            axes[1].set_title('Canny Edges (Circle Detection)', fontsize=12, fontweight='bold', color='blue')
        else:
            axes[1].text(0.5, 0.5, 'No Edge Data', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=14, color='red')
            axes[1].set_title('Canny Edges: Not Available', fontsize=12, fontweight='bold', color='red')
        axes[1].axis('off')

        axes[2].imshow(results['img'], cmap='gray', aspect='equal')
        edge_status = results.get('edge_detection_status', {})
        kmeans_metadata = edge_status.get('kmeans_cluster_metadata')
        rows_established = edge_status.get('rows_established', False)
        if results['circles']:
            self._draw_circle_detections_debug(axes[2], results['circles'], kmeans_metadata, rows_established)

        title = f'Circle Detection ({results["num_circles"]} Detected)'
        if kmeans_metadata and not rows_established:
            n_detected = kmeans_metadata.get('n_clusters_detected', '?')
            n_returned = kmeans_metadata.get('n_clusters_returned', '?')
            sep_dist = kmeans_metadata.get('separation_distance', 0)
            title += f' | K-Means: {n_returned}/{n_detected} Rows (Δ={sep_dist:.1f}px)'

        axes[2].set_title(title, fontsize=12, fontweight='bold', color='blue')
        axes[2].axis('off')

        edge_status = results.get('edge_detection_status', {})
        
        if results['img_bg_removed'] is not None:
            axes[3].imshow(results['img_bg_removed'], cmap='gray', aspect='equal')
            axes[3].set_title('Background Removed (Complete)', fontsize=12, fontweight='bold', color='green')
            
        elif config.use_background_removal and edge_status.get('edge_condition_satisfied', False):
            axes[3].text(0.5, 0.5, 'Background Removal\nDisabled', 
                        ha='center', va='center', transform=axes[3].transAxes, 
                        fontsize=11, color='gray')
            axes[3].set_title('Background Removal: Disabled', fontsize=12, fontweight='bold', color='gray')
            
        elif config.use_background_removal:
            axes[3].imshow(results['img'], cmap='gray', aspect='equal')
            axes[3].set_title('Background Removal: Error/Failed', fontsize=12, fontweight='bold', color='red')
            
            axes[3].text(0.5, 0.5, 'Background Removal\nFailed or Not Applied', 
                        ha='center', va='center', transform=axes[3].transAxes, 
                        fontsize=10, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        else:
            axes[3].text(0.5, 0.5, 'Background Removal\nDisabled', ha='center', va='center', 
                        transform=axes[3].transAxes, fontsize=12, color='gray')
            axes[3].set_title('Background Removal: Disabled', fontsize=12, fontweight='bold', color='gray')

        axes[3].axis('off')

        if results['edge_for_contours'] is not None:
            axes[4].imshow(results['edge_for_contours'], cmap='gray', aspect='equal')
            if edge_status.get('edge_condition_satisfied', False):
                axes[4].set_title('Edge Detection (Complete)', fontsize=12, fontweight='bold', color='green')
            else:
                axes[4].set_title('Edge Detection (Waiting)', fontsize=12, fontweight='bold', color='orange')
        else:
            axes[4].text(0.5, 0.5, 'Edge Detection\nDisabled', ha='center', va='center', 
                        transform=axes[4].transAxes, fontsize=12, color='gray')
            axes[4].set_title('Edge Detection: Disabled', fontsize=12, fontweight='bold', color='gray')
        axes[4].axis('off')

        if results['edge_for_contours'] is not None:
            axes[5].imshow(results['edge_for_contours'], cmap='gray', aspect='equal')
            
            contour_coords = results.get('contour_coords')
            segments = results.get('segments')
            if contour_coords is not None and segments is not None:
                for i, segment in enumerate(segments):
                    if len(segment) > 1:
                        axes[5].plot(segment[:, 0], segment[:, 1], 'r-', linewidth=2, 
                                    alpha=0.8, label='Hull Boundary' if i == 0 else "")
                axes[5].scatter(contour_coords[:, 0], contour_coords[:, 1], c='red', s=20,
                            alpha=0.8, zorder=5, label='Hull Vertices')

            first_line = True
            for xline, yline in results.get('lines', []):
                label = 'Detected Lines' if first_line else None
                axes[5].plot(xline, yline, 'cyan', linewidth=2, alpha=0.9, linestyle='--', label=label)
                first_line = False
            
            if edge_status.get('edge_condition_satisfied', False):
                axes[5].set_title('Contours & Lines (Complete)', fontsize=12, fontweight='bold', color='green')
            else:
                axes[5].set_title('Contours & Lines (Waiting)', fontsize=12, fontweight='bold', color='orange')
        else:
            axes[5].text(0.5, 0.5, 'Edge Detection\nDisabled', ha='center', va='center', 
                        transform=axes[5].transAxes, fontsize=12, color='gray')
            axes[5].set_title('Contours & Lines: Disabled', fontsize=12, fontweight='bold', color='gray')
        axes[5].axis('off')

        self._create_well_tracking_subplot(axes[6], results, motor_data, frame_number,
                                          well_tracker, config, REMBG_AVAILABLE, PIL_AVAILABLE)
    
    def _draw_circle_detections_debug(self, ax, circles: Tuple, kmeans_metadata: Optional[Dict] = None, rows_established: bool = False):
        """Draw detected circles with confidence and K-Means cluster information.

        Only shows K-Means cluster information (labels, centroids) when rows_established=False
        (i.e., during the Learning phase when K-Means is actively being used).
        """
        accum_values, cx, cy, radii = circles

        if len(accum_values) == 0:
            ax.text(0.5, 0.5, 'No Circles Detected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red', fontweight='bold')
            return

        max_accum = np.max(accum_values)
        min_accum = np.min(accum_values)
        accum_range = max_accum - min_accum if max_accum > min_accum else 1.0

        # Get cluster assignments if available and K-Means is actively being used
        cluster_labels = None
        if kmeans_metadata and 'cluster_labels' in kmeans_metadata and not rows_established:
            cluster_labels = kmeans_metadata['cluster_labels']

        for i, (x, y, r, acc) in enumerate(zip(cx, cy, radii, accum_values)):
            normalized_conf = (acc - min_accum) / accum_range if accum_range > 0 else 1.0
            alpha = 0.5 + 0.5 * normalized_conf
            linewidth = 1 + 2 * normalized_conf

            # Choose color based on cluster assignment if available (only during Learning phase)
            if cluster_labels and i < len(cluster_labels):
                cluster_id = cluster_labels[i]
                if cluster_id == 1:
                    color = 'magenta'  # Row 1
                elif cluster_id == 2:
                    color = 'cyan'  # Row 2
                else:
                    color = 'orange'  # Shouldn't happen
            else:
                # Default confidence-based coloring
                if normalized_conf > 0.75:
                    color = 'lime'
                elif normalized_conf > 0.5:
                    color = 'cyan'
                else:
                    color = 'yellow'

            circle = plt.Circle((x, y), r, ec=color, fc='none', ls='-',
                              alpha=alpha, lw=linewidth)
            ax.add_patch(circle)

            # Show cluster assignment in label (only during Learning phase)
            if cluster_labels and i < len(cluster_labels):
                label_text = f'C{cluster_labels[i]}\n{acc:.2f}'
            else:
                label_text = f'{acc:.2f}'

            ax.text(x, y, label_text, ha='center', va='center', fontsize=8,
                   color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

            ax.plot(x, y, 'o', color=color, markersize=2, alpha=alpha)

        # Draw cluster centroids if available and K-Means is actively being used
        if kmeans_metadata and 'centroids' in kmeans_metadata and not rows_established:
            centroids = kmeans_metadata['centroids']
            img_width = ax.get_xlim()[1] - ax.get_xlim()[0]

            for row_id, centroid_y in centroids.items():
                # Draw a horizontal line at the centroid
                color = 'magenta' if row_id == 1 else 'cyan'
                ax.axhline(y=centroid_y, color=color, linestyle='--', linewidth=2, alpha=0.6)

                # Add label for centroid
                ax.text(5, centroid_y, f'Row {row_id} Centroid\ny={centroid_y:.1f}',
                       fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8),
                       verticalalignment='center')
    
    def _create_well_tracking_subplot(self, ax, results: Dict, motor_data, frame_number: int,
                                     well_tracker, config, REMBG_AVAILABLE, PIL_AVAILABLE):
        """Create main well tracking subplot."""
        ax.imshow(results['img'], cmap='gray')

        edge_status = results.get('edge_detection_status', {})
        show_edge_detection = (
            config.enable_edge_detection and 
            not edge_status.get('edge_condition_satisfied', False)
        )
        
        if show_edge_detection:
            contour_coords = results.get('contour_coords')
            segments = results.get('segments')
            if contour_coords is not None and segments is not None:
                for i, segment in enumerate(segments):
                    if len(segment) > 1:
                        ax.plot(segment[:, 0], segment[:, 1], 'r-', linewidth=3, 
                                alpha=0.8, label='Convex Hull Boundary' if i == 0 else "")
                ax.scatter(contour_coords[:, 0], contour_coords[:, 1], c='red', s=30, 
                           alpha=0.8, zorder=5, label='Hull Vertices')
            
            first_line = True
            for xline, yline in results['lines']:
                label = 'Detected Lines' if first_line else None
                ax.plot(xline, yline, 'cyan', linewidth=2, alpha=0.9, linestyle='--', label=label)
                first_line = False
        
        if results['circles']:
            self._draw_circle_detections(ax, results['circles'])
        
        if results.get('tracked_circles') or well_tracker:
            self.add_well_tracking_visualization(
                ax,
                results['tracked_circles'],
                results['well_ids'],
                well_tracker
            )
        
        if well_tracker and len(well_tracker.detected_wells) > 1:
            self._draw_stagger_relationships(ax, well_tracker, config)
        
        title = self._generate_tracking_title(frame_number, motor_data, results, well_tracker, config,
                                             REMBG_AVAILABLE, PIL_AVAILABLE)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        self._add_motor_position_box(ax, motor_data)
        self._add_calibration_info_box(ax, results)
        self._add_motor_suggestion_box(ax, results, config)
        
        legend_elements = self._get_well_tracking_legend_elements(results, config,
                                                                  REMBG_AVAILABLE, PIL_AVAILABLE)
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
                  framealpha=0.9, edgecolor='white', ncol=2)
        
        ax.axis('off')
    
    def _draw_circle_detections(self, ax, circles: Tuple):
        """Draw detected circles with confidence information."""
        accum_values, cx, cy, radii = circles
        first_circle = True
        
        if len(accum_values) == 0:
            return
            
        max_accum = np.max(accum_values)
        min_accum = np.min(accum_values)
        accum_range = max_accum - min_accum if max_accum > min_accum else 1.0
        
        for i, (x, y, r, acc) in enumerate(zip(cx, cy, radii, accum_values)):
            label = 'Hough Circles' if first_circle else None
            
            normalized_conf = (acc - min_accum) / accum_range if accum_range > 0 else 1.0
            alpha = 0.5 + 0.5 * normalized_conf
            linewidth = 2 + 2 * normalized_conf
            
            circle = plt.Circle((x, y), r, ec='cyan', fc='none', ls='--', 
                              alpha=alpha, lw=linewidth, label=label)
            ax.add_patch(circle)
            
            info_text = f"({x:.0f}, {y:.0f})\nConf: {acc:.3f}\nR: {r:.0f}"
            
            if normalized_conf > 0.75:
                box_color = 'lime'
                text_color = 'lime'
            elif normalized_conf > 0.5:
                box_color = 'cyan'
                text_color = 'cyan'
            else:
                box_color = 'yellow'
                text_color = 'yellow'
            
            ax.text(x, y - r - 15, info_text, 
                    ha='center', va='bottom', fontsize=8, 
                    color=text_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', 
                             edgecolor=box_color, alpha=0.9, linewidth=1))
            
            ax.plot(x, y, 'o', color='cyan', markersize=3, alpha=alpha)
            
            first_circle = False
    
    def _draw_stagger_relationships(self, ax, well_tracker, config):
        """Draw stagger relationship lines between rows."""
        if not well_tracker:
            return
            
        row1_wells = {wid: w for wid, w in well_tracker.detected_wells.items() if w.get('row') == 1}
        row2_wells = {wid: w for wid, w in well_tracker.detected_wells.items() if w.get('row') == 2}
        
        for well_id, well_info in row1_wells.items():
            row, col = well_id_to_row_col(well_id, config)
            row2_right_id = config.total_wells_row1 + col
            row2_left_id = config.total_wells_row1 + col + 1
            
            row2_right = row2_wells.get(row2_right_id)
            row2_left = row2_wells.get(row2_left_id)
            
            if row2_right:
                ax.plot([well_info['x'], row2_right['x']], 
                       [well_info['y'], row2_right['y']], 
                       'gray', alpha=0.3, linewidth=1, linestyle=':')
            if row2_left:
                ax.plot([well_info['x'], row2_left['x']], 
                       [well_info['y'], row2_left['y']], 
                       'gray', alpha=0.3, linewidth=1, linestyle=':')
    
    def _get_well_tracking_legend_elements(self, results: Dict, config, REMBG_AVAILABLE, PIL_AVAILABLE) -> List[Line2D]:
        """Get legend elements for well tracking."""
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=10, lw=3, markeredgecolor='lime', label=f'Row 1 Wells (1,1)-(1,{config.total_wells_row1})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='aqua',
                   markersize=10, lw=3, markeredgecolor='aqua', label=f'Row 2 Wells (2,1)-(2,{config.total_wells_row2})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markersize=10, lw=2, markeredgecolor='yellow', linestyle=':', label='Predicted R1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markersize=10, lw=2, markeredgecolor='cyan', linestyle=':', label='Predicted R2'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markersize=10, lw=3, markeredgecolor='orange', label='Unassigned'),
            Line2D([0], [0], color='yellow', lw=3, ls='--', alpha=0.5, label='Row 1 Line'),
            Line2D([0], [0], color='cyan', lw=3, ls='--', alpha=0.5, label='Row 2 Line'),
            Line2D([0], [0], color='gray', lw=2, ls=':', alpha=0.4, label='Extrapolated Row')
        ]

        # Only show edge detection elements if edge condition is not satisfied yet
        edge_status = results.get('edge_detection_status', {})
        show_edge_detection = (
            config.enable_edge_detection and
            not edge_status.get('edge_condition_satisfied', False)
        )

        if show_edge_detection:
            legend_elements.insert(0, Line2D([0], [0], color='red', lw=3, label='Convex Hull'))
            legend_elements.insert(1, Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, lw=0, label='Hull Vertices'))

        if config.enable_edge_detection:
            if edge_status.get('edge_condition_satisfied'):
                legend_elements.append(Line2D([0], [0], color='green', lw=3, label='Edge Detection: Complete'))
            else:
                legend_elements.append(Line2D([0], [0], color='red', lw=3, label='Edge Detection: Waiting'))

        if edge_status.get('rows_established'):
            legend_elements.append(Line2D([0], [0], color='purple', lw=3, label='Rows: Established (Closest-Row)'))
        else:
            legend_elements.append(Line2D([0], [0], color='orange', lw=3, label='Rows: Learning (K-Means)'))

        if config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                if edge_status.get('edge_condition_satisfied', False):
                    legend_elements.append(Line2D([0], [0], color='gray', lw=3, label='Background Removal: Disabled'))
                else:
                    legend_elements.append(Line2D([0], [0], color='purple', lw=3, label='Background Removal: Active'))
            else:
                legend_elements.append(Line2D([0], [0], color='orange', lw=3, label='Background Removal: Unavailable'))
        
        return legend_elements
    
    def _generate_tracking_title(self, frame_number: int, motor_data, results: Dict,
                                well_tracker, config, REMBG_AVAILABLE, PIL_AVAILABLE) -> str:
        """Generate title for tracking subplot."""
        title_parts = [
            f"Frame {frame_number}",
            f"φ={motor_data.phi:.1f}°",
            f"Circles: {results['num_circles']}"
        ]
        
        if well_tracker:
            row1_count = sum(1 for w in well_tracker.detected_wells.values() if w.get('row') == 1)
            row2_count = sum(1 for w in well_tracker.detected_wells.values() if w.get('row') == 2)
            title_parts.append(f"R1={row1_count}/{config.total_wells_row1}, R2={row2_count}/{config.total_wells_row2}")
        
        if well_tracker:
            prediction_count = len(well_tracker.predicted_positions) if well_tracker.predicted_positions else 0
            total_assigned = len(well_tracker.detected_wells)
            unassigned_count = len(well_tracker.unassigned_detections)
            
            if unassigned_count > 0:
                title_parts.append(f"Unassigned={unassigned_count}")
            elif prediction_count > 0:
                title_parts.append(f"Pred={prediction_count}")
            elif total_assigned == 0:
                title_parts.append("All Unassigned")
            elif (not well_tracker.established_spacing or 
                  not any(row_id in well_tracker.row_params for row_id in [1, 2])):
                if total_assigned < 2:
                    title_parts.append(f"Learning ({total_assigned}/2)")
                else:
                    title_parts.append("Learning Complete")
            else:
                title_parts.append("Ready (Established)")

        edge_status = results.get('edge_detection_status', {})
        if config.enable_edge_detection:
            if edge_status.get('edge_condition_satisfied'):
                title_parts.append(f"Edge@F{edge_status.get('edge_detection_frame', '?')}")
            else:
                title_parts.append("Awaiting Edge")

        if edge_status.get('rows_established'):
            title_parts.append("RowsEst(ClosestRow)")
        else:
            title_parts.append("RowsLearn(K-Means)")

        if config.use_background_removal:
            if REMBG_AVAILABLE and PIL_AVAILABLE:
                if edge_status.get('edge_condition_satisfied', False):
                    title_parts.append("BG-Disabled")
                else:
                    title_parts.append("BG-Active")
            else:
                title_parts.append("BG-Error")

        last_successful = edge_status.get('last_successful_frame')
        if last_successful is not None:
            title_parts.append(f"LastOK@F{last_successful}")

        # Row-specific last seen info for debugging
        row1_last = edge_status.get('row1_last_seen')
        row2_last = edge_status.get('row2_last_seen')
        if row1_last is not None and row2_last is not None:
            if frame_number - row1_last > 5:
                title_parts.append(f"R1@F{row1_last}")
            if frame_number - row2_last > 5:
                title_parts.append(f"R2@F{row2_last}")

        # Row extrapolation status
        row_extrap = edge_status.get('row_extrapolation_active', {})
        if row_extrap.get(1) or row_extrap.get(2):
            extrap_rows = []
            if row_extrap.get(1):
                extrap_rows.append('R1')
            if row_extrap.get(2):
                extrap_rows.append('R2')
            title_parts.append(f"Extrapolating: {','.join(extrap_rows)}")

        # Inter-row spacing info
        inter_spacing = edge_status.get('inter_row_spacing')
        if inter_spacing is not None:
            title_parts.append(f"InterRowΔ={inter_spacing:.1f}px")

        return " - ".join(title_parts)

    def _add_motor_position_box(self, ax, motor_data):
        """Add motor position information."""
        motor_text = (f"Motor Positions\n"
                     f"X: {motor_data.x:.3f}\n"
                     f"Y: {motor_data.y:.3f}\n"
                     f"Z: {motor_data.z:.3f}\n"
                     f"φ: {motor_data.phi:.3f}°")
        ax.text(0.02, 0.98, motor_text,
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, 
                         edgecolor='darkblue', linewidth=2))

    def _add_calibration_info_box(self, ax, results: Dict):
        """Add motor calibration information."""
        calibration_info = results.get('motor_calibration_info', {})
        if not calibration_info:
            return
            
        if calibration_info.get('is_calibrated'):
            max_samples_text = calibration_info.get('max_samples', 'Unlimited')
            cal_text = (f"X,Y Shift Calibration (ALL PRED): {calibration_info['method']}\n"
                      f"Score: {calibration_info['avg_score']:.3f} | "
                      f"Samples: {calibration_info['samples_collected']} (Max: {max_samples_text})\n"
                      f"Wells: {calibration_info.get('avg_common_wells', 0):.1f} | "
                      f"Std: σx={calibration_info.get('avg_pixel_std', {}).get('x', 0):.2f}px "
                      f"σy={calibration_info.get('avg_pixel_std', {}).get('y', 0):.2f}px")
            cal_color = 'green' if calibration_info['avg_score'] > 0.8 else 'orange'
        else:
            max_samples_text = calibration_info.get('max_samples', 'Unlimited')
            cal_text = (f"X,Y Shift Calibration (ALL PRED): Learning...\n"
                      f"Samples: {calibration_info['samples_collected']}/{calibration_info['samples_needed']} "
                      f"(Max: {max_samples_text})\n"
                      f"Mode: {'Averaged' if calibration_info.get('averaging_enabled') else 'Individual'}")
            cal_color = 'gray'
        
        ax.text(0.98, 0.98, cal_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=cal_color, alpha=0.6))

    def _add_motor_suggestion_box(self, ax, results: Dict, config):
        """Add motor position suggestions."""
        suggested_positions = results.get('suggested_motor_positions', {})
        if not suggested_positions or len(suggested_positions) == 0:
            return
            
        if 'average' in suggested_positions:
            suggested_motor = suggested_positions['average']
            suggestion_text = (f"To Center Wells (Avg - PRED):\n"
                             f"Move To X: {suggested_motor.x:.3f}\n"
                             f"Y: {suggested_motor.y:.3f}\n"
                             f"(Z={suggested_motor.z:.3f}, φ={suggested_motor.phi:.1f}° constant)")
        else:
            first_well_id = min(suggested_positions.keys())
            suggested_motor = suggested_positions[first_well_id]
            well_label = format_well_label(first_well_id, config)
            suggestion_text = (f"To Center Well {well_label} (PRED):\n"
                             f"Move To X: {suggested_motor.x:.3f}\n"
                             f"Y: {suggested_motor.y:.3f}\n"
                             f"(Z={suggested_motor.z:.3f}, φ={suggested_motor.phi:.1f}° constant)")
        
        ax.text(0.98, 0.02, suggestion_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))