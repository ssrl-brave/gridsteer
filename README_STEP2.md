# Well Identification and Tracking
## A. Edge Condition Check
The system waits for an "edge condition" before starting well labeling:
Edge Condition = Circle within 2 x radius distance of non-horizontal line

* Ensures the system sees the edge of the well plate
* Prevents premature labeling when wells are not fully visible
* Once satisfied, edge detection is disabled for performance

## B. Row Detection and Clustering
Row Configuration:
Row 1: 9 wells labeled (1,1) to (1,9)
Row 2: 10 wells labeled (2,1) to (2,10)

Stagger Pattern: Row 1 wells positioned between adjacent Row 2 wells

Coordinate System and Geometric Relationships:
* X-axis: Wells numbered right-to-left (higher X coordinates = lower well numbers)
* Y-axis: Row 1 on top, Row 2 on bottom
* (1,1) and (2,1) are rightmost wells (largest X coordinates)
* (1,9) and (2,10) are leftmost wells (smallest X coordinates)
* Row 1 well (1,n) positioned between Row 2 wells (2,n) and (2,n+1)
* Expected Row 1 position: x_1n = (x_2n + x_2(n+1)) / 2
* Row 2 edge wells: (2,1) = Row 1 well (1,1) + spacing/2, (2,10) = Row 1 well (1,9) - spacing/2
* Well spacing derived from consecutive detections within each row

Clustering Process:
1. Group detected circles by Y-coordinate using DBSCAN
2. Ensure exactly 2 rows with minimum separation (configurable through the Config class)
3. Handle noise points by assigning to nearest row
4. Apply phi-based row flipping when motor angle changes >90° (i need to remove this)

## C. Well Labeling Strategy
The system uses 5 sequential assignment methods for each detected well:
1. Temporal Matching (Primary):
```best_id = find_best_temporal_match(x, y, row_id)```

* Matches with wells from the last successful frame (successful frame = frame with 0 unassigned wells)
* Uses distance-based matching 
* Helps to provides consistency across frames

2. Cross-Frame Spacing-Based Matching:
```best_id = determine_well_id_from_spacing(x, y, row_id, reference_wells, spacing)```

* Uses reference wells from last successful frame
* Calculates expected positions based on established spacing and previous well's locations

3. Stagger Relationship Matching:
```best_id = identify_well_using_stagger(x, y, row_id, other_row_wells, spacing)```

* Row 1 wells positioned between adjacent Row 2 wells
* Row 2 wells positioned between or adjacent to Row 1 wells
* Uses wells already assigned in the other row of current frame

4. Same-Frame Spacing-Based Matching:
```best_id = determine_id_from_current_frame(x, y, row_id)```

* Uses wells already detected in the same frame and same row
* Calculates expected positions based on established spacing

5. Initial Assignment Fallback:
```best_id = assign_initial_id(x, y, row_id, sorted_detections)```

* Sequential assignment based on position in sorted detection list
* Used when no reference data is available (early frames)

Note: Methods #2 and #4 are both spacing-based calculations, but differ in their reference source:
Method #2: Uses wells from last successful frame
Method #4: Uses wells from current frame 

Re-evaluation Logic:
When unassigned wells are detected after all individual assignments:
```success = reevaluate_all_assignments(rows)```

* Clears all current assignments and starts over
* Uses all 5 methods above in a global optimization approach
* Considers cost functions across all wells simultaneously
