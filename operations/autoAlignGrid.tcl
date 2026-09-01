
proc autoAlignGrid_initialize { } {
    global offaxis_url
    if {[catch {set offaxis_url [::config getSnapshotDirectUrl]} err]} {
        send_operation_update "Error fetching URL: $err"
        set offaxis_url "unknown"
    }
    send_operation_update "autoAlignGrid initialized. Camera: $offaxis_url"
}

proc autoAlignGrid_start { args } {
    # --- Help ---
    if { [llength $args] > 0 && ([lindex $args 0] eq "-h" || [lindex $args 0] eq "--help") } {
        send_operation_update "Usage: autoAlignGrid ?--iters N? ?--phi F? ?--server URL? ?--margin F? ?--dry-run?"
        send_operation_update "  --iters N     — max BO iterations (default: 20)"
        send_operation_update "  --phi F       — face-on phi angle in degrees (default: current gonio_phi)"
        send_operation_update "  --server URL  — alignment server URL (default: http://pxgpu03:8003)"
        send_operation_update "  --margin F    — fraction of limit range to use, 0-1 (default: 0.9)"
        send_operation_update "  --dry-run     — score current position only, do not move"
        send_operation_update ""
        send_operation_update "Aligns a grid/loop tip to the camera crosshair using Bayesian"
        send_operation_update "Optimization over (sample_x, sample_y, sample_z). Scores the"
        send_operation_update "sample at phi (face view) and phi+90 (edge view) each iteration."
        send_operation_update "Runs centerGoodLimits first to start from center of travel."
        return OK
    }

    # --- Parse args ---
    set max_iters 20
    set phi_face ""
    set server_url "http://pxgpu03:8003"
    set dryrun 0
    set margin 0.9
    set i 0
    while {$i < [llength $args]} {
        set a [lindex $args $i]
        if { $a eq "--iters" } {
            incr i
            set max_iters [lindex $args $i]
        } elseif { $a eq "--phi" } {
            incr i
            set phi_face [lindex $args $i]
        } elseif { $a eq "--server" } {
            incr i
            set server_url [lindex $args $i]
        } elseif { $a eq "--dry-run" } {
            set dryrun 1
        } elseif { $a eq "--margin" } {
            incr i
            set margin [lindex $args $i]
        }
        incr i
    }

    variable sample_x
    variable sample_y
    variable sample_z
    variable gonio_phi
    global offaxis_url

    # Use current phi as face-on angle if not specified
    if { $phi_face eq "" } {
        set phi_face $gonio_phi
    }
    set phi_edge [expr {$phi_face + 90.0}]

    # --- Step 1: Move to center of limits ---
    send_operation_update "Step 1: Moving to center of motor limits..."
    set centerOp [start_waitable_operation centerGoodLimits]
    wait_for_operation_to_finish $centerOp

    # --- Step 2: Get motor limits for BO bounds ---
    foreach {x_lo x_hi} [getGoodLimits sample_x] break
    foreach {y_lo y_hi} [getGoodLimits sample_y] break
    foreach {z_lo z_hi} [getGoodLimits sample_z] break

    set x_center [expr {($x_lo + $x_hi) / 2.0}]
    set y_center [expr {($y_lo + $y_hi) / 2.0}]
    set z_center [expr {($z_lo + $z_hi) / 2.0}]
    set x_range [expr {($x_hi - $x_lo) / 2.0 * $margin}]
    set y_range [expr {($y_hi - $y_lo) / 2.0 * $margin}]
    set z_range [expr {($z_hi - $z_lo) / 2.0 * $margin}]

    send_operation_update "BO bounds: x=$x_center +/- $x_range  y=$y_center +/- $y_range  z=$z_center +/- $z_range"
    send_operation_update "Face phi=$phi_face  Edge phi=$phi_edge  Iters=$max_iters"

    # --- Step 3: Initialize BO on server ---
    set initPayload "{\"x_start\": $x_center, \"y_start\": $y_center, \"z_start\": $z_center, \"x_range\": $x_range, \"y_range\": $y_range, \"z_range\": $z_range}"
    set initCmd [list curl -s -X POST "$server_url/bo/align_grid_init" \
             -H "Content-Type: application/json" \
             -d $initPayload]

    if {[catch {exec {*}$initCmd} initRes]} {
        send_operation_update "BO init failed: $initRes"
        return ERROR
    }
    send_operation_update "BO initialized"

    # --- Step 4: BO Loop ---
    for {set iter 0} {$iter < $max_iters} {incr iter} {
        send_operation_update "--- Iteration $iter/$max_iters  x=$sample_x y=$sample_y z=$sample_z ---"

        # 4a: Score face view
        move gonio_phi to $phi_face
        wait_for_devices gonio_phi
        after 500

        set facePayload "{\"url\": \"$offaxis_url\"}"
        set faceCmd [list curl -s -X POST "$server_url/bo/align_grid_score" \
                 -H "Content-Type: application/json" \
                 -d $facePayload]

        if {[catch {exec {*}$faceCmd} faceRes]} {
            send_operation_update "Face score failed: $faceRes"
            break
        }

        set face_score 0.0
        regexp {"score":\s*([-+]?[0-9]*\.?[0-9]+)} $faceRes match face_score

        # 4b: Score edge view + register + get next suggestion
        move gonio_phi to $phi_edge
        wait_for_devices gonio_phi
        after 500

        set stepPayload "{\"url\": \"$offaxis_url\", \"face_score\": $face_score}"
        set stepCmd [list curl -s -X POST "$server_url/bo/align_grid_step" \
                 -H "Content-Type: application/json" \
                 -d $stepPayload]

        if {[catch {exec {*}$stepCmd} stepRes]} {
            send_operation_update "BO step failed: $stepRes"
            break
        }

        set edge_score 0.0
        set composite 0.0
        regexp {"edge_score":\s*([-+]?[0-9]*\.?[0-9]+)} $stepRes match edge_score
        regexp {"composite":\s*([-+]?[0-9]*\.?[0-9]+)} $stepRes match composite
        send_operation_update "  face=$face_score  edge=$edge_score  composite=$composite"

        if { $dryrun } {
            send_operation_update "  Dry run — not moving"
            continue
        }

        # 4c: Move to next suggested position
        set got_next 0
        if {[regexp {"next_x":\s*([-+]?[0-9]*\.?[0-9]+)} $stepRes match next_x] &&
            [regexp {"next_y":\s*([-+]?[0-9]*\.?[0-9]+)} $stepRes match next_y] &&
            [regexp {"next_z":\s*([-+]?[0-9]*\.?[0-9]+)} $stepRes match next_z]} {
            set got_next 1
        }

        if { !$got_next } {
            send_operation_update "  No next suggestion in response — stopping"
            break
        }

        send_operation_update "  Next: x=$next_x y=$next_y z=$next_z"
        move sample_x to $next_x
        move sample_y to $next_y
        move sample_z to $next_z
        wait_for_devices sample_x sample_y sample_z
        after 300
    }

    # --- Step 5: Final homing to best position ---
    if { !$dryrun } {
        send_operation_update "Homing to best position..."
        set bestCmd [list curl -s -X POST "$server_url/bo/align_grid_best" \
                 -H "Content-Type: application/json"]

        if {![catch {exec {*}$bestCmd} bestRes]} {
            set got_best 0
            if {[regexp {"best_x":\s*([-+]?[0-9]*\.?[0-9]+)} $bestRes match best_x] &&
                [regexp {"best_y":\s*([-+]?[0-9]*\.?[0-9]+)} $bestRes match best_y] &&
                [regexp {"best_z":\s*([-+]?[0-9]*\.?[0-9]+)} $bestRes match best_z]} {
                set got_best 1
            }

            if { $got_best } {
                move sample_x to $best_x
                move sample_y to $best_y
                move sample_z to $best_z
                wait_for_devices sample_x sample_y sample_z

                set best_score 0.0
                regexp {"score":\s*([-+]?[0-9]*\.?[0-9]+)} $bestRes match best_score
                send_operation_update "Best: x=$best_x y=$best_y z=$best_z (score=$best_score)"
            }
        }
    }

    # Return to face-on view
    move gonio_phi to $phi_face
    wait_for_devices gonio_phi

    send_operation_update "autoAlignGrid done: x=$sample_x y=$sample_y z=$sample_z phi=$gonio_phi"
    return OK
}
