
proc autoFocusLR_initialize { } {
    global afLR_offaxis_url
    global afLR_inline_url

    # Fetch URLs from the beamline config
    if {[catch {set afLR_offaxis_url [::config getSnapshotDirectUrl]} err]} {
        send_operation_update "Error fetching Off-axis URL: $err"
        set afLR_offaxis_url "unknown"
    }

    if {[catch {set afLR_inline_url [::config getSnapshotDirectInlineUrl]} err]} {
        send_operation_update "Error fetching Inline URL: $err"
        set afLR_inline_url "unknown"
    }

    send_operation_update "AutofocusLR Initialized. Inline: $afLR_inline_url | Offaxis: $afLR_offaxis_url."
}


proc autoFocusLR_start { {move_type "inline" } {range 500} {iters 20} {server_url "http://pxgpu03:8004"} } {
    global afLR_offaxis_url
    global afLR_inline_url

    # --- Help ---
    if { $move_type eq "-h" || $move_type eq "--help" } {
        send_operation_update "Usage: autoFocusLR ?camera? ?range? ?iters? ?server_url?"
        send_operation_update "  camera      — inline or sample (default: inline)"
        send_operation_update "  range       — search range in um, +/- about the start (default: 500)"
        send_operation_update "  iters       — max Bayesian optimization iterations (default: 20)"
        send_operation_update "  server_url  — alignment server (default: http://pxgpu03:8004)"
        send_operation_update ""
        send_operation_update "Same as autoFocus, but moves the sample left/right in the chosen"
        send_operation_update "camera's view (moveSampleOnVideo, horizontal only) instead of along"
        send_operation_update "the view axis. Scores image sharpness on the alignment server, then"
        send_operation_update "returns to the best position found. Uses the camera's current zoom."
        return OK
    }

    if { $move_type eq "inline" } {
        if {![info exists afLR_inline_url] || $afLR_inline_url eq "unknown"} {
            send_operation_update "Error: Inline camera not initialized."
            return ERROR
        }
        set cam_url $afLR_inline_url
    } elseif { $move_type eq "sample" } {
        if {![info exists afLR_offaxis_url] || $afLR_offaxis_url eq "unknown"} {
            send_operation_update "Error: Off-axis camera not initialized."
            return ERROR
        }
        set cam_url $afLR_offaxis_url
    } else {
        send_operation_update "move_type must be inline or sample"
        return ERROR
    }
    send_operation_update "Starting $move_type AutofocusLR..."
    return [run_autofocusLR_core $cam_url $move_type $range $iters $server_url]
}


proc run_autofocusLR_core { camera_url move_type range iters server } {
    # 1. Init BO on server (same 1-D focus optimizer; "z" here is the horizontal offset in um)
    set initPayload "{\"z_start\": 0.0, \"z_range\": $range}"
    set initCmd [list curl -s -X POST "$server/bo/focus_init" -H "Content-Type: application/json" -d $initPayload]

    if {[catch {exec {*}$initCmd} res]} { return ERROR }

    set current_h 0.0

    # 2. Optimization Loop
    for {set i 0} {$i < $iters} {incr i} {
        set stepPayload "{\"url\": \"$camera_url\", \"iter\": $i, \"samp\": 2, \"med_size\": 3}"
        set stepCmd [list curl -s -X POST "$server/bo/focus_step" -H "Content-Type: application/json" -d $stepPayload]

        if {[catch {exec {*}$stepCmd} response]} break

        if {[regexp {"next_z":\s*([-+]?[0-9]*\.?[0-9]+)} $response match next_h]} {
            set delta [expr {$next_h - $current_h}]

            # Execute physical move: horizontal in the camera view, no vertical
            set move [start_waitable_operation moveSampleOnVideo $move_type $delta 0]
            wait_for_operation_to_finish $move
            after 300

            set current_h $next_h
        } else {
            break
        }
    }

    # 3. Final Move to Best Position
    set bestCmd [list curl -s -X POST "$server/bo/best_params" -H "Content-Type: application/json"]
    if {![catch {exec {*}$bestCmd} best_res]} {
        if {[regexp {"z":\s*([-+]?[0-9]*\.?[0-9]+)} $best_res match best_h]} {
            set final_delta [expr {$best_h - $current_h}]
            set move [start_waitable_operation moveSampleOnVideo $move_type $final_delta 0]
            wait_for_operation_to_finish $move
        }
    }
    return OK
}
