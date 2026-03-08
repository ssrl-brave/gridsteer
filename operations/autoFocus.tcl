
proc autoFocus_initialize { } {
    global af_offaxis_url
    global af_inline_url
    
    # Fetch URLs from the beamline config
    if {[catch {set af_offaxis_url [::config getSnapshotDirectUrl]} err]} {
        send_operation_update "Error fetching Off-axis URL: $err"
        set af_offaxis_url "unknown"
    }
    
    if {[catch {set af_inline_url [::config getSnapshotDirectInlineUrl]} err]} {
        send_operation_update "Error fetching Inline URL: $err"
        set af_inline_url "unknown"
    }

    send_operation_update "Autofocus Initialized. Inline: $af_inline_url | Offaxis: $af_offaxis_url."
}


proc autoFocus_start { {move_type "inline" } {range 250} {iters 12} {server_url "http://pxgpu03:8003"} } {
    global af_offaxis_url
    global af_inline_url
    
    if { $move_type eq "inline" } {
        if {![info exists af_inline_url] || $af_inline_url eq "unknown"} {
            send_operation_update "Error: Inline camera not initialized."
            return ERROR
        }
        set cam_url $af_inline_url
    } elseif { $move_type eq "sample" } {
        if {![info exists af_offaxis_url] || $af_offaxis_url eq "unknown"} {
            send_operation_update "Error: Off-axis camera not initialized."
            return ERROR
        }
        set cam_url $af_offaxis_url
    } else { 
        send_operation_update "move_type must be inline or sample"
        return ERROR
    }
    send_operation_update "Starting $move_type Autofocus..."
    return [run_autofocus_core $cam_url $move_type $range $iters $server_url]
}


proc run_autofocus_core { camera_url move_type range iters server } {
    # 1. Init BO on server
    set initPayload "{\"z_start\": 0.0, \"z_range\": $range}"
    set initCmd [list curl -s -X POST "$server/bo/focus_init" -H "Content-Type: application/json" -d $initPayload]
    
    if {[catch {exec {*}$initCmd} res]} { return ERROR }

    set current_z 0.0

    # 2. Optimization Loop
    for {set i 0} {$i < $iters} {incr i} {
        set stepPayload "{\"url\": \"$camera_url\", \"iter\": $i, \"samp\": 2, \"med_size\": 3}"
        set stepCmd [list curl -s -X POST "$server/bo/focus_step" -H "Content-Type: application/json" -d $stepPayload]
        
        if {[catch {exec {*}$stepCmd} response]} break

        if {[regexp {"next_z":\s*([-+]?[0-9]*\.?[0-9]+)} $response match next_z]} {
            set delta [expr {$next_z - $current_z}]
            
            # Execute physical move
            set move [start_waitable_operation moveSampleOutVideo $move_type $delta]
            wait_for_operation_to_finish $move
            after 300
            
            set current_z $next_z
        } else {
            break 
        }
    }

    # 3. Final Move to Best Focus
    set bestCmd [list curl -s -X POST "$server/bo/best_params" -H "Content-Type: application/json"]
    if {![catch {exec {*}$bestCmd} best_res]} {
        if {[regexp {"z":\s*([-+]?[0-9]*\.?[0-9]+)} $best_res match best_z]} {
            set final_delta [expr {$best_z - $current_z}]
            set move [start_waitable_operation moveSampleOutVideo $move_type $final_delta]
            wait_for_operation_to_finish $move
        }
    }
    return OK
}

