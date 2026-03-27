# New standalone Autofocus Initialization
proc goCirc_initialize { } {
    global BLnum
	global pyExe
    variable beamlineID
    set BLnum [regsub -all {BL|-} $beamlineID ""]
    set pyExe "/home/blctl/miniforge/envs/blctl/bin/python"
    send_operation_update "init goCirc for bl $BLnum"
    # define camera URLs
    global offaxis_url
    global inline_url
    if {[catch {set offaxis_url [::config getSnapshotDirectUrl]} err]} {
            send_operation_update "Error fetching URL: $err"
            set offaxis_url "unknown"
        }
    if {[catch {set inline_url [::config getSnapshotDirectInlineUrl]} err]} {
            send_operation_update "Error fetching Inline URL: $err"
            set inline_url "unknown"
        }
    send_operation_update "Offaxis camera feeding: $offaxis_url"
    send_operation_update "Inline camera feeding: $inline_url"
}

proc goCirc_start { dirname well_a well_b {autofocus 0} {focus_range 500} {focus_iters 12} } {
    # access the current motor positions
    variable sample_x
    variable sample_y
    variable sample_z
    variable gonio_phi
    global pyExe
          
    # log the current motor positions
    set start_x $sample_x
    set start_y $sample_y
    set start_z $sample_z
    set start_G $gonio_phi

	send_operation_update "Moving  to well ($well_a,$well_b) ... "
    set pyCmd "$pyExe -m gridsteer.step2.read $dirname $well_a $well_b"
    send_operation_update "PyCMD: $pyCmd"
    set pyOut [eval exec $pyCmd]
    set count [scan $pyOut "%f %f %f %f" well_x well_y well_z well_phi]

    ## reset the sample to starting position
    move sample_x to $well_x 
    move sample_y to $well_y 
    move sample_z to $well_z
    move gonio_phi to $well_phi
    wait_for_devices sample_x sample_y sample_z gonio_phi
	send_operation_update "Done."
    if { $autofocus } {
       # view sample well in the inline camera
       set phi_plus_90 [expr {$well_phi + 90.0}]
       move gonio_phi to $phi_plus_90
       wait_for_devices gonio_phi
       send_operation_update "Beginning autofocus..."
       set autofocus [goCirc_autofocus $focus_range $focus_iters]
       send_operation_update "autofocus complete: $autofocus"
    }
    return OK
}

proc goCirc_autofocus { {z_range 250} {max_iters 12} {serverUrl "http://pxgpu03:8003"} } {
    global inline_url
    # 1. Initialize BO Focus
    set z_start 0.0
    send_operation_update "Initializing BO Autofocus (Range: +/- $z_range)..."
    set initPayload "{\"z_start\": $z_start, \"z_range\": $z_range}"
    set initCmd [list curl -s -X POST "$serverUrl/bo/focus_init" \
             -H "Content-Type: application/json" \
             -d $initPayload]
    #set initCmd "curl -s -X POST $serverUrl/bo/focus_init -H {Content-Type: application/json} -d '$initPayload'"

    if {[catch {exec {*}$initCmd} initRes]} {
        send_operation_update "Focus Init Failed: $initRes"
        return ERROR
    }

    set current_virtual_z $z_start

    # 2. Optimization Loop
    for {set i 0} {$i < $max_iters} {incr i} {
        send_operation_update "Autofocus Iteration $i..."

        set stepPayload "{\"url\": \"$inline_url\", \"iter\": $i, \"samp\": 2, \"med_size\": 3}"
        set stepCmd [list curl -s -X POST "$serverUrl/bo/focus_step" \
             -H "Content-Type: application/json" \
             -d $stepPayload]
        #set stepCmd "curl -s -X POST $serverUrl/bo/focus_step -H {Content-Type: application/json} -d '$stepPayload'"

        if {[catch {exec {*}$stepCmd} response]} {
            send_operation_update "Server communication error: $response"
            break
        }

        # Parse next absolute Z suggested by BO
        if {[regexp {"next_z":\s*([-+]?[0-9]*\.?[0-9]+)} $response match next_z]} {
            # Calculate relative delta for the motor
            set delta [expr {$next_z - $current_virtual_z}]

            send_operation_update "Current: $current_virtual_z | Next: $next_z | Delta: $delta"

            # Execute physical move and wait
            set move [start_waitable_operation moveSampleOutVideo inline $delta]
            wait_for_operation_to_finish $move

            set current_virtual_z $next_z
        } else {
            send_operation_update "Loop termination condition met."
            break
        }
    }

    # 3. Final Homing Move
    # last point might not be the best. Snap back to the peak.
    send_operation_update "Moving to final best focus position..."
    #set bestCmd "curl -s -X POST $serverUrl/bo/best_params -H {Content-Type: application/json}"
    set bestCmd [list curl -s -X POST "$serverUrl/bo/best_params" \
             -H "Content-Type: application/json"]
    #if {[catch {exec {*}$bestCmd} bestRes]} { ... }
    if {![catch {exec {*}$bestCmd} best_res]} {
        if {[regexp {"z":\s*([-+]?[0-9]*\.?[0-9]+)} $best_res match best_z]} {
            set final_delta [expr {$best_z - $current_virtual_z}]
            send_operation_update "Best Z found at $best_z. Homing move: $final_delta"

            set move [start_waitable_operation moveSampleOutVideo inline $final_delta]
            wait_for_operation_to_finish $move
        }
    }

    send_operation_update "Autofocus Complete."
    return OK
}

