proc fourCorners_initialize { } {
    global BLnum
	global pyExe
    variable beamlineID
    set BLnum [regsub -all {BL|-} $beamlineID ""]
    set pyExe "/home/blctl/miniforge/envs/blctl/bin/python"
    send_operation_update "init fourCorners for bl $BLnum"
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


proc fourCorners_start { args } {
    # --- Help ---
    if { [llength $args] > 0 && ([lindex $args 0] eq "-h" || [lindex $args 0] eq "--help") } {
        send_operation_update "Usage: fourCorners <dirname> ?--re-detect-radius? ?--dry-run? ?--focus-range N? ?--focus-iters N?"
        send_operation_update "  dirname              — scan directory (must contain output_json_2/)"
        send_operation_update "  --re-detect-radius   — auto-detect radius from live camera instead"
        send_operation_update "                         of using the value in wells.json (optional)"
        send_operation_update "  --dry-run            — move to each well but skip the nudge;"
        send_operation_update "                         only record positions and write diagnostics"
        send_operation_update "  --focus-range <um>   — autofocus Z search range in um (default: 500)"
        send_operation_update "  --focus-iters <n>    — max autofocus iterations (default: 12)"
        send_operation_update ""
        send_operation_update "Moves to four corner wells — (2,1), (1,1), (1,9), (2,10) — and"
        send_operation_update "refines each position using ring correlation on the off-axis camera."
        send_operation_update "Writes refined motor positions to <dirname>/output_json_2/four_corners.json."
        send_operation_update "Requires optCirc to have been run first."
        return OK
    }

    # --- Parse args ---
    set dirname ""
    set redetect 0
    set dryrun 0
    set focus_range 500
    set focus_iters 12
    set i 0
    while {$i < [llength $args]} {
        set a [lindex $args $i]
        if { $a eq "--re-detect-radius" } {
            set redetect 1
        } elseif { $a eq "--dry-run" } {
            set dryrun 1
        } elseif { $a eq "--focus-range" } {
            incr i
            set focus_range [lindex $args $i]
        } elseif { $a eq "--focus-iters" } {
            incr i
            set focus_iters [lindex $args $i]
        } elseif { $dirname eq "" } {
            set dirname $a
        }
        incr i
    }
    if { $dirname eq "" } {
        send_operation_update "ERROR: dirname is required. Run fourCorners --help for usage."
        return FAIL
    }

    variable sample_x
    variable sample_y
    variable sample_z
    variable gonio_phi
    global pyExe
    global offaxis_url

    # The four corner wells: A=(2,1) B=(1,1) C=(1,9) D=(2,10)
    set corners { {2 1} {1 1} {1 9} {2 10} }
    set corner_names {A B C D}

    # Verify mapping.json exists
    set mapping_file "$dirname/output_json_2/mapping.json"
    if {![file exists $mapping_file]} {
        send_operation_update "ERROR: $mapping_file not found. Run optCirc first."
        return FAIL
    }

    # Get well radius: from wells.json by default, or auto-detect
    if { $redetect } {
        set radius "auto"
        send_operation_update "Will auto-detect radius from live camera"
    } else {
        set wells_file "$dirname/output_json_2/wells.json"
        set radius [string trim [exec $pyExe -c "import json; ws=json.load(open('$wells_file')); print(ws\[0\]\['r'\])"]]
        send_operation_update "Well radius from wells.json: $radius px"
    }

    # Collect refined positions: list of {name wa wb x y z phi}
    set results {}

    for {set i 0} {$i < [llength $corners]} {incr i} {
        set well [lindex $corners $i]
        set name [lindex $corner_names $i]
        set wa [lindex $well 0]
        set wb [lindex $well 1]

        send_operation_update "Corner $name: moving to well ($wa,$wb) ..."

        # Move to the well using the mapping
        set pyCmd "$pyExe -m gridsteer.step2.read $dirname $wa $wb"
        set pyOut [eval exec $pyCmd]
        scan $pyOut "%f %f %f %f" well_x well_y well_z well_phi

        move sample_x to $well_x
        move sample_y to $well_y
        move sample_z to $well_z
        move gonio_phi to $well_phi
        wait_for_devices sample_x sample_y sample_z gonio_phi

        # Autofocus on the inline camera: rotate +90, focus, rotate back
        set phi_plus_90 [expr {$well_phi + 90.0}]
        move gonio_phi to $phi_plus_90
        wait_for_devices gonio_phi
        send_operation_update "Corner $name: autofocusing on inline camera ..."
        goCirc_autofocus $focus_range $focus_iters
        after 1000
        move gonio_phi to $well_phi
        wait_for_devices gonio_phi

        # Let vibration settle and camera stream catch up
        after 1000

        # Grab a frame and find the pixel offset to the true well center
        send_operation_update "Corner $name: refining center ..."
        set diagImg "$dirname/output_json_2/refine_${name}_${wa}_${wb}.png"
        set wellLabel "($wa,$wb)"
        set refineCmd "$pyExe -m gridsteer.step2.refine_center $offaxis_url $radius $diagImg $wellLabel"
        set refineOut [eval exec $refineCmd]
        scan $refineOut "%f %f %f" dx_px dy_px det_radius

        send_operation_update "Corner $name: pixel offset dx=$dx_px dy=$dy_px (detected radius=$det_radius)"

        if { !$dryrun } {
            # Convert pixel offset to image-fraction units for moveSample
            # Camera frame is 720x480 (standard off-axis resolution)
            set dx_frac [expr {-$dx_px / 720.0}]
            set dy_frac [expr {-$dy_px / 480.0}]
            send_operation_update "Corner $name: nudging by frac dx=$dx_frac dy=$dy_frac"
            set moveOp [start_waitable_operation moveSample $dx_frac $dy_frac]
            wait_for_operation_to_finish $moveOp
            after 300
        } else {
            send_operation_update "Corner $name: dry-run, skipping nudge"
        }

        # Record the motor position (current if dry-run, post-nudge otherwise)
        send_operation_update "Corner $name ($wa,$wb): x=$sample_x y=$sample_y z=$sample_z phi=$gonio_phi"

        lappend results [list $name $wa $wb $sample_x $sample_y $sample_z $gonio_phi]
    }

    # Write four_corners.json via Python to avoid Tcl quoting pain
    set outfile "$dirname/output_json_2/four_corners.json"
    set json_arg {}
    foreach r $results {
        lappend json_arg "[lindex $r 0] [lindex $r 1] [lindex $r 2] [lindex $r 3] [lindex $r 4] [lindex $r 5] [lindex $r 6]"
    }
    set pyCmd "$pyExe -m gridsteer.step2.write_four_corners $outfile $json_arg"
    eval exec $pyCmd

    send_operation_update "four_corners.json ready at $outfile"

    # Register the refined 4-point grid in DCSS via gridGroupConfig
    # Each result is {name wa wb x y z phi} — extract {x y z phi} for each corner
    set p0 [list [lindex [lindex $results 0] 3] [lindex [lindex $results 0] 4] [lindex [lindex $results 0] 5] [lindex [lindex $results 0] 6]]
    set p1 [list [lindex [lindex $results 1] 3] [lindex [lindex $results 1] 4] [lindex [lindex $results 1] 5] [lindex [lindex $results 1] 6]]
    set p2 [list [lindex [lindex $results 2] 3] [lindex [lindex $results 2] 4] [lindex [lindex $results 2] 5] [lindex [lindex $results 2] 6]]
    set p3 [list [lindex [lindex $results 3] 3] [lindex [lindex $results 3] 4] [lindex [lindex $results 3] 5] [lindex [lindex $results 3] 6]]

    send_operation_update "Registering 4-point grid: $p0 $p1 $p2 $p3"
    set ggcOp [start_waitable_operation gridGroupConfig add_4PointGrid $p0 $p1 $p2 $p3 sample]
    wait_for_operation_to_finish $ggcOp

    send_operation_update "4-point grid registered in DCSS"
    return OK
}
