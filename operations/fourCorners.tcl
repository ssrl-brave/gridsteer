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


proc fourCorners_sampleFocus { focus_range focus_iters } {
    # For beamlines without an inline camera: zoom the sample (off-axis)
    # camera to MedL, autofocus on it, then restore the original zoom so
    # refine_center sees the same pixel scale as the optCirc scan.
    set zoomDev [::config getStr sampleDigitalZoom]
    if { $zoomDev eq "" } {
        set zoomDev camera_zoom
    }
    variable $zoomDev
    set zStart [set $zoomDev]
    set zMedL [::config getStr bluice.zoomMedLSample]
    if { $zMedL eq "" } {
        set zMedL 3.0
    }

    send_operation_update "Zooming $zoomDev $zStart -> $zMedL (MedL) for sample-camera autofocus"
    move $zoomDev to $zMedL
    wait_for_devices $zoomDev
    # Let the camera stream catch up with the new zoom
    after 2000

    # Same path as running "autoFocus sample": BO on the off-axis camera,
    # moving the sample depth-wise along that camera's view axis
    set afOp [start_waitable_operation autoFocus sample $focus_range $focus_iters]
    wait_for_operation_to_finish $afOp

    move $zoomDev to $zStart
    wait_for_devices $zoomDev
    after 2000
    send_operation_update "Restored $zoomDev to $zStart"
}


proc fourCorners_register { results } {
    # Register the refined 4-point grid in DCSS via gridGroupConfig
    # Each result is {name wa wb x y z phi}. add_4PointGrid reads only
    # {x y z} per corner; the grid angle comes from the corner layout.
    foreach k {0 1 2 3} {
        set p$k [lrange [lindex $results $k] 3 5]
    }

    # The grid snapshot is taken from the live view, so register at the
    # corners' (face-on) phi; at another phi BluIce draws the grid squashed.
    set corner_phi [lindex [lindex $results 0] 6]
    send_operation_update "Moving gonio_phi to corner phi $corner_phi for the grid snapshot"
    move gonio_phi to $corner_phi
    wait_for_devices gonio_phi
    after 1000

    send_operation_update "Registering 4-point grid: $p0 $p1 $p2 $p3"
    # Operation args lose one level of list braces in transit (namespace
    # eval in stoh_start_operation), so wrap each point once more to keep
    # {x y z} intact; unwrapped, the 12 numbers arrive flat.
    set ggcOp [start_waitable_operation gridGroupConfig add_4PointGrid \
        [list $p0] [list $p1] [list $p2] [list $p3] sample]
    wait_for_operation_to_finish $ggcOp

    send_operation_update "4-point grid registered in DCSS"
    return OK
}


proc fourCorners_start { args } {
    # --- Help ---
    if { [llength $args] > 0 && ([lindex $args 0] eq "-h" || [lindex $args 0] eq "--help") } {
        send_operation_update "Usage: fourCorners <dirname> ?--re-detect-radius? ?--dry-run? ?--focus-range N? ?--focus-iters N? ?--inline-focus? ?--from-json? ?--from-optCirc? ?--from-optCirc-lattice?"
        send_operation_update "  dirname              — scan directory (must contain output_json_2/)"
        send_operation_update "  --re-detect-radius   — auto-detect radius from live camera instead"
        send_operation_update "                         of using the value in wells.json (optional)"
        send_operation_update "  --dry-run            — move to each well but skip the nudge;"
        send_operation_update "                         only record positions and write diagnostics"
        send_operation_update "  --focus-range <um>   — autofocus Z search range in um (default: 500)"
        send_operation_update "  --focus-iters <n>    — max autofocus iterations (default: 12)"
        send_operation_update "  --inline-focus       — autofocus on the inline camera at phi+90"
        send_operation_update "                         (default: sample camera at MedL zoom, face-on,"
        send_operation_update "                         then restore the previous zoom)"
        send_operation_update "  --from-json          — skip the corner visits; register the 4-point grid"
        send_operation_update "                         from an existing output_json_2/four_corners.json"
        send_operation_update "  --from-optCirc       — skip the corner visits; register the 4-point grid"
        send_operation_update "                         from the optCirc mapping (no focus or snap)"
        send_operation_update "  --from-optCirc-lattice — as --from-optCirc, but all four corners on the"
        send_operation_update "                         fitted lattice (output_json_2/mappings_on_lattice.json)"
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
    set sample_focus 1
    set from_json 0
    set from_optcirc 0
    set optcirc_lattice 0
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
        } elseif { $a eq "--inline-focus" } {
            set sample_focus 0
        } elseif { $a eq "--from-json" } {
            set from_json 1
        } elseif { $a eq "--from-optCirc" } {
            set from_optcirc 1
        } elseif { $a eq "--from-optCirc-lattice" } {
            set from_optcirc 1
            set optcirc_lattice 1
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

    set outfile "$dirname/output_json_2/four_corners.json"

    if { $from_json } {
        if {![file exists $outfile]} {
            send_operation_update "ERROR: $outfile not found. Run fourCorners without --from-json first."
            return FAIL
        }
        send_operation_update "Loading corners from $outfile (skipping corner visits)"
        set results {}
        foreach line [split [exec $pyExe -m gridsteer.step2.read_four_corners $outfile] "\n"] {
            if { [string trim $line] eq "" } continue
            lappend results $line
            send_operation_update "Corner [lindex $line 0] ([lindex $line 1],[lindex $line 2]): x=[lindex $line 3] y=[lindex $line 4] z=[lindex $line 5] phi=[lindex $line 6]"
        }
        return [fourCorners_register $results]
    }

    # The four corner wells: A=(2,1) B=(1,1) C=(1,9) D=(2,10)
    set corners { {2 1} {1 1} {1 9} {2 10} }
    set corner_names {A B C D}

    # Verify mapping.json exists
    set mapping_file "$dirname/output_json_2/mapping.json"
    if {![file exists $mapping_file]} {
        send_operation_update "ERROR: $mapping_file not found. Run optCirc first."
        return FAIL
    }

    if { $from_optcirc } {
        # Corner positions straight from the optCirc mapping (as goCirc
        # uses them); four_corners.json is left untouched. With the
        # lattice flag, detected corners use their fitted-lattice position
        # instead of their snap.
        set readFlag ""
        if { $optcirc_lattice } {
            set mapping_file "$dirname/output_json_2/mappings_on_lattice.json"
            set readFlag "--lattice"
            if {![file exists $mapping_file]} {
                send_operation_update "ERROR: $mapping_file not found. Re-run the optCirc mapping with the current gridsteer."
                return FAIL
            }
        }
        send_operation_update "Loading corners from $mapping_file (skipping corner visits)"
        set results {}
        for {set i 0} {$i < [llength $corners]} {incr i} {
            set name [lindex $corner_names $i]
            set wa [lindex [lindex $corners $i] 0]
            set wb [lindex [lindex $corners $i] 1]
            set pyOut [exec $pyExe -m gridsteer.step2.read $dirname $wa $wb {*}$readFlag]
            scan $pyOut "%f %f %f %f" well_x well_y well_z well_phi
            send_operation_update "Corner $name ($wa,$wb): x=$well_x y=$well_y z=$well_z phi=$well_phi"
            lappend results [list $name $wa $wb $well_x $well_y $well_z $well_phi]
        }
        return [fourCorners_register $results]
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

        if { $sample_focus } {
            # Sample camera is already face-on at well_phi: focus there at MedL
            send_operation_update "Corner $name: autofocusing on sample camera (MedL) ..."
            fourCorners_sampleFocus $focus_range $focus_iters
        } else {
            # Autofocus on the inline camera: rotate +90, focus, rotate back
            set phi_plus_90 [expr {$well_phi + 90.0}]
            move gonio_phi to $phi_plus_90
            wait_for_devices gonio_phi
            send_operation_update "Corner $name: autofocusing on inline camera ..."
            set afOp [start_waitable_operation autoFocus inline $focus_range $focus_iters]
            wait_for_operation_to_finish $afOp
            after 1000
            move gonio_phi to $well_phi
            wait_for_devices gonio_phi
        }

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
    set json_arg {}
    foreach r $results {
        lappend json_arg "[lindex $r 0] [lindex $r 1] [lindex $r 2] [lindex $r 3] [lindex $r 4] [lindex $r 5] [lindex $r 6]"
    }
    set pyCmd "$pyExe -m gridsteer.step2.write_four_corners $outfile $json_arg"
    eval exec $pyCmd

    send_operation_update "four_corners.json ready at $outfile"

    return [fourCorners_register $results]
}
