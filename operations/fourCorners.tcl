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


proc fourCorners_sampleFocus { focus_range focus_iters focus_server } {
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
    set afOp [start_waitable_operation autoFocus sample $focus_range $focus_iters $focus_server]
    wait_for_operation_to_finish $afOp

    move $zoomDev to $zStart
    wait_for_devices $zoomDev
    after 2000
    send_operation_update "Restored $zoomDev to $zStart"
}


proc fourCorners_writeMap { dirname } {
    # Full well map interpolated from the refined corners, for
    # goCirc --from-4corners
    global pyExe
    if {[catch {exec $pyExe -m gridsteer.step2.four_corners_map \
            "$dirname/output_json_2" 2>@1} res]} {
        send_operation_update "WARNING: could not write mapping_from_4corners.json: $res"
    } else {
        send_operation_update "goCirc --from-4corners map ready: [lindex [split [string trim $res] "\n"] end]"
    }
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
        send_operation_update "Usage: fourCorners ?dirname? ?--re-detect-radius? ?--snap? ?--snapped? ?--focus-range N? ?--focus-iters N? ?--focus-server URL? ?--sample-focus? ?--inline-phi-offset DEG? ?--from-json? ?--from-optCirc?"
        send_operation_update "  dirname              — scan directory (must contain output_json_2/);"
        send_operation_update "                         default: this beamline's latest optCirc run"
        send_operation_update "  --re-detect-radius   — auto-detect radius from live camera instead"
        send_operation_update "                         of using the value in wells.json (optional)"
        send_operation_update "  --snap               — nudge each corner by the ring-correlation center"
        send_operation_update "                         refinement. Default is no nudge: the refinement"
        send_operation_update "                         still runs and saves refine_*.png for diagnostics"
        send_operation_update "                         (snapping proved unreliable on domed/haloed wells)"
        send_operation_update "  --snapped            — start from mapping.json (detected corners at their"
        send_operation_update "                         snapped positions) instead of the default"
        send_operation_update "                         mappings_on_lattice.json (all corners on the fitted lattice)"
        send_operation_update "  --focus-range <um>   — autofocus Z search range in um (default: 500)"
        send_operation_update "  --focus-iters <n>    — max autofocus iterations (default: 20)"
        send_operation_update "  --focus-server <url> — alignment server for autofocus (default: http://pxgpu03:8003)"
        send_operation_update "  --sample-focus       — autofocus on the sample camera at MedL zoom, face-on"
        send_operation_update "                         (no rotation), then restore the previous zoom; for"
        send_operation_update "                         beamlines without an inline camera (e.g. BL7-1)."
        send_operation_update "                         Default: inline camera at phi+offset (--inline-focus)"
        send_operation_update "  --inline-phi-offset <deg> — phi rotation for inline focus: 90 or -90 (default: 90)"
        send_operation_update "  --from-json          — skip the corner visits; register the 4-point grid"
        send_operation_update "                         from an existing output_json_2/four_corners.json"
        send_operation_update "  --from-optCirc       — skip the corner visits; register the 4-point grid"
        send_operation_update "                         straight from the optCirc mapping (lattice unless --snapped)"
        send_operation_update ""
        send_operation_update "Moves to four corner wells — (2,1), (1,1), (1,9), (2,10) — autofocuses"
        send_operation_update "each, optionally (--snap) refines the center by ring correlation on the"
        send_operation_update "off-axis camera, writes the positions to"
        send_operation_update "<dirname>/output_json_2/four_corners.json and registers the 4-point grid."
        send_operation_update "Also writes mapping_from_4corners.json (all wells, interpolated per row"
        send_operation_update "between the refined corners) for goCirc --from-4corners."
        send_operation_update "Requires optCirc to have been run first."
        return OK
    }

    # --- Parse args ---
    set dirname ""
    set redetect 0
    set nosnap 1
    set focus_range 500
    set focus_iters 20
    set focus_server "http://pxgpu03:8003"
    set inline_phi_offset 90
    set sample_focus 0
    set from_json 0
    set from_optcirc 0
    set use_lattice 1
    set i 0
    while {$i < [llength $args]} {
        set a [lindex $args $i]
        if { $a eq "--re-detect-radius" } {
            set redetect 1
        } elseif { $a eq "--snap" } {
            set nosnap 0
        } elseif { $a eq "--no-snap" } {
            # the default now; kept so old commands still work
            set nosnap 1
        } elseif { $a eq "--snapped" } {
            set use_lattice 0
        } elseif { $a eq "--focus-range" } {
            incr i
            set focus_range [lindex $args $i]
        } elseif { $a eq "--focus-iters" } {
            incr i
            set focus_iters [lindex $args $i]
        } elseif { $a eq "--focus-server" } {
            incr i
            set focus_server [lindex $args $i]
        } elseif { $a eq "--inline-phi-offset" } {
            incr i
            set inline_phi_offset [lindex $args $i]
        } elseif { $a eq "--inline-focus" } {
            set sample_focus 0
        } elseif { $a eq "--sample-focus" } {
            set sample_focus 1
        } elseif { $a eq "--from-json" } {
            set from_json 1
        } elseif { $a eq "--from-optCirc" } {
            set from_optcirc 1
        } elseif { $a eq "--from-optCirc-lattice" } {
            # lattice is the default now; kept so old commands still work
            set from_optcirc 1
        } elseif { [string match -* $a] } {
            send_operation_update "ERROR: unknown option $a. Run fourCorners --help for usage."
            return FAIL
        } elseif { $dirname eq "" } {
            set dirname $a
        }
        incr i
    }
    # No dirname (or "latest"): this beamline's latest optCirc run
    variable beamlineID
    if { $dirname eq "" || $dirname eq "latest" } {
        set dirname "/data/blctl/gridsteer_operation_data/optCirc.$beamlineID.latest"
        if {[catch {file readlink $dirname} target]} {
            send_operation_update "ERROR: no latest optCirc run ($dirname). Run optCirc first or pass a dirname."
            return FAIL
        }
        send_operation_update "Using latest optCirc run: $dirname -> $target"
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
        fourCorners_writeMap $dirname
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

    # Corner start positions: fitted lattice by default (all four corners
    # from one fit), or --snapped for mapping.json (detected corners at
    # their snapped positions)
    if { $use_lattice } {
        set mapping_file "$dirname/output_json_2/mappings_on_lattice.json"
        set readFlag "--lattice"
    } else {
        set mapping_file "$dirname/output_json_2/mapping.json"
        set readFlag ""
    }
    if {![file exists $mapping_file]} {
        send_operation_update "ERROR: $mapping_file not found. Run optCirc (current gridsteer) first, or try --snapped."
        return FAIL
    }
    send_operation_update "Corner positions from $mapping_file"

    if { $from_optcirc } {
        # Corner positions straight from the optCirc mapping;
        # four_corners.json is left untouched
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
        set pyOut [exec $pyExe -m gridsteer.step2.read $dirname $wa $wb {*}$readFlag]
        scan $pyOut "%f %f %f %f" well_x well_y well_z well_phi

        move sample_x to $well_x
        move sample_y to $well_y
        move sample_z to $well_z
        move gonio_phi to $well_phi
        wait_for_devices sample_x sample_y sample_z gonio_phi

        if { $sample_focus } {
            # Sample camera is already face-on at well_phi: focus there at MedL
            send_operation_update "Corner $name: autofocusing on sample camera (MedL) ..."
            fourCorners_sampleFocus $focus_range $focus_iters $focus_server
        } else {
            # Autofocus on the inline camera: rotate to face-on, focus, rotate back
            set phi_inline [expr {$well_phi + $inline_phi_offset}]
            move gonio_phi to $phi_inline
            wait_for_devices gonio_phi
            send_operation_update "Corner $name: autofocusing on inline camera ..."
            set afOp [start_waitable_operation autoFocus inline $focus_range $focus_iters $focus_server]
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

        if { !$nosnap } {
            # Convert pixel offset to image-fraction units for moveSample
            # Camera frame is 720x480 (standard off-axis resolution)
            set dx_frac [expr {-$dx_px / 720.0}]
            set dy_frac [expr {-$dy_px / 480.0}]
            send_operation_update "Corner $name: nudging by frac dx=$dx_frac dy=$dy_frac"
            set moveOp [start_waitable_operation moveSample $dx_frac $dy_frac]
            wait_for_operation_to_finish $moveOp
            after 300
        } else {
            send_operation_update "Corner $name: no nudge (pass --snap to apply it)"
        }

        # Record the motor position (current unless --snap nudged it)
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
    fourCorners_writeMap $dirname

    return [fourCorners_register $results]
}
