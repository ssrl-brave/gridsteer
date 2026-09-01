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
        send_operation_update "Usage: fourCorners <dirname>"
        send_operation_update "  dirname  — scan directory (must contain output_json_2/mapping.json)"
        send_operation_update ""
        send_operation_update "Moves to four corner wells — (2,1), (1,1), (1,9), (1,10) — and"
        send_operation_update "refines each position using ring correlation on the off-axis camera."
        send_operation_update "Writes refined motor positions to <dirname>/output_json_2/four_corners.json."
        send_operation_update "Requires optCirc to have been run first."
        return OK
    }

    set dirname [lindex $args 0]
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

    # The four corner wells: A=(2,1) B=(1,1) C=(1,9) D=(1,10)
    set corners { {2 1} {1 1} {1 9} {1 10} }
    set corner_names {A B C D}

    # Read the well radius from mapping.json
    set mapping_file "$dirname/output_json_2/mapping.json"
    if {![file exists $mapping_file]} {
        send_operation_update "ERROR: $mapping_file not found. Run optCirc first."
        return FAIL
    }
    set pyCmd "$pyExe -c \"import json; print(json.load(open('$mapping_file')).get('well_radius_px', 80))\""
    set radius [string trim [eval exec $pyCmd]]
    send_operation_update "Well radius for refinement: $radius px"

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

        # Grab a frame and find the pixel offset to the true well center
        send_operation_update "Corner $name: refining center ..."
        set refineCmd "$pyExe -m gridsteer.step2.refine_center $offaxis_url $radius"
        set refineOut [eval exec $refineCmd]
        scan $refineOut "%f %f" dx_px dy_px

        send_operation_update "Corner $name: pixel offset dx=$dx_px dy=$dy_px"

        # Nudge the sample so the well center lands on the image center
        # moveSampleOnVideo_start works in pixel units on the off-axis view
        moveSampleOnVideo_start sample $dx_px $dy_px

        # Record the refined motor position (read back after the nudge)
        send_operation_update "Corner $name ($wa,$wb): refined x=$sample_x y=$sample_y z=$sample_z phi=$gonio_phi"

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
    return OK
}
