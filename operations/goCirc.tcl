# goCirc: move the motors to center a well from the optCirc mapping
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

proc goCirc_start { args } {
    # --- Help ---
    if { [llength $args] > 0 && ([lindex $args 0] eq "-h" || [lindex $args 0] eq "--help") } {
        send_operation_update "Usage: goCirc ?dirname? <well_a> <well_b>"
        send_operation_update "  dirname      — scan directory (must contain output_json_2/mapping.json);"
        send_operation_update "                 default: this beamline's latest optCirc run"
        send_operation_update "                 (/data/blctl/gridsteer_operation_data/optCirc.<beamline>.latest)"
        send_operation_update "  well_a       — well row index (integer)"
        send_operation_update "  well_b       — well column index (integer)"
        send_operation_update ""
        send_operation_update "Moves motors to center the specified well using the mapping"
        send_operation_update "produced by optCirc."
        return OK
    }

    # --- Parse positional args with defaults ---
    # Two args (well_a well_b) or dirname "latest": use the latest optCirc run
    variable beamlineID
    if { [llength $args] == 2 } {
        set args [linsert $args 0 latest]
    }
    if { [llength $args] < 3 } {
        send_operation_update "ERROR: well_a and well_b are required. Run goCirc --help for usage."
        return FAIL
    }
    set dirname     [lindex $args 0]
    set well_a      [lindex $args 1]
    set well_b      [lindex $args 2]
    if { $dirname eq "latest" } {
        set dirname "/data/blctl/gridsteer_operation_data/optCirc.$beamlineID.latest"
        if {[catch {file readlink $dirname} target]} {
            send_operation_update "ERROR: no latest optCirc run ($dirname). Run optCirc first."
            return FAIL
        }
        send_operation_update "Using latest optCirc run: $dirname -> $target"
    }

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
    return OK
}
