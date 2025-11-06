proc goCirc_initialize { } {
    global BLnum
	global pyExe
    variable beamlineID
    set BLnum [regsub -all {BL|-} $beamlineID ""]
    set pyExe "/home/blctl/miniforge/envs/blctl/bin/python"
    send_operation_update "init goCirc for bl $BLnum"
}

proc goCirc_start { dirname well_a well_b } {
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

