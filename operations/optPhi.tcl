
set $BLnum [regsub {BL|-} $beamlinID ""]

proc optPhi_initialize { } {
    puts "init optPhi"
}
proc optPhi_start { dirname args } {
    global BLnum
    set pyExe "/home/blctl/miniforge/envs/blctl/bin/python"

    # access the current motor positions
    variable sample_x
    variable sample_y
    variable sample_z
    variable gonio_phi
          
    send_operation_update "sample x,y,z,phi: $sample_x, $sample_y, $sample_z, $gonio_phi"

    # log the current motor positions
    set start_x $sample_x
    set start_y $sample_y
    set start_z $sample_z
    set start_G $gonio_phi
    send_operation_update "Will write to dirname: $dirname"
    
    set Grange [lindex $args 0]
    set Gstep [lindex $args 1]
    set Glow [expr {$start_G - ($Grange / 2)}]
    set Ghigh [expr {$start_G + ($Grange / 2)}]
    send_operation_update "Will scan gonio phi from $Glow - $Ghigh with stepsize=$Gstep"
        
    # scan the goniometer
    set count 0 
    for {set G $Glow} {$G <= $Ghigh} {set G [expr $G+$Gstep]} {
        move gonio_phi to $G
        wait_for_devices gonio_phi
        set pyCmd "$pyExe -m gridsteer.snapshot $BLnum $dirname $sample_x $sample_y $sample_z $gonio_phi 0 $count"
        set pyOut [eval exec $pyCmd]
        send_operation_update "got python output: =$pyOut"
        set count [expr $count+1]
    }

    # reset the sample to starting position
    move sample_x to $start_x
    move sample_y to $start_y
    move sample_z to $start_z
    move gonio_phi to $start_G
    wait_for_devices sample_x sample_y sample_z gonio_phi
        
    set pyCmd "$pyExe -m gridsteer.optimize_phi $dirname $count"
    set pyOut [eval exec $pyCmd]
    send_operation_update "got python output: =$pyOut"

    return OK
}

