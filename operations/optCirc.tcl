

proc optCirc_initialize { } {
    global BLnum
	global pyExe
    variable beamlineID
    set BLnum [regsub -all {BL|-} $beamlineID ""]
    send_operation_update "init optCirc for bl $BLnum"
    set pyExe "/home/blctl/miniforge/envs/blctl/bin/python"
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

proc panSample_mm { horiz_mm vert_mm } {
    set h [expr $horiz_mm * 1000.0]
    set v [expr $vert_mm * 1000.0]
    moveSampleOnVideo_start sample $h $v
    
    return OK
}


proc zigZagScan { dirname {n_passes 3}  {horiz_step 0.05} {vert_step 0.2} } {
    # scans the sample in a zig zag pattern. Start this after align tip, with cursor
    # at the tip of the grid, at the center, in a face-on view. 
    # The sample will move such that the cursor travels towards the base along a zigzag.. 
    global BLnum
    global pyExe
    global offaxis_url
    global inline_url
    send_operation_update "starting zigZag scan for BL=${BLnum}"
    variable sample_x
    variable sample_y
    variable sample_z
    variable gonio_phi

    set count 0
    # roughly the grid width
    set scan_height 2.5
    set vert_dir 1
    set horiz_dir -1
    
    # --- Initial Setup Move to upper corner from center---
    panSample_mm 0.0 -1.25
        
    set horiz_incr [expr $horiz_dir * $horiz_step]
    
    # Move up/down and towards base, switching vertical direction once scan_height is exceeded
    for {set i 0} {$i < $n_passes} {incr i} {
      
      set vert_incr [expr $vert_dir * $vert_step]
      set vert_moves [expr int($scan_height / $vert_step)]
      
      # loop through the vertical steps
	  set pass_id [expr $i+1]
	  send_operation_update "Pass $pass_id / $n_passes"
      for {set j 0} {$j < $vert_moves} {incr j} {
          
        # Pan the sample: 
        panSample_mm $horiz_incr $vert_incr

        set pyCmd "$pyExe -m gridsteer.snapshot $BLnum $dirname $sample_x $sample_y $sample_z $gonio_phi 0 $count -o $offaxis_url -i $inline_url"
        set pyOut [eval exec $pyCmd]
        #send_operation_update "got python output: =$pyOut"
        set count [expr $count+1]
      }
      
      # Flip vertical direction
      set vert_dir [expr -$vert_dir]
    }
    
    return OK
}


proc optCirc_start { dirname {n_passes 3}  {horiz_step 0.05} {vert_step 0.2} } {
    # access the current motor positions
    variable sample_x
    variable sample_y
    variable sample_z
    variable gonio_phi
    global pyExe
          
    send_operation_update "sample x,y,z,phi: $sample_x, $sample_y, $sample_z, $gonio_phi"
    send_operation_update "python executable: $pyExe"
    send_operation_update "Will write to dirname: $dirname"

    # log the current motor positions
    set start_x $sample_x
    set start_y $sample_y
    set start_z $sample_z
    set start_G $gonio_phi

    zigZagScan $dirname $n_passes $horiz_step $vert_step
    
    ## reset the sample to starting position
    move sample_x to $start_x
    move sample_y to $start_y
    move sample_z to $start_z
    move gonio_phi to $start_G
    wait_for_devices sample_x sample_y sample_z gonio_phi

	# Step 1.5: Auto-detect well radius from the first snapshot
	send_operation_update "Detecting well radius from first snapshot ..."
	set radiusCmd "$pyExe -m gridsteer.step1_5.find_radius $dirname/test0.npz --output-dir $dirname"
	send_operation_update "radiusCmd: $radiusCmd"
	set detected_radius [string trim [eval exec $radiusCmd]]
	if {$detected_radius eq "NaN" || ![string is double $detected_radius]} {
		send_operation_update "WARNING: radius detection failed (got '$detected_radius'), falling back to 115"
		set detected_radius 115
	} else {
		set detected_radius [expr {int(round($detected_radius))}]
		send_operation_update "Detected well radius: $detected_radius px"
	}

	# Step 2: Call the circle map optimizer with detected radius
	send_operation_update "Running the optimizer to map out circular wells in the grid ... "
    set pyCmd "$pyExe -m gridsteer.step2.main $dirname --target_radius $detected_radius --outdir $dirname"
	send_operation_update "pyCmd: $pyCmd"
    set pyOut [eval exec $pyCmd]

    return OK
}

