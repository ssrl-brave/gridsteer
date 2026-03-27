

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


proc zigZagScan { dirname {n_passes 3}  {horiz_step 0.05} {vert_step 0.2} {scan_height 2.5} } {
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
    set vert_dir 1
    set horiz_dir -1
    
    # --- Initial Setup Move to upper corner from center---
    set half_height [expr {$scan_height / 2.0}]
    panSample_mm 0.0 [expr {-$half_height}]
        
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

        # Launch radius detection in background as soon as snapshot is saved
        set convergenceFile "$dirname/radius_${count}.txt"
        exec $pyExe -m gridsteer.step1_5.find_radius $dirname/test${count}.npz --output-dir $dirname > $convergenceFile 2>/dev/null &

        set count [expr $count+1]
      }
      
      # Flip vertical direction
      set vert_dir [expr -$vert_dir]
    }
    
    return OK
}


proc optCirc_start { dirname {n_passes 3}  {horiz_step 0.05} {vert_step 0.2} {scan_height 2.5} } {
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

    zigZagScan $dirname $n_passes $horiz_step $vert_step $scan_height
    
    ## reset the sample to starting position
    move sample_x to $start_x
    move sample_y to $start_y
    move sample_z to $start_z
    move gonio_phi to $start_G
    wait_for_devices sample_x sample_y sample_z gonio_phi

	# Step 1.5: Collect radius detection results (launched in background during scan)
	# Wait briefly for any stragglers to finish writing
	after 3000
	send_operation_update "Collecting radius detection results ..."
	set radii {}
	set n_success 0
	set n_fail 0
	foreach rfile [glob -nocomplain $dirname/radius_*.txt] {
		set fp [open $rfile r]
		set val [string trim [read $fp]]
		close $fp
		if {$val ne "" && $val ne "NaN" && [string is double $val]} {
			lappend radii $val
			incr n_success
		} else {
			incr n_fail
		}
	}
	if {[llength $radii] > 0} {
		# Take the median of all successful detections
		set sorted [lsort -real $radii]
		set n [llength $sorted]
		set mid [expr {$n / 2}]
		set detected_radius [expr {int(round([lindex $sorted $mid]))}]
		send_operation_update "Detected well radius: $detected_radius px (from $n_success/$[expr {$n_success+$n_fail}] frames)"
	} else {
		send_operation_update "ERROR: no frames returned a valid radius - aborting"
		return FAIL
	}

	# Step 2: Call the circle map optimizer with detected radius
	send_operation_update "Running the optimizer to map out circular wells in the grid ... "
    set pyCmd "$pyExe -m gridsteer.step2.main $dirname --target_radius $detected_radius --outdir $dirname"
	send_operation_update "Debug log will be in $dirname/logs/"
	send_operation_update "pyCmd: $pyCmd"
    set pyOut [eval exec $pyCmd]

	# Verify mapping.json was produced (required by goCirc)
	set mapping_file "$dirname/output_json_2/mapping.json"
	if {![file exists $mapping_file]} {
		send_operation_update "ERROR: mapping.json was not produced - motor calibration likely failed. Check $dirname/logs/"
		return FAIL
	}
	send_operation_update "mapping.json ready at $mapping_file"

    return OK
}

