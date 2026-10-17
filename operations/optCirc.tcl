

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

        set count [expr $count+1]
      }
      
      # Flip vertical direction
      set vert_dir [expr -$vert_dir]
    }
    
    return OK
}


proc optCirc_start { args } {
    # --- Help ---
    if { [llength $args] > 0 && ([lindex $args 0] eq "-h" || [lindex $args 0] eq "--help") } {
        send_operation_update "Usage: optCirc ?dirname? ?n_passes? ?horiz_step? ?vert_step? ?scan_height?"
        send_operation_update "  dirname      — output directory for scan data and results (optional;"
        send_operation_update "                 omit or 'auto' for a new run folder under"
        send_operation_update "                 /data/blctl/gridsteer_operation_data)"
        send_operation_update "  n_passes     — number of zigzag passes (default: 3)"
        send_operation_update "  horiz_step   — horizontal step size in mm (default: 0.05)"
        send_operation_update "  vert_step    — vertical step size in mm (default: 0.2)"
        send_operation_update "  scan_height  — total vertical travel in mm (default: 2.5)"
        send_operation_update ""
        send_operation_update "Scans the sample in a zigzag pattern, captures off-axis and inline"
        send_operation_update "camera frames, then runs whole-layout template matching (step2) to"
        send_operation_update "detect circular wells and compute motor centering positions."
        send_operation_update "Results are written to <dirname>/output_json_2/mapping.json."
        send_operation_update "On success, /data/blctl/gridsteer_operation_data/optCirc.<beamline>.latest"
        send_operation_update "is pointed at this run; goCirc and fourCorners use it by default."
        send_operation_update "Use goCirc to move to a specific well afterwards."
        return OK
    }

    # --- Parse positional args with defaults ---
    # No dirname (or "auto", or a leading number = n_passes): new run folder
    variable beamlineID
    set dataRoot /data/blctl/gridsteer_operation_data
    set first [lindex $args 0]
    if { $first eq "" || $first eq "auto" || [string is double -strict $first] } {
        if { $first eq "auto" } {
            set args [lrange $args 1 end]
        }
        if {![file isdirectory $dataRoot]} {
            send_operation_update "ERROR: $dataRoot does not exist"
            return FAIL
        }
        set dirname "$dataRoot/optCirc.$beamlineID.[clock format [clock seconds] -format %Y%m%d_%H%M%S]"
        set args [linsert $args 0 $dirname]
    }
    set dirname    [lindex $args 0]
    set n_passes   [expr {[llength $args] > 1 ? [lindex $args 1] : 3}]
    set horiz_step [expr {[llength $args] > 2 ? [lindex $args 2] : 0.05}]
    set vert_step  [expr {[llength $args] > 3 ? [lindex $args 3] : 0.2}]
    set scan_height [expr {[llength $args] > 4 ? [lindex $args 4] : 2.5}]

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

	# Step 2: Map circular wells via whole-layout template matching.
	# Runs on pxgpu03 over ssh (the beamline host lacks the memory for the
	# mosaic); the scan dir is on /data, shared by both machines.
	set outdir "$dirname/output_json_2"
	set mapHost pxgpu03
	# dirname goes into a remote shell command: allow plain path characters only
	if {![regexp {^[A-Za-z0-9._/-]+$} $dirname]} {
		send_operation_update "ERROR: dirname has characters not allowed for the remote run: $dirname"
		return FAIL
	}
	set remoteCmd "cd /data/blctl && $pyExe -m gridsteer.step2.map_wells $dirname --outdir $outdir"
	send_operation_update "Running the optimizer to map out circular wells in the grid on $mapHost ... "
	send_operation_update "ssh $mapHost: $remoteCmd"
	# 2>@1: stderr warnings are logged, not treated as failure; only a
	# non-zero exit fails
	if {[catch {exec ssh -o BatchMode=yes $mapHost $remoteCmd 2>@1} mapOut]} {
		send_operation_update "ERROR: map_wells on $mapHost failed:"
		foreach line [lrange [split $mapOut "\n"] end-15 end] {
			send_operation_update "  $line"
		}
		return FAIL
	}
	foreach line [lrange [split $mapOut "\n"] end-3 end] {
		send_operation_update "map_wells: $line"
	}

	# Verify mapping.json was produced (required by goCirc)
	set mapping_file "$outdir/mapping.json"
	if {![file exists $mapping_file]} {
		send_operation_update "ERROR: mapping.json was not produced - motor calibration likely failed"
		return FAIL
	}
	send_operation_update "mapping.json ready at $mapping_file"

	# Point this beamline's "latest" link at the run (relative link, so it
	# resolves the same on the beamline host and pxgpu03)
	set dataRoot /data/blctl/gridsteer_operation_data
	set latest "$dataRoot/optCirc.$beamlineID.latest"
	set runDir [file normalize $dirname]
	if { [file dirname $runDir] eq $dataRoot } {
		set target [file tail $runDir]
	} else {
		set target $runDir
	}
	if {[catch {exec ln -sfn $target $latest} err]} {
		send_operation_update "WARNING: could not update $latest: $err"
	} else {
		send_operation_update "$latest -> $target"
	}

    return OK
}

