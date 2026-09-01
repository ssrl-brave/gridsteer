
proc centerGoodLimits_initialize { } {
    send_operation_update "centerGoodLimits initialized"
}

proc centerGoodLimits_start { args } {
    # --- Help ---
    if { [llength $args] > 0 && ([lindex $args 0] eq "-h" || [lindex $args 0] eq "--help") } {
        send_operation_update "Usage: centerGoodLimits"
        send_operation_update ""
        send_operation_update "Moves sample_x, sample_y, and sample_z to the midpoint"
        send_operation_update "of their getGoodLimits ranges. Useful as a starting point"
        send_operation_update "before alignment operations."
        return OK
    }

    variable sample_x
    variable sample_y
    variable sample_z

    # Get motor limits
    foreach {x_lo x_hi} [getGoodLimits sample_x] break
    foreach {y_lo y_hi} [getGoodLimits sample_y] break
    foreach {z_lo z_hi} [getGoodLimits sample_z] break

    set x_mid [expr {($x_lo + $x_hi) / 2.0}]
    set y_mid [expr {($y_lo + $y_hi) / 2.0}]
    set z_mid [expr {($z_lo + $z_hi) / 2.0}]

    send_operation_update "Motor limits:"
    send_operation_update "  sample_x: $x_lo to $x_hi (mid: $x_mid)"
    send_operation_update "  sample_y: $y_lo to $y_hi (mid: $y_mid)"
    send_operation_update "  sample_z: $z_lo to $z_hi (mid: $z_mid)"

    send_operation_update "Moving to center of limits..."

    move sample_x to $x_mid
    move sample_y to $y_mid
    move sample_z to $z_mid
    wait_for_devices sample_x sample_y sample_z

    send_operation_update "Arrived: x=$sample_x y=$sample_y z=$sample_z"
    return OK
}
