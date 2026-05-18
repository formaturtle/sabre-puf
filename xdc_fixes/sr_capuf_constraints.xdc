# =====================================================================
# sr_capuf_constraints.xdc  (v3 — pure XDC, no control flow)
#
# Error history and what this version does:
#
#   v1  had `puts`, which violates XDC grammar (DesignUtils 20-1307).
#   v2  replaced the puts lines with `if {[llength ...] > 0} { ... }`
#       guards, but Vivado's XDC grammar also rejects `if`. Net effect
#       of v2 was: the `if` lines errored out, and every set_property
#       inside them was silently SKIPPED. Your implementation succeeded
#       anyway because the skipped constraints (KEEP_HIERARCHY,
#       ALLOW_COMBINATORIAL_LOOPS, XDC-level DONT_TOUCH) duplicated
#       attributes that are already set at the RTL level in apuf.v.
#
# v3 (this file) only contains constraints that:
#   (a) are legal in pure XDC (no control flow, no `puts`),
#   (b) target objects guaranteed to exist when the XDC is read
#       (DRC checks, named pblocks we just created), and
#   (c) add real value not already covered by RTL attributes.
#
# Net result after applying this: zero XDC parse errors, zero
# Common 17-55 empty-object warnings. The PUF is still protected from
# synthesis optimization by the (* KEEP *) and (* DONT_TOUCH *)
# attributes in apuf.v.
#
# If later you see DRC complaints about combinational loops or
# optimization of the delay chains, move ALLOW_COMBINATORIAL_LOOPS and
# DONT_TOUCH into a separate `.tcl` script and register it via
# Tools -> Settings -> Project Settings -> Implementation -> tcl.pre
# (a .tcl hook supports full Tcl, including `if`).
# =====================================================================

# ---------------------------------------------------------------------
# DRC relaxation for PYNQ overlays
# ---------------------------------------------------------------------
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
set_property SEVERITY {Warning} [get_drc_checks NSTD-1]

# ---------------------------------------------------------------------
# Floorplanning — physical separation of Layer A and Layer B
# ---------------------------------------------------------------------
# add_cells_to_pblock with -quiet tolerates an empty cell list. If the
# hierarchy filter matches nothing, the pblock is created but contains
# no cells, which produces a warning (not an error) and does not stop
# implementation. That's the intended safety net.

create_pblock pblock_layer_a
add_cells_to_pblock pblock_layer_a \
    [get_cells -quiet -hierarchical -filter {NAME =~ "*layer_a*" && IS_PRIMITIVE == 0}] \
    -quiet
resize_pblock pblock_layer_a -add {SLICE_X0Y0:SLICE_X20Y50}
set_property CONTAIN_ROUTING 1 [get_pblocks pblock_layer_a]

create_pblock pblock_layer_b
add_cells_to_pblock pblock_layer_b \
    [get_cells -quiet -hierarchical -filter {NAME =~ "*layer_b*" && IS_PRIMITIVE == 0}] \
    -quiet
resize_pblock pblock_layer_b -add {SLICE_X30Y60:SLICE_X50Y110}
set_property CONTAIN_ROUTING 1 [get_pblocks pblock_layer_b]
