reinitialize
load C:/Users/NK/AppData/Local/Temp/mgca_factor_xa_render_efoj7tsx/receptor_occlusion.pdb, receptor
load C:/Users/NK/AppData/Local/Temp/mgca_factor_xa_render_efoj7tsx/factor_xa_05.sdf, factor_xa_05
hide everything, all
show cartoon, receptor
color gray80, receptor
spectrum b, blue_white_red, receptor and chain A, minimum=-99, maximum=99
color gray80, receptor and chain B
show sticks, factor_xa_05
color cyan, factor_xa_05
select native_only, receptor and ((chain A and resi 227+228))
select dock_only, receptor and ((chain A and resi 57+61+98+99))
select contact_overlap, receptor and ((chain A and resi 97+174+189+190+191+192+195+213+214+215+216+217+219+220+226))
show sticks, (native_only or dock_only or contact_overlap) and not name N+C+O
color red, native_only
color orange, dock_only
color magenta, contact_overlap
set stick_radius, 0.13, native_only or dock_only or contact_overlap
set stick_radius, 0.22, factor_xa_05
set cartoon_transparency, 0.18
set cartoon_fancy_helices, 1
set depth_cue, 0
set antialias, 2
set ray_opaque_background, 0
set ray_trace_mode, 1
set label_color, black
set label_size, 18
set label_outline_color, white
set label_connector, 1
bg_color white
select pocket_view, byres (receptor within 8.0 of factor_xa_05)
select view_focus, pocket_view or factor_xa_05
orient view_focus
center factor_xa_05
zoom view_focus, 3.5
turn x, -10
turn y, 20
turn z, 5
label (receptor and chain A and resi 57 and name CA), "His57 (H276)"
set label_position, [1.8, 1.5, 0.0], receptor and chain A and resi 57
label (receptor and chain A and resi 189 and name CA), "Asp189 (D413)"
set label_position, [1.8, -1.5, 0.0], receptor and chain A and resi 189
label (receptor and chain A and resi 195 and name CA), "Ser195 (S419)"
set label_position, [-1.8, -1.5, 0.0], receptor and chain A and resi 195
png C:/Users/NK/AppData/Local/Temp/mgca_factor_xa_render_efoj7tsx/factor_xa_pocket_zoom.png, 1800, 1600, dpi=300, ray=1
quit
