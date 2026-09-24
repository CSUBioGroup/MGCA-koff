reinitialize
load /mnt/e/Desktop/Bio大修/mgca_hyperparameter_tuning/mgca_final_unbounded_2773_v1/structure/1NFU_receptor_occlusion_2773.pdb, receptor
load /mnt/e/Desktop/Bio大修/case_study_mgca_final/geometry/analysis/selected_factor_xa_05_pose.sdf, factor_xa_05
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
set cartoon_transparency, 0.05
set cartoon_fancy_helices, 1
set depth_cue, 0
set antialias, 2
set ray_opaque_background, 0
set ray_trace_mode, 1
set label_color, black
set label_size, 18
set label_outline_color, white
set label_connector, 0
bg_color white
orient receptor
zoom receptor, 3
turn x, -8
turn y, 18

png /mnt/e/Desktop/Bio大修/mgca_hyperparameter_tuning/mgca_final_unbounded_2773_v1/structure/factor_xa_structure_2773.png, 1800, 1600, dpi=300, ray=1
quit
