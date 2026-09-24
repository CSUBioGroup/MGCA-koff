reinitialize
load /mnt/e/Desktop/Bio大修/mgca_hyperparameter_tuning/mgca_final_unbounded_2773_v1/structure/1NFU_receptor_occlusion_2773.pdb, receptor
load /mnt/e/Desktop/Bio大修/case_study_mgca_final/geometry/analysis/selected_factor_xa_05_pose.sdf, factor_xa_05
hide everything, all
show cartoon, receptor and chain A
color gray70, receptor
select high_occlusion, receptor and ((chain A and resi 58+59+60+61+225+226+227+228+229+230+231+232+233+234+235+236+237+238+239+240+241+242+243+244))
color red, high_occlusion
show sticks, factor_xa_05
color cyan, factor_xa_05
select labelled_residues, receptor and chain A and resi 189+195
show sticks, labelled_residues and not name N+C+O
color gray40, labelled_residues
set stick_radius, 0.14, labelled_residues
set stick_radius, 0.22, factor_xa_05
set cartoon_fancy_helices, 1
set cartoon_transparency, 0.0
set depth_cue, 0
set antialias, 2
set ray_opaque_background, 0
set ray_trace_mode, 0
set label_color, black
set label_outline_color, white
set label_size, 14
set label_connector, 0
bg_color white
orient receptor and chain A
zoom receptor and chain A, 3
turn x, -8
turn y, 18
label (receptor and chain A and resi 189 and name CA), "ASP189"
set label_position, [1.5, -0.8, 0.0], receptor and chain A and resi 189
label (receptor and chain A and resi 195 and name CA), "SER195"
set label_position, [-1.5, 1.0, 0.0], receptor and chain A and resi 195
png /mnt/e/Desktop/Bio大修/mgca_hyperparameter_tuning/mgca_final_unbounded_2773_v1/structure/factor_xa_reference_structure_2773.png, 1400, 1800, dpi=300, ray=1
quit
