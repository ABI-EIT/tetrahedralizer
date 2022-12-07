from mesh_lib import assign_from_json
fname=["mock_torso", "lower_lobe_of_left_lung_surface","lower_lobe_of_right_lung_surface","middle_lobe_of_right_lung_surface","upper_lobe_of_left_lung_surface","upper_lobe_of_right_lung_surface"]
confloc="C:\\Users\\samri\\Documents\\LinconAg\\tetrahedralizer\\tetrahedralizer\\mesh_lib"
testit= assign_from_json(fname,confloc)
print(testit)