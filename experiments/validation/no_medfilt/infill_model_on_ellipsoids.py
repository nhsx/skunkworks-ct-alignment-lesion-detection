"""
Runs through the found ellipsoids, and the known lesions, and runs the infiller on them

"""

from experiments.validation import val_set_find_ellipsoids_infill

out_dir = "no_med_filt"

for infiller in [32, 64]:
    val_set_find_ellipsoids_infill.val_run_infiller(out_dir, infiller=infiller)
