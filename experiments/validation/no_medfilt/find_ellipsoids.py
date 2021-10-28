"""
Runs through the validation set of patients, finding all valid ellipsoids, and pickles them out for rapid inspection
in another script. No median filtering is used after the sectioner in this version

"""
from experiments.validation import val_set_find_ellipsoids_infill

out_dir = "no_med_filt"

filterer = None

val_set_find_ellipsoids_infill.val_find_ellipsoids(
    out_dir=out_dir, filterer=filterer, min_ellipse_long_axis=10
)
