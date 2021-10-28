"""
Runs through the validation set of patients, finding all valid ellipsoids, and pickles them out for rapid inspection
in another script. No median filtering is used after the sectioner in this version

"""

from experiments.validation import val_set_find_ellipsoids_infill
from scipy.signal import medfilt2d

out_dir = "med_filt_long_ellipse_axis_15"
filterer = medfilt2d

val_set_find_ellipsoids_infill.val_find_ellipsoids(
    out_dir=out_dir, filterer=filterer, min_ellipse_long_axis=15
)
