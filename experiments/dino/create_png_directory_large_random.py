"""
Produce a directory filled with .png images extracted from the CT dataset, in the style expected by DINO main_dino.py
"""

from ai_ct_scans.data_loading import data_root_directory, MultiPatientAxialStreamer
import cv2
from tqdm import tqdm

out_dir = (
    data_root_directory().parent / "dino_training_set" / "large_random_set" / "train"
)

out_dir.mkdir(exist_ok=True, parents=True)

streamer = MultiPatientAxialStreamer()

# don't accept any images into the dataset with only CT intensities below threshold
threshold = 500

"""
# actual maximum sensible values within the dataset are unknown, run through a subset of random samples to get sensible
# values
np.random.seed(555)
normalisation_num_images = 1000
sampled_max = -1
sampled_high_percenile = -1
for i in tqdm(range(normalisation_num_images)):
    curr_sample = streamer.stream_next(threshold=threshold, random=True)
    curr_max = curr_sample.max()
    curr_perc = np.percentile(curr_sample, 99.9)
    if curr_max > sampled_max:
        sampled_max = curr_max
    if curr_perc > sampled_high_percenile:
        sampled_high_percenile = curr_perc
print(f'Highest value overall {sampled_max}')
print(f'Highest high percentile {sampled_high_percenile}')
# results: highest 4628, 99.9th percentile 4133.0
"""

streamer.reset_indices()
num_samples = 50000
max_intensity = 4628.0

for i in tqdm(range(num_samples)):
    curr_sample = streamer.stream_next(threshold=threshold, random=True)
    curr_sample[curr_sample > max_intensity] = max_intensity
    curr_sample = curr_sample.astype(float)
    curr_sample /= max_intensity
    curr_sample *= 255
    curr_sample = curr_sample.astype("uint8")
    curr_file_name = str(out_dir / f"{i}.png")
    cv2.imwrite(curr_file_name, curr_sample)
