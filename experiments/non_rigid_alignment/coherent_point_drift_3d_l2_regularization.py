import cv2
import numpy as np
import cycpd

from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [10, 16]

from ai_ct_scans import (
    data_loading,
    keypoint_alignment,
    point_matching,
    phase_correlation_image_processing,
)

# Load full scan data
patient_dir = data_loading.data_root_directory() / "1"
patient_loader = data_loading.PatientLoader(patient_dir)

patient_loader.abdo.scan_1.load_scan()
patient_loader.abdo.scan_2.load_scan()

scan_1_mid = int(patient_loader.abdo.scan_1.full_scan.shape[1] / 2)
scan_2_mid = int(patient_loader.abdo.scan_2.full_scan.shape[1] / 2)

reference = patient_loader.abdo.scan_1.full_scan[100:-100, :300, :]
image = patient_loader.abdo.scan_2.full_scan[100:-100, :300, :]

# thresh out noise
thresh = 500
reference[reference < thresh] = 0
image[image < thresh] = 0

reference = phase_correlation_image_processing.lmr(
    reference, filter_type=None, radius=3
)
# test = phase_correlation_image_processing.zero_crossings(reference[:,200,:], thresh='auto')
crossings = phase_correlation_image_processing.zero_crossings(reference, thresh="auto")
reference *= crossings
pointcloud_1 = np.where(reference > 400)
# reference = None

image = phase_correlation_image_processing.lmr(image, filter_type=None, radius=3)
crossings = phase_correlation_image_processing.zero_crossings(image, thresh="auto")
image *= crossings
# image = phase_correlation_image_processing.lmr(image, filter_type=None, radius=3)


pointcloud_2 = np.where(image > 400)
# image = None

print(np.amax(reference))
print(len(pointcloud_1[0]))
print(len(pointcloud_2[0]))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    pointcloud_1[0], pointcloud_1[1], pointcloud_1[2], color="b", alpha=0.5, s=0.5
)
ax.scatter(
    pointcloud_2[0], pointcloud_2[1], pointcloud_2[2], color="r", alpha=0.5, s=0.5
)

# import math

dim_1_mid = int(patient_loader.abdo.scan_1.full_scan.shape[1] / 2)
dim_2_mid = int(patient_loader.abdo.scan_1.full_scan.shape[2] / 2)

print(dim_1_mid)
print(dim_2_mid)

x = np.stack(pointcloud_1, axis=1).astype(np.float64)
y = np.stack(pointcloud_2, axis=1).astype(np.float64)

# x_filtered = np.empty((0, 3))
# y_filtered = np.empty((0, 3))

dists_x = x[:, 1:] - np.array([dim_1_mid, dim_2_mid])
x_filtered = x[np.where(np.linalg.norm(dists_x, axis=1) < 150)]
dists_y = y[:, 1:] - np.array([dim_1_mid, dim_2_mid])
y_filtered = y[np.where(np.linalg.norm(dists_y, axis=1) < 150)]

x_filtered = x_filtered[::10]
y_filtered = y_filtered[::10]

# for i in range(x.shape[0]):
#     if math.hypot(x[i, 1] - dim_1_mid, x[i, 2] - dim_2_mid) < 200:
#         x_filtered = np.append(x_filtered, np.array([x[i]]), axis=0)
#
# for i in range(y.shape[0]):
#     if math.hypot(y[i, 1] - dim_1_mid, y[i, 2] - dim_2_mid) < 200:
#         y_filtered = np.append(y_filtered, np.array([y[i]]), axis=0)

print(x_filtered.shape)
print(y_filtered.shape)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    x_filtered[:, 0], x_filtered[:, 1], x_filtered[:, 2], color="b", alpha=0.5, s=0.5
)
ax.scatter(
    y_filtered[:, 0], y_filtered[:, 1], y_filtered[:, 2], color="r", alpha=0.5, s=0.5
)

reg = cycpd.deformable_registration(
    **{
        "X": x_filtered,
        "Y": y_filtered,
        "alpha": 0.2,
        "beta": 20,
        "max_iterations": 500,
    }
)
# reg = cycpd.deformable_registration(**{'X': x_filtered, 'Y': y_filtered, 'max_iterations': 100, 'alpha': 0.2, 'beta': 20})

non_rigid_out = reg.register()

fig = plt.figure()
ax2 = fig.add_subplot(132, projection="3d")
ax2.scatter(
    x_filtered[:, 0], x_filtered[:, 1], x_filtered[:, 2], color="b", alpha=0.5, s=0.5
)
ax2.scatter(
    non_rigid_out[0][:, 0],
    non_rigid_out[0][:, 1],
    non_rigid_out[0][:, 2],
    color="r",
    alpha=0.5,
    s=0.5,
)
ax2.set_title("After alignment with non-rigid CPD")

patient_loader.abdo.scan_1.full_scan = None
patient_loader.abdo.scan_2.full_scan = None
x = None
y = None

matched_indices = point_matching.match_indices(x_filtered, non_rigid_out[0])

# image = patient_loader.abdo.scan_2.full_scan[100:600]
# image = phase_correlation_image_processing.lmr(image, filter_type=None, radius=3)
# image * phase_correlation_image_processing.zero_crossings(image, thresh='auto')
#
# pointcloud_3 = np.where(image > 500)
# image = None
#
# dim_1_mid = int(patient_loader.abdo.scan_2.full_scan.shape[1] / 2)
# dim_2_mid = int(patient_loader.abdo.scan_2.full_scan.shape[2] / 2)
#
# z = np.stack(pointcloud_3, axis=1).astype(np.float64)
#
# z_filtered = np.empty((0, 3))
#
# for i in range(z.shape[0]):
#     if math.hypot(z[i, 1] - dim_1_mid, z[i, 2] - dim_2_mid) < 150:
#         z_filtered = np.append(z_filtered, np.array([z[i]]), axis=0)
#
# print(z_filtered.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(z_filtered[:, 0], z_filtered[:, 1], z_filtered[:, 2], color='b', alpha=0.5, s=0.5)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X = x_filtered[matched_indices[1]]
y = y_filtered[matched_indices[0]]

polyreg = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
polyreg.fit(y, X)

z_transformed = polyreg.predict(z_filtered)
print(z_transformed.shape)

# %matplotlib qt
fig = plt.figure()
ax1 = fig.add_subplot(131, projection="3d")
ax2 = fig.add_subplot(132, projection="3d")
ax1.scatter(
    x_filtered[:, 0], x_filtered[:, 1], x_filtered[:, 2], color="b", alpha=0.5, s=0.5
)
ax1.scatter(
    z_filtered[:, 0], z_filtered[:, 1], z_filtered[:, 2], color="r", alpha=0.5, s=0.5
)
ax1.set_title("Before alignment")
ax2.scatter(
    x_filtered[:, 0], x_filtered[:, 1], x_filtered[:, 2], color="b", alpha=0.5, s=0.5
)
ax2.scatter(
    z_transformed[:, 0],
    z_transformed[:, 1],
    z_transformed[:, 2],
    color="r",
    alpha=0.5,
    s=0.5,
)
ax2.set_title("After alignment")
