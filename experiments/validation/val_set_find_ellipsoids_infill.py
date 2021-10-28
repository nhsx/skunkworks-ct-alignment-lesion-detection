"""
Defines methods for running through the validation set of known lesion locations, finding points of interest
via ellipsoid detection, and running the infiller DL network against those locations
"""

from ai_ct_scans import sectioning
from ai_ct_scans import data_loading
from ai_ct_scans.data_loading import data_root_directory
import pickle
import torch
import numpy as np
from ai_ct_scans.model_trainers import InfillTrainer
import matplotlib.pyplot as plt


def val_find_ellipsoids(out_dir, filterer=None, min_ellipse_long_axis=10):
    """Run ellipsoid detection on the validation set, saving out locations to a set of .pkl files, one for each scan

    Args:
        out_dir (str): A directory name in which to save the sets of ellipsoids. Will be appended to an existing
        ellipsoid_val_out directory in the extra_data directory
        filterer (method): A method that can be used on 2D images, e.g. filterer(image, (kernel_x, kernel_y))
        min_ellipse_long_axis (int): Minimum length of any 2D ellipse long axis to be allowed as part of an ellipsoid


    """
    validation_set = data_loading.load_validation_set()

    out_dir = data_loading.data_root_directory().parent / "ellipsoid_val_out" / out_dir
    out_dir.mkdir(exist_ok=True, parents=True)

    threshold = 500

    ### setting up sectioner
    filter_type = ["intensity"]
    total_samples = 5000
    model_path = (
        data_root_directory().parent
        / "sectioning_out"
        / f'{"_".join(filter_type)}_{total_samples}'
        / "model.pkl"
    )
    sectioner = sectioning.TextonSectioner()
    sectioner.load(model_path)

    ellipsoid_fitter = sectioning.CTEllipsoidFitter(
        min_area_ratio=0.75,
        min_ellipse_long_axis=min_ellipse_long_axis,
        max_ellipse_long_axis=200,
        max_area=25000,
    )

    sectioner_kwargs = {"full_sub_structure": True, "threshold": threshold}
    filterer = filterer
    filter_kernel = (3, 3)

    # begin looping through scans, via patient loaders that each have body_part and scan loaders
    for patient_key in validation_set.keys():
        patient_dir = data_loading.data_root_directory() / f"{patient_key}"
        patient_loader = data_loading.PatientLoader(patient_dir, rescale_on_load=False)
        bodypart_dict = validation_set[patient_key]
        for bodypart_key in bodypart_dict.keys():
            scan_dict = bodypart_dict[bodypart_key]
            for scan_key in scan_dict.keys():
                scan = patient_loader.__getattribute__(bodypart_key).__getattribute__(
                    scan_key
                )
                scan.load_scan()
                print(f"Scanning {patient_key}_{bodypart_key}_{scan_key}_ellipsoids")
                _, ellipsoid_list, _ = ellipsoid_fitter.draw_ellipsoid_walls(
                    scan.full_scan,
                    sectioner=sectioner,
                    sectioner_kwargs=sectioner_kwargs,
                    filterer=filterer,
                    filter_kernel=filter_kernel,
                    return_sectioned=True,
                )
                curr_path = (
                    out_dir / f"{patient_key}_{bodypart_key}_{scan_key}_ellipsoids.pkl"
                )
                with open(curr_path, "wb") as f:
                    pickle.dump(ellipsoid_list, f)


def val_run_infiller(out_dir, infiller=64):
    """Run an infiller DL model against ellipsoid locations loaded from .pkl files already generated in out_dir via
    val_find_ellipsoids. The known lesion locations will also be run through the model, to check the infiller's
    behaviour when definitely centred on known lesions

    Args:
        out_dir (str): A directory name in which to save the sets of images. Will be appended to an existing
        ellipsoid_val_out directory in the extra_data directory
        infiller (int, 64 or 32): Flag for which InfillTrainer to use, which has an infiller model and
        a batch building method suitable for preparing tensors from the dataset with minimal wrapping of ellipsoid
        locations. Defaults to the 64 pixel wide centrally blanked infiller trained in
        experiments.masked_data_dl.with_blur_64_mask
    """
    validation_set = data_loading.load_validation_set()

    pkl_dir = data_loading.data_root_directory().parent / "ellipsoid_val_out" / out_dir
    im_out_dir = (
        data_loading.data_root_directory().parent
        / "ellipsoid_val_out"
        / out_dir
        / str(infiller)
    )

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = "cpu"

    if infiller == 64:
        infiller_dir = data_root_directory().parent / "infiller_with_blur_64_mask"

        trainer = InfillTrainer(
            num_encoder_convs=6,
            num_decoder_convs=6,
            batch_size=9,
            clear_previous_memmaps=False,
            save_dir=infiller_dir,
            save_freq=1000,
            encoder_filts_per_layer=24,
            decoder_filts_per_layer=24,
            num_dense_layers=1,
            neurons_per_dense=128,
            learning_rate=0.000005,
            blank_width=64,
            kernel_size=7,
            blur_kernel=(5, 5),
            show_outline=True,
        )
        trainer.load_model(trainer.save_dir, "latest_model.pth")
    elif infiller == 32:
        infiller_dir = data_root_directory().parent / "infiller_with_blur"

        trainer = InfillTrainer(
            num_encoder_convs=6,
            num_decoder_convs=6,
            batch_size=3,
            clear_previous_memmaps=False,
            save_dir=infiller_dir,
            save_freq=1000,
            encoder_filts_per_layer=24,
            decoder_filts_per_layer=24,
            num_dense_layers=1,
            neurons_per_dense=128,
            learning_rate=0.00003,
            blank_width=32,
            kernel_size=7,
            blur_kernel=(5, 5),
            show_outline=True,
        )

        trainer.load_model(trainer.save_dir, "model.pth")

    def _get_top_n_ellipsoids(expected_lesion_location, ellipsoid_list, n=3):
        ellipsoid_centres = np.stack([ell["centre"] for ell in ellipsoid_list], axis=0)
        dist_from_expected = np.linalg.norm(
            ellipsoid_centres - expected_lesion_location, axis=1
        )
        min_dist_args = dist_from_expected.argsort()[:n]
        out_ellipsoid_list = [
            ell for ell_i, ell in enumerate(ellipsoid_list) if ell_i in min_dist_args
        ]
        for arg_i, dist_arg in enumerate(min_dist_args):
            out_ellipsoid_list[arg_i]["distance"] = dist_from_expected[dist_arg]
        return out_ellipsoid_list

    for patient_key in validation_set.keys():
        bodypart_dict = validation_set[patient_key]
        for bodypart_key in bodypart_dict.keys():
            scan_dict = bodypart_dict[bodypart_key]
            for scan_key in scan_dict.keys():
                curr_path = (
                    pkl_dir / f"{patient_key}_{bodypart_key}_{scan_key}_ellipsoids.pkl"
                )
                with open(curr_path, "rb") as f:
                    ellipsoid_list = pickle.load(f)

                curr_save_dir = im_out_dir / f"{patient_key}_{bodypart_key}_{scan_key}"
                curr_save_dir.mkdir(exist_ok=True, parents=True)

                for expected_location_i, expected_location in enumerate(
                    validation_set[patient_key][bodypart_key][scan_key]
                ):

                    curr_save_location_dir = curr_save_dir / str(expected_location_i)
                    curr_save_location_dir.mkdir(exist_ok=True, parents=True)
                    log_path = curr_save_location_dir / "log.txt"

                    ellipsoid_matches = _get_top_n_ellipsoids(
                        expected_location, ellipsoid_list=ellipsoid_list, n=10
                    )
                    big_ellipsoids = [
                        ell for ell in ellipsoid_list if ell["volume"] >= 225
                    ]
                    with open(log_path, "w+") as f:
                        f.write(
                            f"Number of ellipses found {len(ellipsoid_list)}\n"
                            f"Number of ellipsoids above 225 volume found {len(big_ellipsoids)}\n"
                            f'Minimum distance {ellipsoid_matches[0]["distance"]}'
                        )
                    # prepend the true lesion location to see how DL does on it even if not discovered by ellipsoid fitting
                    ellipsoid_matches = [
                        {"centre": expected_location, "distance": 0}
                    ] + ellipsoid_matches
                    # off by one error of str vs int patient number labelling
                    patient_index = patient_key - 1
                    patient_indices = np.ones(3, dtype=int) * (patient_index)
                    if bodypart_key == "abdo":
                        bodypart_num = 0
                    else:
                        bodypart_num = 1
                    body_part_indices = np.ones(3, dtype=int) * bodypart_num
                    if scan_key == "scan_1":
                        scan_num = 0
                    else:
                        scan_num = 1
                    scan_num_indices = np.ones(3, dtype=int) * scan_num

                    # load the scan of the patient to get rescaling factors, then clear scan to save memory
                    scan = (
                        trainer.multi_patient_loader.patients[patient_index]
                        .__getattribute__(bodypart_key)
                        .__getattribute__(scan_key)
                    )
                    scan.rescale_on_load = False
                    scan.load_scan()
                    original_scan_shape = np.array(scan.full_scan.shape)
                    scan.full_scan = None
                    scale_factors = (
                        np.array(scan.full_memmap.shape) / original_scan_shape
                    )
                    for ellipsoid_i, ellipsoid in enumerate(ellipsoid_matches):
                        scaled_location = np.array(ellipsoid["centre"]) * scale_factors
                        coords_input_array = []
                        for coord_i in range(3):

                            if coord_i == 0:
                                curr_coords = [
                                    scaled_location[0],
                                    scaled_location[1] - 128,
                                    scaled_location[2] - 128,
                                ]
                            elif coord_i == 1:
                                curr_coords = [
                                    scaled_location[0] - 128,
                                    scaled_location[1],
                                    scaled_location[2] - 128,
                                ]
                            elif coord_i == 2:
                                curr_coords = [
                                    scaled_location[0] - 128,
                                    scaled_location[1] - 128,
                                    scaled_location[2],
                                ]
                            curr_coords = [int(np.round(val)) for val in curr_coords]
                            coords_input_array.append(curr_coords)
                        three_views = trainer.build_batch(
                            patient_indices=patient_indices,
                            body_part_indices=body_part_indices,
                            plane_indices=np.array([0, 1, 2]),
                            scan_num_indices=scan_num_indices,
                            coords_input_array=coords_input_array,
                            batch_size=3,
                            require_above_thresh=False,
                            allow_off_edge=True,
                        )
                        predictions = trainer.model(three_views)

                        input_images = np.squeeze(
                            three_views["labels"].cpu().detach().numpy()
                        )
                        predictions = np.squeeze(predictions.cpu().detach().numpy())

                        f, axes = plt.subplots(3, 3, figsize=[32, 32])
                        for save_i, (input_image, prediction, ax) in enumerate(
                            zip(input_images, predictions, axes)
                        ):
                            ax[0].imshow(
                                input_image, cmap="gray", interpolation="nearest"
                            )
                            ax[1].imshow(
                                prediction, cmap="gray", interpolation="nearest"
                            )
                            ax[2].imshow(
                                np.abs(input_image - prediction),
                                cmap="gray",
                                interpolation="nearest",
                                vmin=0,
                            )
                        axes[0][0].set_title("True region")
                        axes[0][1].set_title("Infiller prediction")
                        axes[0][2].set_title("Difference")
                        axes[0][0].set_ylabel("Axial")
                        axes[1][0].set_ylabel("Coronal")
                        axes[2][0].set_ylabel("Sagittal")
                        location_string = "_".join(
                            [
                                str(int(np.round(element)))
                                for element in ellipsoid["centre"]
                            ]
                        )
                        if ellipsoid_i == 0:
                            curr_fig_path = (
                                curr_save_location_dir
                                / f"{ellipsoid_i}_location_{location_string}.png"
                            )
                        else:
                            curr_fig_path = (
                                curr_save_location_dir
                                / f'{ellipsoid_i}_location_{location_string}_distance_{int(np.round(ellipsoid["distance"]))}_volume_{int(np.round(ellipsoid["volume"]))}.png'
                            )
                        plt.tight_layout()
                        plt.savefig(curr_fig_path)
                        plt.close("all")
