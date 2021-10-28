from ai_ct_scans import data_loading
import torch
from ai_ct_scans import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from cv2 import blur
from ai_ct_scans import phase_correlation_image_processing
from ai_ct_scans.data_writing import ndarray_to_memmap

plt.ion()

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"


def det(tensor):
    """Detach a torch Tensor to a cpu numpy version

    Args:
        tensor (torch.Tensor): A tensor to be detached and turned into an ndarray

    Returns:
        (ndarray): The same data as an ndarray

    """
    return tensor.cpu().detach().numpy()


def debug_plot(model_out, batch, index=0):
    """Plot the original image, the masked image, and the infilled version. Useful during debugging.

    Args:
        model_out: The infilled image stack from an Infiller model
        batch (dict of tensors): A dictionary with torch Tensors 'labels' and 'input_images'
        index (int): The index of the model's output to compare to the original image and masked version, [0-batch size]


    """
    out = det(model_out)
    ims = det(batch["labels"])
    inputs = det(batch["input_images"])
    f, axes = plt.subplots(1, 3)
    axes = np.ravel(axes)
    axes[0].imshow(ims[index, 0, :, :])
    axes[0].set_title("original image")
    axes[1].imshow(inputs[index, 0, :, :])
    axes[1].set_title("masked inputs")
    axes[2].imshow(out[index, 0, :, :])
    axes[2].set_title("output")


class InfillTrainer:
    """A class for training an ai_ct_scans.models.Infiller network"""

    def __init__(
        self,
        axial_width=256,
        coronal_width=256,
        sagittal_width=256,
        batch_size=8,
        batch_width=256,
        batch_height=256,
        blank_width=64,
        num_encoder_convs=3,
        encoder_filts_per_layer=10,
        neurons_per_dense=512,
        num_dense_layers=3,
        decoder_filts_per_layer=10,
        num_decoder_convs=3,
        kernel_size=3,
        learning_rate=1e-5,
        save_dir=None,
        clear_previous_memmaps=False,
        save_freq=200,
        blur_kernel=None,
        show_outline=False,
    ):
        """Initialises the network and dataset handling, gets the trainer ready for run self.train_for_iterations()

        Args:
            axial_width (int): How wide the model will expect views taken from the axial plane to be in pixels
            coronal_width (int): How wide the model will expect views taken from the coronal plane to be in pixels
            sagittal_width (int):How wide the model will expect views taken from the sagittal plane to be in pixels
            batch_size (int): How many random views to take for a single training iteration (typically 1-8 trialled)
            batch_width (int): How wide the views should be at the point of input to the model in pixels
            batch_height (int): How high the views should be at the point of input to the model in pixels
            blank_width (int): Square size of the centre masked region to be applied in the middle of each view before
            input to network
            num_encoder_convs (int): How many convolution-maxpool steps to build into the model in the encoder
            encoder_filts_per_layer (int): How many filters to include in the first convolution layer (to be doubled at
            each subsequent layer Unet style)
            neurons_per_dense (int): (currently disconnected) How many neurons in each dense layer that connects the
            convolutional layers in the encoder to the convolutional layers in the decoder
            num_dense_layers (int): (currently disconnected) How many layers of dense neurons to use to connect the
            convolutional encoder and decoder layers
            decoder_filts_per_layer (int): (currently must be same as encoder filts_per_layer)
            num_decoder_convs (int): How many upsample-convolutional layers to include in the decoder, currently
            throws an error if not equal to num_encoder_convs to fit Unet style of the network
            kernel_size (int or tuple of two ints): 2D size of kernels used in Conv2D layers
            learning_rate (float): parameter to control the rate at which the model learns, typically <1e-4
            save_dir (pathlib Path): A directory in which to save the model during training
            clear_previous_memmaps (bool): Whether to re-initialise the dataset (i.e. rebuild memmaps off of original
            DICOM data)
            save_freq (int): How often to save the model, every save_freq iterations
            blur_kernel (None or tuple of ints): If not None, apply a blur to the input views before masking and feeding
            into the network. This is theorised to prevent the model getting stuck due to attemptin to recreate high
            frequency random noise
            show_outline (bool): Whether to perform an edge detection and expose these edges in the masked region. This
            helps the model to get the correct shapes at output, without showing it much about the intensity/texture it
            should aim for.
        """
        self.multi_patient_loader = data_loading.MultiPatientLoader()
        # Included just abdo due to simplicity of focusing on one body part
        # Mentioned twice in the array to preserve the index for testing for multiple body parts
        # self.body_parts = ['abdo', 'thorax']
        self.body_parts = ["abdo", "abdo"]
        self.scan_nums = ["scan_1", "scan_2"]
        for patient in self.multi_patient_loader.patients:
            for body_part in self.body_parts:
                for scan_num in self.scan_nums:
                    if clear_previous_memmaps is True:
                        patient.__getattribute__(body_part).__getattribute__(
                            scan_num
                        ).delete_memmap()
                    patient.__getattribute__(body_part).__getattribute__(
                        scan_num
                    ).load_memmap_and_clear_scan()
        self.blur_kernel = blur_kernel
        self.show_outline = show_outline
        self.axial_width = axial_width
        self.coronal_width = coronal_width
        self.sagittal_width = sagittal_width
        self.batch_width = batch_width
        self.batch_height = batch_height
        self.batch_size = batch_size
        self.blank_width = blank_width
        self.loss_weighting_width = int(self.blank_width * 1.5)
        self.slicers = [
            self.random_axial_slicer,
            self.random_coronal_slicer,
            self.random_sagittal_slicer,
        ]
        self.plane_masks = self.plane_mask_builder()
        self.edge_detection_pad = 3
        self.edge_window_width = 2 * self.edge_detection_pad + self.blank_width
        self.edge_detection_mask = np.logical_not(
            self.plane_mask_builder(blank_width=self.edge_window_width)[0]
        )
        self.inv_plane_mask = np.logical_not(self.plane_masks[0])
        self.loss_masks = self.plane_mask_builder(self.loss_weighting_width)
        self.loss_masks = self._convert_loss_masks_to_tensor()
        self.label_masks = [
            np.logical_not(plane_mask) for plane_mask in self.plane_masks
        ]
        self.patient_indices = list(range(len(self.multi_patient_loader.patients)))
        self.model = models.Infiller(
            input_height=self.batch_height,
            input_width=self.batch_width,
            output_height=self.blank_width,
            output_width=self.blank_width,
            num_encoder_convs=num_encoder_convs,
            encoder_filts_per_layer=encoder_filts_per_layer,
            neurons_per_dense=neurons_per_dense,
            num_dense_layers=num_dense_layers,
            decoder_filts_per_layer=decoder_filts_per_layer,
            num_decoder_convs=num_decoder_convs,
            kernel_size=kernel_size,
        )
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        if save_dir is None:
            save_dir = data_loading.data_root_directory().parent / "infiller"
        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir
        self.iteration = 0
        self.last_n_losses = []
        self.loss_num_to_ave_over = 100
        self.latest_loss = np.inf
        self.save_freq = save_freq
        self.input_stack = np.zeros(
            [self.batch_size, 1, self.batch_height, self.batch_width], dtype="float64"
        )
        self.plane_mask_stack = np.zeros_like(self.input_stack)
        self.error_weighting = (
            self.loss_weighting_width ** 2 / self.axial_width ** 2 + 1
        )
        self.best_loss = np.inf

    def _convert_loss_masks_to_tensor(self):
        """Convert existing loss masks, used to reweight the central masked region in the loss function, to tensors

        Returns:
            (tensor): A stack of tensors that are 1 in the central and border regions around the mask, and 0 elsewhere

        """
        return torch.Tensor(self.loss_masks).to(dev)

    def loss(self, model_out, batch):
        """Defines a custom loss function for the network. Weights the loss such that reproduction of the masked region
        (and a small border area around it) contributes to the overall loss on the same order of magnitude as all other
        pixels that were predicted

        Args:
            model_out (torch Tensor): Stack of images that the model has predicted
            batch (dict as built by self.build_batch): The batch that was used for the iteration, which should include
            at least a 'labels' stack of Tensor images of the same shape as model_out

        Returns:
            (torch Tensor): the MSE error in the output prediction after reweighting masked region of prediction

        """
        error = model_out - batch["labels"]
        squared_error = error ** 2
        weighted_error = squared_error * (self.error_weighting - self.loss_masks[0])
        mse = torch.mean(weighted_error)
        return mse

    def train_step(self):
        """Build a single batch, do a single forward and backward pass."""
        batch = self.build_batch()
        self.optimiser.zero_grad()
        out = self.model(batch)
        loss = self.loss(out, batch)
        loss.backward()
        self.optimiser.step()
        detached_loss = loss.cpu().detach().numpy()
        if len(self.last_n_losses) == self.loss_num_to_ave_over:
            self.last_n_losses.pop(0)
            self.last_n_losses.append(detached_loss)
        else:
            self.last_n_losses.append(detached_loss)
        self.iteration += 1
        if self.iteration % 5000 == 0:
            print(f"{self.iteration} iterations complete")

    def train_for_iterations(self, iterations):
        """Train the model for a set number of iterations

        Args:
            iterations (int): Number of iterations to train for


        """
        self.model.train()
        progress_bar = tqdm(range(iterations))
        for _ in progress_bar:
            self.train_step()
            progress_bar.set_description(f"Average loss {np.mean(self.last_n_losses)}")
            if (self.iteration % self.save_freq) == 0:
                self.save_model(self.save_dir)

    def save_model(self, directory, bypass_loss_check=False):
        """Save the model. If it has achieved the best loss, save to 'model.pth' within directory, otherwise save to
        'latest_model.pth'

        Args:
            directory (pathlib Path): A directory in which to save the model


        """
        directory.mkdir(exist_ok=True, parents=True)
        curr_loss = np.mean(self.last_n_losses)
        if curr_loss < self.best_loss or bypass_loss_check:
            torch.save(
                {
                    "iteration": self.iteration,
                    "model_state_dict": self.model.state_dict(),
                    "optimiser_state_dict": self.optimiser.state_dict(),
                    "loss": curr_loss,
                    "running_loss": self.last_n_losses,
                },
                str(directory / "model.pth"),
            )
            self.best_loss = curr_loss
        else:
            torch.save(
                {
                    "iteration": self.iteration,
                    "model_state_dict": self.model.state_dict(),
                    "optimiser_state_dict": self.optimiser.state_dict(),
                    "loss": curr_loss,
                    "running_loss": self.last_n_losses,
                },
                str(directory / "latest_model.pth"),
            )
            self.best_loss = curr_loss

    def load_model(self, directory, model="model.pth"):
        """Load a pretrained model, optimiser state, loss at time of saving, iteration at time of saving

        Args:
            directory (pathlib Path): Directory in which the model is saved
            model (str): Model filename, defaults to 'model.pth'


        """

        checkpoint = torch.load(str(directory / model))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.latest_loss = checkpoint["loss"]
        self.iteration = checkpoint["iteration"]
        self.last_n_losses = checkpoint["running_loss"]
        self.best_loss = checkpoint["loss"]

    def plane_mask_builder(self, blank_width=None):
        """Get a list of logical ndarrays that can be used to mask out the central region of an input image, and
        extract that local region for a 'label' array at output of the model

        Returns:
            (list of 2D ndarrays): A set of masks to apply to the axial, coronal and sagittal views taken during
            building of a batch
        """
        if blank_width is None:
            blank_width = self.blank_width
        axial_mask = np.ones([self.coronal_width, self.sagittal_width], dtype=bool)
        coronal_mask = np.ones([self.axial_width, self.sagittal_width], dtype=bool)
        sagittal_mask = np.ones([self.axial_width, self.coronal_width], dtype=bool)
        for mask in [axial_mask, coronal_mask, sagittal_mask]:
            row_start = int(np.floor(mask.shape[0] / 2) - blank_width / 2)
            col_start = int(np.floor(mask.shape[1] / 2) - blank_width / 2)
            mask[
                row_start : row_start + blank_width, col_start : col_start + blank_width
            ] = False
        # for mask, border_mask in zip([axial_mask, coronal_mask, sagittal_mask], self.border_masks):
        #     mask *= border_mask
        return [axial_mask, coronal_mask, sagittal_mask]

    def border_mask_builder(self):
        """Get a list of logical ndarrays that can be used to mask out the border region of an input image, and
        extract that local region for a 'label' array at output of the model. Applying these should help with aliasing
        effects at the edges of cnn output

        Returns:
            (list of 2D ndarrays): A set of masks to apply to the axial, coronal and sagittal views taken during
            building of a batch
        """
        axial_mask = np.ones([self.coronal_width, self.sagittal_width], dtype=bool)
        coronal_mask = np.ones([self.axial_width, self.sagittal_width], dtype=bool)
        sagittal_mask = np.ones([self.axial_width, self.coronal_width], dtype=bool)
        for mask in [axial_mask, coronal_mask, sagittal_mask]:
            mask[: self.border_width] = False
            mask[-self.border_width :] = False
            mask[:, : self.border_width] = False
            mask[:, -self.border_width :] = False
        return [axial_mask, coronal_mask, sagittal_mask]

    @staticmethod
    def _rand_nd_ints(high, low=0):
        return np.random.randint(low, high)

    def random_coronal_slicer(self, arr, indices=None, allow_off_edge=False):
        """Takes a random crop from a random coronal plane of 3D array arr

        Args:
            allow_off_edge (bool): optional, defaults to False. Whether to allow indices which will take the view
            off the edges of arr
            indices (list of 3 ints): Coordinates at which to take the slice from. 0th and 2nd indices define a top
            left corner of the view, 1st index defines the coronal slice
            arr (ndarray): 3D volume

        Returns:
            (ndarray): 2D image

        """
        if indices is None:
            indices = self._rand_nd_ints(
                high=[
                    arr.shape[0] - self.axial_width,
                    arr.shape[1],
                    arr.shape[2] - self.sagittal_width,
                ]
            )

        if allow_off_edge:
            out_arr = np.zeros([self.axial_width, self.sagittal_width])
            new_out_start_inds = []
            new_out_end_inds = []
            new_arr_start_inds = []
            new_arr_end_inds = []
            non_coronal_inds = [indices[0], indices[2]]
            for ind, width, arr_width in zip(
                non_coronal_inds,
                [self.axial_width, self.sagittal_width],
                [arr.shape[0], arr.shape[2]],
            ):
                if ind < 0:
                    new_out_start_inds.append(-ind)
                else:
                    new_out_start_inds.append(0)
                new_arr_start_inds.append(max(ind, 0))
                remaining_width = width - new_out_start_inds[-1]
                if new_arr_start_inds[-1] + remaining_width > arr_width:
                    new_arr_end_inds.append(arr_width)
                else:
                    new_arr_end_inds.append(new_arr_start_inds[-1] + remaining_width)
                curr_width = new_arr_end_inds[-1] - new_arr_start_inds[-1]
                new_out_end_inds.append(new_out_start_inds[-1] + curr_width)
            out_arr[
                new_out_start_inds[0] : new_out_end_inds[0],
                new_out_start_inds[1] : new_out_end_inds[1],
            ] = arr[
                new_arr_start_inds[0] : new_arr_end_inds[0],
                indices[1],
                new_arr_start_inds[1] : new_arr_end_inds[1],
            ]
        else:
            out_arr = arr[
                indices[0] : indices[0] + self.axial_width,
                indices[1],
                indices[2] : indices[2] + self.sagittal_width,
            ]
        return out_arr, indices

    def random_sagittal_slicer(self, arr, indices=None, allow_off_edge=False):
        """Takes a random crop from a random sagittal plane of 3D array arr

        Args:
            allow_off_edge (bool): optional, defaults to False. Whether to allow indices which will take the view
            off the edges of arr
            indices (list of 3 ints): Coordinates at which to take the slice from. 0th and 1st indices define a top
            left corner of the view, 0th index defines the sagittal slice
            arr (ndarray): 3D volume

        Returns:
            (ndarray): 2D image

        """
        if indices is None:
            indices = self._rand_nd_ints(
                high=[
                    arr.shape[0] - self.axial_width,
                    arr.shape[1] - self.coronal_width,
                    arr.shape[2],
                ]
            )

        if allow_off_edge:
            out_arr = np.zeros([self.axial_width, self.coronal_width])
            new_out_start_inds = []
            new_out_end_inds = []
            new_arr_start_inds = []
            new_arr_end_inds = []
            for ind, width, arr_width in zip(
                indices[:2],
                [self.axial_width, self.coronal_width],
                [arr.shape[0], arr.shape[1]],
            ):
                if ind < 0:
                    new_out_start_inds.append(-ind)
                else:
                    new_out_start_inds.append(0)
                new_arr_start_inds.append(max(ind, 0))
                remaining_width = width - new_out_start_inds[-1]
                if new_arr_start_inds[-1] + remaining_width > arr_width:
                    new_arr_end_inds.append(arr_width)
                else:
                    new_arr_end_inds.append(new_arr_start_inds[-1] + remaining_width)
                curr_width = new_arr_end_inds[-1] - new_arr_start_inds[-1]
                new_out_end_inds.append(new_out_start_inds[-1] + curr_width)
            out_arr[
                new_out_start_inds[0] : new_out_end_inds[0],
                new_out_start_inds[1] : new_out_end_inds[1],
            ] = arr[
                new_arr_start_inds[0] : new_arr_end_inds[0],
                new_arr_start_inds[1] : new_arr_end_inds[1],
                indices[2],
            ]
        else:
            out_arr = arr[
                indices[0] : indices[0] + self.axial_width,
                indices[1] : indices[1] + self.coronal_width,
                indices[2],
            ]
        return out_arr, indices

    def random_axial_slicer(self, arr, indices=None, allow_off_edge=False):
        """Takes a random crop from a random axial plane of 3D array arr

        Args:
            allow_off_edge (bool): optional, defaults to False. Whether to allow indices which will take the view
            off the edges of arr
            indices (list of 3 ints): Coordinates at which to take the slice from. 1st and 2nd indices define a top
            left corner of the view, 0th index defines the axial slice
            arr (ndarray): 3D volume

        Returns:
            (ndarray): 2D image

        """
        if indices is None:
            indices = self._rand_nd_ints(
                high=[
                    arr.shape[0],
                    arr.shape[1] - self.coronal_width,
                    arr.shape[2] - self.sagittal_width,
                ]
            )

        if allow_off_edge:
            out_arr = np.zeros([self.coronal_width, self.sagittal_width])
            new_out_start_inds = []
            new_out_end_inds = []
            new_arr_start_inds = []
            new_arr_end_inds = []
            for ind, width, arr_width in zip(
                indices[1:],
                [self.coronal_width, self.sagittal_width],
                [arr.shape[1], arr.shape[2]],
            ):
                if ind < 0:
                    new_out_start_inds.append(-ind)
                else:
                    new_out_start_inds.append(0)
                new_arr_start_inds.append(max(ind, 0))
                remaining_width = width - new_out_start_inds[-1]
                if new_arr_start_inds[-1] + remaining_width > arr_width:
                    new_arr_end_inds.append(arr_width)
                else:
                    new_arr_end_inds.append(new_arr_start_inds[-1] + remaining_width)
                curr_width = new_arr_end_inds[-1] - new_arr_start_inds[-1]
                new_out_end_inds.append(new_out_start_inds[-1] + curr_width)
            out_arr[
                new_out_start_inds[0] : new_out_end_inds[0],
                new_out_start_inds[1] : new_out_end_inds[1],
            ] = arr[
                indices[0],
                new_arr_start_inds[0] : new_arr_end_inds[0],
                new_arr_start_inds[1] : new_arr_end_inds[1],
            ]
        else:
            out_arr = arr[
                indices[0],
                indices[1] : indices[1] + self.coronal_width,
                indices[2] : indices[2] + self.sagittal_width,
            ]
        return out_arr, indices

    def build_batch(
        self,
        patient_indices=None,
        body_part_indices=None,
        plane_indices=None,
        scan_num_indices=None,
        coords_input_array=None,
        batch_size=None,
        require_above_thresh=True,
        allow_off_edge=False,
    ):
        """Get a batch of inputs and labels for a ai_ct_scans.models.

        Args:
            patient_indices (optional, ndarray of type int): The indices of patients to access for each batch element,
            with zero-indexing (i.e. patient 1 will be at 0). Should be length equal to batch_size, if batch_size used
            body_part_indices (optional, ndarray of type int): Indices of body parts to use to build each batch element,
            0 for abdomen, 1 for thorax
            plane_indices (optional, ndarray of type int): Indices of plane to view the batch element from, 0 for axial,
            1 for coronal, 2 for sagittal
            scan_num_indices (optional, ndarray of type int): Indices of which sequential scan from each patient to use,
            0 for first scan, 1 for second scan
            coords_input_array (optional, list of lendth 3 1D ndarrays of type int): The coordinates to use when
            building each batch element. The coordinate corresponding to the plane_index will be the slice along that
            index, while the other two coordinates will define the top left coordinate of the rectangle extracted from
            that plane
            batch_size (optional, int): How many slices to return for a batch
            require_above_thresh (bool): Whether to reject random slices that do not have any elements above
            self.threshold and seek out new slices until one is found
            allow_off_edge (bool): Whether to allow coords_input_array to cause the output slice to overlap the edges of
            the original scans - useful to ensure that it is possible for every part of a scan to occur at the central
            masked region

        Returns:
            (dict of torch.Tensors): 'input_images': a stack of 2d axial, coronal and sagittal slices
                                     'input_planes': a stack of one hot vectors, that correspond to which view the slice
                                     was taken from. Shape [batch size, 3]
                                     'input_body_part': a stack of one hot vectors, that correspond to the body part the
                                     slice was taken from. Shape [batch size, 2]
                                     'input_coords': a stack of 1D vectors describing the original xyz location of the
                                     slice taken
                                     'labels': a stack of 2D axial, coronal and sagittal slices, representing the data
                                     that was masked at the centre of each element of input_images

        """
        coords_sets = []
        labels = []
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size != self.batch_size:
            # if batch size for evaluation is different to what was used originally for the trainer, resize the input
            # stack and reset batch size
            self.input_stack = np.zeros(
                [batch_size, 1, self.batch_height, self.batch_width]
            )
            self.batch_size = batch_size
        if patient_indices is None:
            patient_indices = np.random.choice(self.patient_indices, batch_size)
        if plane_indices is None:
            plane_indices = np.random.randint(0, 3, batch_size)
        plane_one_hots = []
        if body_part_indices is None:
            body_part_indices = np.random.randint(0, 2, batch_size)
        body_part_one_hots = []
        if scan_num_indices is None:
            scan_num_indices = np.random.randint(0, 2, batch_size)
        if coords_input_array is None:
            coords_input_array = [None for _ in range(batch_size)]

        for i in range(batch_size):

            filled = False
            first_attempt = True
            while not filled:
                if first_attempt:
                    patient_index = patient_indices[i]
                    plane_index = plane_indices[i]
                    body_part_index = body_part_indices[i]
                    scan_index = scan_num_indices[i]
                else:
                    patient_index = np.random.choice(self.patient_indices, 1)[0]
                    plane_index = np.random.randint(0, 3, 1)[0]
                    body_part_index = np.random.randint(0, 2, 1)[0]
                    scan_index = np.random.randint(0, 2, 1)[0]

                plane_array = np.zeros(3)
                plane_array[plane_indices[i]] = 1
                body_part_array = np.zeros(2)
                body_part_array[body_part_indices[i]] = 1
                if plane_indices[i] == 2:
                    if coords_input_array[i] is not None:
                        coords_input_array[i][1], coords_input_array[i][2] = (
                            coords_input_array[i][2],
                            coords_input_array[i][1],
                        )
                    curr_slice, coords = self.slicers[plane_index - 1](
                        self.multi_patient_loader.patients[patient_index]
                        .__getattribute__(self.body_parts[body_part_index])
                        .__getattribute__(self.scan_nums[scan_index])
                        .transpose_memmap,
                        indices=coords_input_array[i],
                        allow_off_edge=allow_off_edge,
                    )
                    # switch coords 1 and 2 because the view is taken from the transpose scan
                    coords[1], coords[2] = coords[2], coords[1]
                else:
                    curr_slice, coords = self.slicers[plane_index](
                        self.multi_patient_loader.patients[patient_index]
                        .__getattribute__(self.body_parts[body_part_index])
                        .__getattribute__(self.scan_nums[scan_index])
                        .full_memmap,
                        indices=coords_input_array[i],
                        allow_off_edge=allow_off_edge,
                    )

                if curr_slice.max() >= 0.05 or not require_above_thresh:
                    filled = True
                else:
                    first_attempt = False
                    continue

                if self.blur_kernel is not None:
                    blurred_slice = blur(curr_slice, self.blur_kernel)
                else:
                    blurred_slice = curr_slice

                self.input_stack[i, 0][:] = blurred_slice[:]
                labels.append(blurred_slice)
                coords_sets.append(coords)
                plane_one_hots.append(plane_array)
                body_part_one_hots.append(body_part_array)
        if self.show_outline is True:
            for im_i, im in enumerate(self.input_stack):
                edges = im[0][self.edge_detection_mask].reshape(
                    [self.edge_window_width, self.edge_window_width]
                )

                edge_weight = np.mean(edges)

                edges = phase_correlation_image_processing.lmr(
                    edges, radius=self.edge_detection_pad
                )
                edges = edges[
                    self.edge_detection_pad : -self.edge_detection_pad,
                    self.edge_detection_pad : -self.edge_detection_pad,
                ]
                edges = phase_correlation_image_processing.zero_crossings(
                    edges, thresh="auto"
                )
                self.input_stack[im_i, 0] *= self.plane_masks[0]
                self.input_stack[im_i, 0][self.inv_plane_mask] += (
                    edges.reshape(-1) * edge_weight
                )

        else:
            self.input_stack *= self.plane_masks[0]

        labels = np.stack(labels)
        labels = labels.reshape([labels.shape[0], 1, *labels.shape[1:]])
        out_dict = {
            "input_images": torch.Tensor(self.input_stack).to(dev),
            "input_planes": torch.Tensor(np.stack(plane_one_hots)).to(dev),
            "input_body_part": torch.Tensor(np.stack(body_part_one_hots)).to(dev),
            "input_coords": torch.Tensor(np.stack(coords_sets)).to(dev),
            "plane_mask": torch.Tensor(self.plane_masks[0]).to(dev),
            "labels": torch.Tensor(labels).to(dev),
        }
        return out_dict

    def process_full_scan(
        self,
        patient_index,
        body_part_index,
        plane_index,
        scan_index,
        batch_size,
        overlap=1,
        save_path=None,
        allow_off_edge=True,
    ):
        """Run a single scan through the model in batches along a chosen axis, patching the prediction of the masked
        region together and subtracting from the real scan to form an 'anomaly scan'.

        Args:
            patient_index (int): The index of a patient as stored in ai_ct_scans.data_loading.MultiPatientLoader, ie
            Patient 1 at 0 index, Patient 2 at 1 etc
            body_part_index (int): Index of a body part, 0 for abdomen, 1 for thorax currently supported
            plane_index (int): Index of plane to stack inputs from, 0 for axial, 1 for coronal, 2 for sagittal
            scan_index (int): 0 for scan 1, 1 for scan 2
            batch_size (int): How many layers to take along the plane_index stack for each batch, 24 with image sizes
            256x256 seems to work fine with 8GB of VRAM
            overlap (int): Number of times to overlap the masked regions to build up an average predicted view. The
            remainder of self.blank_width and overlap should be 0 for well-stitched output
            save_path (pathlib Path): File path at which to save the anomaly scan as a memmap (shape will be appended
            into the filename for ease of reloading with ai_ct_scans.data_loading.load_memmap)
            allow_off_edge (bool): Whether to allow the model input to move off the edge of the scans, so that the
            stitched central blank regions can cover the entire scan. If False, the central square column of masked
            regions will be returned

        Returns:
            (ndarray): 3D volume of input - predicted output

        """

        self.model.eval()
        with torch.inference_mode():
            torch.no_grad()

            if allow_off_edge:
                pad_sizes = (
                    (
                        np.array(
                            [self.axial_width, self.coronal_width, self.sagittal_width]
                        )
                        - self.blank_width
                    )
                    / 2
                ).astype("int")
                pad_sizes[plane_index] = 0
            else:
                pad_sizes = np.zeros(3)

            mask = torch.Tensor(self.inv_plane_mask).to(dev)
            blank_inds = np.ones(batch_size, dtype=int)
            patient_indices = blank_inds * patient_index
            body_part_indices = blank_inds * body_part_index
            plane_indices = blank_inds * plane_index
            scan_num_indices = blank_inds * scan_index

            scan = (
                self.multi_patient_loader.patients[patient_index]
                .__getattribute__(self.body_parts[body_part_index])
                .__getattribute__(self.scan_nums[scan_index])
                .full_memmap
            )

            # anomaly_scan = np.zeros_like(scan)
            anomaly_scan = np.zeros(np.array(scan.shape) + pad_sizes * 2)

            step = int(self.blank_width / overlap)

            if plane_index == 0:
                axial_inds = list(range(0, scan.shape[plane_index], batch_size))
            else:
                axial_inds = list(
                    range(
                        -pad_sizes[0],
                        scan.shape[0] - self.axial_width + pad_sizes[0],
                        step,
                    )
                )
            if plane_index == 1:
                coronal_inds = list(range(0, scan.shape[plane_index], batch_size))
            else:
                coronal_inds = list(
                    range(
                        -pad_sizes[1],
                        scan.shape[1] - self.coronal_width + pad_sizes[1],
                        step,
                    )
                )
            if plane_index == 2:
                sagittal_inds = list(range(0, scan.shape[plane_index], batch_size))
            else:
                sagittal_inds = list(
                    range(
                        -pad_sizes[2],
                        scan.shape[2] - self.sagittal_width + pad_sizes[2],
                        step,
                    )
                )

            for axial_ind in tqdm(axial_inds):
                for coronal_ind in coronal_inds:
                    for sagittal_ind in sagittal_inds:
                        coords_start_array = [axial_ind, coronal_ind, sagittal_ind]
                        if plane_index == 0:
                            curr_batch_size = min(batch_size, scan.shape[0] - axial_ind)
                            coords_input_array = [
                                [axial_ind + i, coronal_ind, sagittal_ind]
                                for i in range(curr_batch_size)
                            ]
                            coords_end_array = [
                                axial_ind + curr_batch_size,
                                coronal_ind + self.batch_height,
                                sagittal_ind + self.batch_width,
                            ]
                        elif plane_index == 1:
                            curr_batch_size = min(
                                batch_size, scan.shape[1] - coronal_ind
                            )
                            coords_input_array = [
                                [axial_ind, coronal_ind + i, sagittal_ind]
                                for i in range(curr_batch_size)
                            ]
                            coords_end_array = [
                                axial_ind + self.batch_height,
                                coronal_ind + curr_batch_size,
                                sagittal_ind + self.batch_width,
                            ]
                        elif plane_index == 2:
                            curr_batch_size = min(
                                batch_size, scan.shape[2] - sagittal_ind
                            )
                            coords_input_array = [
                                [axial_ind, coronal_ind, sagittal_ind + i]
                                for i in range(curr_batch_size)
                            ]
                            coords_end_array = [
                                axial_ind + self.batch_height,
                                coronal_ind + self.batch_width,
                                sagittal_ind + curr_batch_size,
                            ]

                        batch = self.build_batch(
                            patient_indices=patient_indices,
                            body_part_indices=body_part_indices,
                            plane_indices=plane_indices,
                            scan_num_indices=scan_num_indices,
                            coords_input_array=coords_input_array,
                            require_above_thresh=False,
                            batch_size=curr_batch_size,
                            allow_off_edge=allow_off_edge,
                        )
                        diff = torch.squeeze(self.model(batch) - batch["labels"])
                        diff *= mask
                        diff = diff.cpu().detach().numpy()
                        anomaly_scan[
                            coords_start_array[0]
                            + pad_sizes[0] : coords_end_array[0]
                            + pad_sizes[0],
                            coords_start_array[1]
                            + pad_sizes[1] : coords_end_array[1]
                            + pad_sizes[1],
                            coords_start_array[2]
                            + pad_sizes[2] : coords_end_array[2]
                            + pad_sizes[2],
                        ][:] += diff[:]
            # normalise by number of times each reach is overlapped
            anomaly_scan /= overlap ** 2
            anomaly_scan = anomaly_scan[
                pad_sizes[0] : anomaly_scan.shape[0] - pad_sizes[0],
                pad_sizes[1] : anomaly_scan.shape[1] - pad_sizes[1],
                pad_sizes[2] : anomaly_scan.shape[2] - pad_sizes[2],
            ]
            if save_path is not None:
                ndarray_to_memmap(anomaly_scan, save_path.parent, save_path.stem)

        return anomaly_scan
