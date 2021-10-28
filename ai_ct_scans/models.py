import torch
from torch import nn
import matplotlib.pyplot as plt

plt.ion()

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"


class Infiller(nn.Module):
    """A class for building a model capapble of predicting masked central patches in images"""

    def __init__(
        self,
        input_height=256,
        input_width=256,
        output_height=256,
        output_width=256,
        num_encoder_convs=3,
        encoder_filts_per_layer=10,
        neurons_per_dense=512,
        num_dense_layers=3,
        decoder_filts_per_layer=10,
        kernel_size=3,
        num_decoder_convs=3,
    ):
        """Instantiate the various layers

        Args:
            input_height (int): Height of images model should expect
            input_width (int): Width of images model should expect
            output_height (int): Height that model should output, for current Unet architecture must be same as
            input_height
            output_width (int): Width that model should output, for current Unet architecture must be same as
            input_width
            num_encoder_convs (int): Number of Encoder layers to apply
            encoder_filts_per_layer (int): Number of filters to apply at the first layer of the encoder
            neurons_per_dense (int): Number of neurons to use for each layer other than first and last of latent space
            bridge
            num_dense_layers (int): Number of dense layers to use between first and last of latent space bridge
            decoder_filts_per_layer (int): Number of filters to apply at the first layer of the decoder
            kernel_size (tuple of ints or int): Kernel shape to apply in all convolutional layers
            num_decoder_convs (int): Number of Decoder layers to apply
        """

        super(Infiller, self).__init__()

        self.kernel_size = kernel_size
        self.conv_bias = True
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.activation = nn.ELU().to(dev)
        self.num_encoder_convs = num_encoder_convs
        self.encoder_filts_per_layer = encoder_filts_per_layer
        if self.num_encoder_convs < 3:
            raise ValueError(
                "num_encoder_convs must be at least 3 to allow for reshapings performed internally"
            )
        self.encoder_convs = self.build_encoder_convs()
        self.neurons_per_dense = neurons_per_dense
        self.num_dense_layers = num_dense_layers
        self.decoder_filts_per_layer = decoder_filts_per_layer
        self.num_decoder_convs = num_decoder_convs
        if self.num_decoder_convs != self.num_encoder_convs:
            raise ValueError(
                "Current architecture similar to Unet, requires num_encoder_convs==num_decoder_convs"
            )
        if self.encoder_filts_per_layer != self.decoder_filts_per_layer:
            raise ValueError(
                "Current architecture similar to Unet, requires encoder_filts_per_layer==decoder_filts_per_layer"
            )
        self.bridge_input_channels = (
            self.encoder_filts_per_layer * 2 ** self.num_encoder_convs
        )
        self.bridge_input_output_channels = 8
        # self.bridge_input_conv = nn.Conv2d(self.bridge_input_channels, self.bridge_input_output_channels,
        #                                    kernel_size=self.kernel_size,
        #                                    padding='same').to(dev)
        self.decoder_convs = self.build_decoder_convs()
        self.bridge_output_height = int(
            self.input_height / (2 ** self.num_encoder_convs)
        )
        self.bridge_output_width = int(self.input_width / (2 ** self.num_encoder_convs))
        # self.bridge_output_neurons = self.output_height * self.output_width
        self.bridge_input_output_neurons = self.bridge_input_output_channels * (
            self.bridge_output_width * self.bridge_output_height
        )

        # self.latent_space_bridge = self.build_latent_space_bridge()
        self.final_conv = nn.Conv2d(
            self.encoder_filts_per_layer * 2,
            1,
            self.kernel_size,
            padding="same",
            bias=self.conv_bias,
        ).to(dev)

    def forward(self, batch):
        """Perform the forward pass of the model. Currently a Unet implementation

        Args:
            batch (dict of tensors): A batch, e.g. as built by ai_ct_scans.model_trainers.InfillTrainer.build_batch,
            having at minimum a 4D stack of tensors at 'input_images'

        Returns:
            (Tensor): A tensor the same shape as input

        """
        x = batch["input_images"]
        conv_outs = []
        conv_outs.append(self.activation(self.encoder_convs[0](x)))
        for layer in self.encoder_convs[1:]:
            conv_outs.append(self.activation(layer(conv_outs[-1])))

        """
        The below commented code was useful for connecting the latent space bridge, which could include information
        about body part, view and original coordinates to the model. It was disconnected as satisfactory performance
        was observed without its use, but may be useful to retain for future
        # reshape and apply dense layers
        x = self.bridge_input_conv(conv_outs[-1])
        x = x.view([x.shape[0], self.bridge_input_output_neurons])
        x = self.activation(self.latent_space_bridge[0](x))
        # inform the network of the plane it is looking at, the body part, and the original location
        x = torch.cat([x, batch['input_planes'], batch['input_body_part'], batch['input_coords']], dim=1)
        # some dense layers
        x = self.activation(self.latent_space_bridge[1](x))
        for layer in self.latent_space_bridge[2:-1]:
            x = self.activation(layer(x)) + x
        # final dense layer brings back to a number of neurons that can be shaped into expected output shape
        x = self.activation(self.latent_space_bridge[-1](x))
        x = x.view(x.shape[0], self.bridge_input_output_channels, self.bridge_output_height, self.bridge_output_width)
        # concat bridge output with unet encoder output
        x = torch.cat([conv_outs[0], x], dim=1)

        # apply decoder convolutions
        x = self.activation(self.decoder_convs[0](x))
        """

        # reverse conv_outs such that they can be applied with skip connections in Unet structure
        conv_outs = conv_outs[::-1]
        x = conv_outs[0]

        # concat encoder outputs with unet decoder outputs and apply decoder layers sequentially
        for dec_layer, conv_out in zip(self.decoder_convs, conv_outs[:-1]):
            x = torch.cat([conv_out, x], dim=1)
            x = self.activation(dec_layer(x))
        # no activation or residual connection on final layer, as this is a regression problem where any real valued
        # number is valid as a model prediction
        x = torch.cat([conv_outs[-1], x], dim=1)
        x = self.final_conv(x)
        return x

    def build_encoder_convs(self):
        """Create a ModuleList of SingleEncoderLayers with exponentially increasing filter number after each layer,
        in Unet style

        Returns:
            (ModuleList): The layers of the encoder

        """
        layers = nn.ModuleList()
        layers.append(
            nn.Conv2d(
                1, self.encoder_filts_per_layer, 3, padding="same", bias=False
            ).to(dev)
        )
        for i in range(self.num_encoder_convs):
            layers.append(
                SingleEncoderLayer(
                    self.encoder_filts_per_layer * 2 ** i,
                    self.encoder_filts_per_layer * 2 ** (i + 1),
                    self.kernel_size,
                    conv_bias=self.conv_bias,
                )
            )

        return layers

    def build_latent_space_bridge(self):
        """Create a ModuleList of Linear layers that define the 'latent space bridge', connecting the output of the
        convolutional encoder to the convolutional decoder, and allows for additional neurons defining the plane, body
        part and slice location to be passed to the model. Not currently included in the forward of the overall model,
        but may be useful in future work

        Returns:
            (ModuleList): The layers of the decoder

        """
        layers = nn.ModuleList()
        layers.append(
            nn.Linear(self.bridge_input_output_neurons, self.neurons_per_dense).to(dev)
        )
        layers.append(
            nn.Linear(self.neurons_per_dense + 8, self.neurons_per_dense).to(dev)
        )
        for _ in range(self.num_dense_layers):
            layers.append(
                nn.Linear(self.neurons_per_dense, self.neurons_per_dense).to(dev)
            )
        layers.append(
            nn.Linear(
                self.neurons_per_dense,
                self.bridge_input_output_channels
                * self.bridge_output_width
                * self.bridge_output_height,
            ).to(dev)
        )
        return layers

    def build_decoder_convs(self):
        """Create a ModuleList of SingleDecoderLayers with exponentially decreasing filter number after each layer

        Returns:
            (ModuleList): The layers of the decoder

        """

        layers = nn.ModuleList()
        for i in list(range(self.num_decoder_convs))[::-1]:
            if i == self.num_decoder_convs - 1:

                # The commented line below is useful for inclusion of the densely connected section of the network,
                # which can allow information about the body part and plane of view to be included at latent space to
                # inform the decoder. Currently left out as performance was satisfactory without its inclusion and
                # training rate was higher without the additional layers
                # input_filts = self.decoder_filts_per_layer * 2 ** (i + 1) + self.bridge_input_output_channels

                # below line defines filters when a densely connected bridge is not included in the model
                input_filts = self.decoder_filts_per_layer * 2 ** (i + 2)
            else:
                input_filts = self.decoder_filts_per_layer * 2 ** (i + 2)
            input_filts = int(input_filts)
            output_filts = int(self.decoder_filts_per_layer * 2 ** i)
            layers.append(
                SingleDecoderLayer(
                    input_filts,
                    output_filts,
                    self.kernel_size,
                    conv_bias=self.conv_bias,
                )
            )

        return layers


class SingleEncoderLayer(nn.Module):
    def __init__(
        self, num_input_filters, num_output_filters, kernel_size, conv_bias=True
    ):
        """Two layer network that performs a convolution followed by a max pool of stride 2

        Args:
            num_input_filters (int): Number of input filters expected
            num_output_filters (int): Number of output filters
            kernel_size (tuple of two ints or int): The kernel shape to use during convolution
            conv_bias (bool): Whether to use bias in the convolution layer
        """
        super(SingleEncoderLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(
                num_input_filters,
                num_output_filters,
                kernel_size=kernel_size,
                padding="same",
                bias=conv_bias,
            ).to(dev)
        )
        self.layers.append(nn.MaxPool2d(2, stride=2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SingleDecoderLayer(nn.Module):
    def __init__(
        self, num_input_filters, num_output_filters, kernel_size, conv_bias=True
    ):
        """Two layer network that performs a 2x upsample followed by a convolution

        Args:
            num_input_filters (int): Number of input filters expected
            num_output_filters (int): Number of output filters
            kernel_size (tuple of two ints or int): The kernel shape to use during convolution
            conv_bias (bool): Whether to use bias in the convolution layer
        """
        super(SingleDecoderLayer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.layers.append(
            nn.Conv2d(
                num_input_filters,
                num_output_filters,
                kernel_size=kernel_size,
                padding="same",
                padding_mode="reflect",
                bias=conv_bias,
            ).to(dev)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
