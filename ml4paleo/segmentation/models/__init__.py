"""
This module contains PyTorch models for segmentation.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """
    A 3D U-Net model.

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        u_depth: int = 4,
        n_filters: int = 16,
    ):
        """
        Initialize the model.

        Arguments:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            u_depth (int): The depth of the U-Net.
            n_filters (int): The number of filters to use.

        """
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.u_depth = u_depth
        self.n_filters = n_filters

        # Create the encoder:
        self.encoder = nn.ModuleList()
        for i in range(self.u_depth):
            self.encoder.append(
                UNet3DEncoderBlock(
                    self.in_channels if i == 0 else self.n_filters * 2 ** (i - 1),
                    self.n_filters * 2**i,
                )
            )

        # Create the decoder:
        self.decoder = nn.ModuleList()
        for i in range(self.u_depth - 1):
            self.decoder.append(
                UNet3DDecoderBlock(
                    self.n_filters * 2 ** (self.u_depth - i - 1),
                    self.n_filters * 2 ** (self.u_depth - i - 2),
                )
            )

        # Create the output layer:
        self.output = nn.Conv3d(self.n_filters, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the model forward.

        Arguments:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        # Run the encoder:
        encoder_outputs = []
        for i in range(self.u_depth):
            x = self.encoder[i](x)
            encoder_outputs.append(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)

        # Run the decoder:
        for i in range(self.u_depth - 1):
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = torch.cat([x, encoder_outputs[self.u_depth - i - 2]], dim=1)
            x = self.decoder[i](x)

        # Run the output layer:
        x = self.output(x)
        return x


class UNet3DEncoderBlock(nn.Module):
    """
    A 3D U-Net encoder block.

    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the model.

        Arguments:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        """
        super(UNet3DEncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create the layers:
        self.conv1 = nn.Conv3d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv3d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm3d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the model forward.

        Arguments:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x


class UNet3DDecoderBlock(nn.Module):

    """
    A 3D U-Net decoder block.

    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the model.

        Arguments:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.

        """
        super(UNet3DDecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create the layers:
        self.conv1 = nn.Conv3d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv3d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm3d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the model forward.

        Arguments:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x
