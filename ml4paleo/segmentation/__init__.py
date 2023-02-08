import abc
import numpy as np
import torch
import torch.nn.functional as F


class Segmentation3D(abc.ABC):
    """
    An abstract base class for 3D segmentation algorithms.

    Does not handle stitching.

    """

    @abc.abstractmethod
    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment the given volume.

        Arguments:
            volume (np.ndarray<any>): The volume to segment.

        Returns:
            np.ndarray<u64>: The segmentation mask.

        """
        pass


class ThresholdSegmentation(Segmentation3D):
    """
    A segmentation algorithm that uses a threshold to segment the data.

    """

    def __init__(self, threshold: float):
        """
        Initialize the segmentation algorithm.

        Arguments:
            threshold (float): The threshold to use.

        """
        self.threshold = threshold

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment the given volume.

        Arguments:
            volume (np.ndarray<any>): The volume to segment.

        Returns:
            np.ndarray<u64>: The segmentation mask.

        """
        return (volume > self.threshold).astype(np.uint64)


class UNetSegmentation(Segmentation3D):
    """
    A segmentation algorithm that uses a U-Net to segment the data.

    """

    def __init__(self, model_path: str = None):
        """
        Initialize the segmentation algorithm.

        Arguments:
            model_path (str): The path to the model to use.

        """
        self.model_path = model_path

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment the given volume.

        Arguments:
            volume (np.ndarray<any>): The volume to segment.

        Returns:
            np.ndarray<u64>: The segmentation mask.

        """
        # Load the model:

        from ml4paleo.segmentation.models import UNet3D

        model = UNet3D()
        if self.model_path is not None:
            model.load_state_dict(torch.load(self.model_path))
        else:
            # Initialize the model with Xavier initialization:
            for m in model.modules():
                if isinstance(m, torch.nn.Conv3d):
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.ConvTranspose3d):
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.BatchNorm3d):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        model.eval()

        # Segment the volume:

        with torch.no_grad():
            volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
            mask = F.softmax(model(volume), dim=1).argmax(dim=1).squeeze(0).numpy()

        return mask
