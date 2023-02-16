import functools
from typing import Callable, Optional
import numpy as np
import skimage
import skimage.feature
import joblib
from sklearn.ensemble import RandomForestClassifier

from .segmenter import Segmenter3D


_default_features_func = functools.partial(
    skimage.feature.multiscale_basic_features,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=1,
    sigma_max=16,
)


class RandomForest3DSegmenter(Segmenter3D):
    def __init__(
        self,
        rf_kwargs: Optional[dict] = None,
        features_fn: Callable = _default_features_func,
    ):
        """
        Initialize the segmentation algorithm.

        Arguments:
            rf_kwargs (dict): The keyword arguments to pass to the random forest.

        """
        self.rf_kwargs = rf_kwargs or {}
        self.features_fn = features_fn or (lambda x: x)

        estimators = self.rf_kwargs.pop("n_estimators", 25)
        max_depth = self.rf_kwargs.pop("max_depth", 8)
        n_jobs = self.rf_kwargs.pop("n_jobs", -1)

        self._clf = RandomForestClassifier(
            n_estimators=estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            **self.rf_kwargs
        )

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment the given volume.

        Arguments:
            volume (np.ndarray<any>): The volume to segment.

        Returns:
            np.ndarray<u64>: The segmentation mask.

        """
        # Extract features:
        mask = np.zeros(volume.shape, dtype=np.uint64)

        for z in range(volume.shape[2]):
            mask[:, :, z] = self._segment_slice(volume[:, :, z])

        return mask

    def _segment_slice(self, imgslice: np.ndarray) -> np.ndarray:
        """
        Segment the given slice.

        Arguments:
            slice (np.ndarray<any>): The slice to segment.

        Returns:
            np.ndarray<u64>: The segmentation mask.

        """
        # Extract features:
        features = self.features_fn(imgslice)

        # Segment the slice:
        mask = self._clf.predict(features.reshape(-1, features.shape[-1]))
        mask = mask.reshape(features.shape[:2])

        return mask

    def fit(self, volume: np.ndarray, mask: np.ndarray) -> None:
        """
        Train the segmentation algorithm.

        Arguments:
            volume (np.ndarray<any>): The volume to segment.
            mask (np.ndarray<u64>): The segmentation mask.

        """
        # Extract features:
        # features = self.features_fn(volume)

        # Train the classifier:
        # self._clf.fit(features.reshape(-1, features.shape[-1]), mask.reshape(-1))

        # Extract features:
        features = []
        labels = []

        for z in range(volume.shape[2]):
            f, l = self._fit_slice(volume[:, :, z], mask[:, :, z])
            features.append(f)
            labels.append(l)

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Train the classifier:
        self._clf.fit(features, labels)

    def _fit_slice(self, imgslice: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Train the segmentation algorithm on the given slice.

        Arguments:
            slice (np.ndarray<any>): The slice to segment.
            mask (np.ndarray<u64>): The segmentation mask.

        Returns:
            tuple: The features and labels.

        """
        # Extract features:
        features = self.features_fn(imgslice)

        # Train the classifier:
        features = features.reshape(-1, features.shape[-1])
        mask = mask.reshape(-1)

        return features, mask

    def save(self, path: str) -> None:
        """
        Save the segmentation algorithm.

        Arguments:
            path (str): The path to save the segmentation algorithm to.

        """
        joblib.dump(self._clf, path)

    def load(self, path: str) -> None:
        """
        Load the segmentation algorithm.

        Arguments:
            path (str): The path to load the segmentation algorithm from.

        """
        self._clf = joblib.load(path)
