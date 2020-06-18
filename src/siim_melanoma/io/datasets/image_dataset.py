from pathlib import PurePosixPath
from typing import Any, Dict

from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path

import fsspec
import numpy as np
from PIL import Image
from kedro.config import ConfigLoader, MissingConfigException
from torchvision import transforms, datasets


class ImageDataSet(AbstractDataSet):
    """``ImageDataSet`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> ImageDataSet(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataSet to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)

        self._filepath = filepath


    def _load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """

        conf_paths = ["conf/base", "conf/local"]
        conf_loader = ConfigLoader(conf_paths)
        load_path = self._filepath


        try:
            parameters = conf_loader.get("parameters*", "parameters*/**")
        except MissingConfigException:
            parameters = {}

        transformation = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(parameters['transformations']['random_rotation']),
                                             transforms.Resize(size=(parameters['transformations']['resize']['height'],
                                                                     parameters['transformations']['resize']['width'])),
                                             transforms.ToTensor(),
                                             transforms.Normalize((parameters['transformations']['normalize']['mean']),
                                                                  (parameters['transformations']['normalize']['std']))
                                             ])

        dataset = datasets.ImageFolder(load_path, transform=transformation)

        return dataset

    def _save(self, data: np.ndarray) -> None:
        """Saves image data to the specified filepath.
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        pass
    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset.
        """
        return dict(
            filepath=self._filepath,
        )