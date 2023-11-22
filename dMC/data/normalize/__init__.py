from abc import ABC, abstractmethod
import logging

import torch

log = logging.getLogger(__name__)


class Normalize(ABC):
    """
    Abstract base class for handling normalization.

    Methods:
        normalize: Abstract method for normalizing a given tensor.
        setup_normalization: Abstract method for setting up the normalization process.
    """
    @abstractmethod
    def __call__(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for normalizing a given tensor.

        Implement this method in subclasses to normalize the provided tensor.

        :param attributes: The tensor to be normalized.
        :type attributes: torch.Tensor
        :return: The normalized tensor.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def setup_normalization(self) -> None:
        """
        Abstract method for setting up the normalization process.

        Implement this method in subclasses to initialize any required resources or parameters for normalization.

        :return: None
        """
        raise NotImplementedError