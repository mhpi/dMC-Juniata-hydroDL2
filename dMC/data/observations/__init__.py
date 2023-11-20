from abc import ABC, abstractmethod
import logging

import torch


log = logging.getLogger(__name__)


class Observations(ABC):
    """
    Abstract base class for handling observations.

    Methods:
        _read_observations: Abstract method for reading observations based on given dates.
        get: Abstract method for retrieving a tensor of observations based on given dates.
    """

    @abstractmethod
    def _read_observations(self) -> None:
        """
        A method for reading observations based on given dates.

        Implement this method in subclasses to read observations for the provided dates.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> torch.Tensor:
        """
        A method for retrieving a tensor of observations based on given dates, data data, and normalization.

        Implement this method in subclasses to return a tensor of observations for the provided dates and data data,
        using the provided Normalize object for any required normalization tasks.
        :return: A tensor containing the observations.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
