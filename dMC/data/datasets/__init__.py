from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Union

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class Network:
    gage_indices: List[torch.Tensor]
    explicit_network_matrix: torch.Tensor
    index_graph: np.ndarray


@dataclass
class Hydrofabric:
    attributes: Union[torch.Tensor, None] = field(default=None)
    forcings: Union[torch.Tensor, None] = field(default=None)
    network: Union[Network, None] = field(default=None)
    normalized_attributes: Union[torch.Tensor, None] = field(default=None)
    normalized_forcings: Union[torch.Tensor, None] = field(default=None)


class Dataset(ABC):
    """
    Abstract base class for handling data data and network construction.

    Methods:
        _create_graph_data: Abstract method for creating graph data.
        _create_network: Abstract method for creating the data network.
        _read_attributes: Abstract method for reading attributes related to the data.
        get: Abstract method for retrieving data data as a dictionary.
    """

    @abstractmethod
    def _read_attributes(self) -> None:
        """
        Abstract method for reading attributes related to the data.

        Implement this method in subclasses to read attributes needed for the data.

        :return: None
        """

    @abstractmethod
    def _read_forcings(self) -> None:
        """
        Abstract method for reading attributes related to the data.

        Implement this method in subclasses to read attributes needed for the data.

        :return: None
        """

    @abstractmethod
    def _read_data(self) -> None:
        """The method to read all data"""

    @abstractmethod
    def get_data(self) -> Hydrofabric:
        """
        Abstract method for retrieving data data as a dictionary.

        Implement this method in subclasses to return data data,
        potentially using the provided Normalize object for any required normalization tasks.

        :param normalize: An instance of the Normalize class for performing normalization.
        :type normalize: Normalize
        :return: A dictionary containing data data.
        :rtype: Dict
        """
