import logging

from omegaconf import DictConfig
from sklearn import preprocessing
import torch

log = logging.getLogger(__name__)


class MinMax:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.scalar = None
        self.setup_normalization()

    def __call__(self, attributes: torch.Tensor) -> torch.Tensor:
        """
        Creates the normalized attributes from a Min/Max Scaler
        """
        x_trans = attributes.transpose(1, 0)
        x_tensor = torch.zeros(x_trans.shape)
        for i in range(0, x_trans.shape[0]):
            x_tensor[i, :] = torch.tensor(
                self.scalar.fit_transform(x_trans[i, :].reshape(-1, 1).cpu()).transpose(
                    1, 0
                )
            )
        return torch.transpose(x_tensor, 1, 0)

    def setup_normalization(self) -> None:
        self.scalar = preprocessing.MinMaxScaler()
