from dataclasses import dataclass
import logging
from typing import Tuple

from omegaconf import DictConfig
import torch

from dMC.data.datasets import Dataset, Hydrofabric
from dMC.data.observations import Observations

log = logging.getLogger(__name__)


@dataclass
class _Dataset(torch.utils.data.Dataset):
    raw_data: Hydrofabric
    observations: torch.Tensor

    def __len__(self):
        """Method from the torch.Dataset parent class"""
        return self.observations.shape[0]

    def __getitem__(self, idx):
        """Method from the torch.Dataset parent class"""
        return self.raw_data, self.observations[idx]


@dataclass
class DataLoader:
    data: Dataset
    observations: Observations

    @staticmethod
    def collate_fn(batch) -> Tuple[torch.Tensor, ...]:
        """
        A custom function to make sure the dataloader is getting the right dimensions
        :param batch:
        :return:
        """
        hydrofabric = batch[0][0]
        observations = torch.stack([item[1] for item in batch]).transpose(0, 1)
        return (
            hydrofabric,
            observations
        )

    def __call__(self, cfg: DictConfig) -> torch.utils.data.DataLoader:
        hydrofabric = self.data.get_data()
        observations = self.observations.get_data()
        ds = _Dataset(hydrofabric, observations)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.time.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )


