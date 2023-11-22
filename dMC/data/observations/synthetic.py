import logging
from pathlib import Path

from omegaconf import DictConfig
import pandas as pd
import torch

from dMC.data.dates import Dates
from dMC.data.normalize import Normalize

log = logging.getLogger(__name__)


class Synthetic:
    def __init__(self, cfg: DictConfig, dates: Dates, normalize: Normalize):
        self.cfg = cfg
        self.dates = dates
        self.observations = None
        self._read_observations()

    def _read_observations(self) -> None:
        synthetic_path = Path(self.cfg.observations.dir)
        loss_gages = [str(x) for x in self.cfg.observations.loss_nodes]
        log.info(f"loss gages: {loss_gages}")
        log.info(f"observations file: {self.cfg.observations.file_name}")
        df = pd.read_csv(
            synthetic_path / self.cfg.observations.file_name, usecols=loss_gages
        )
        self.observations = torch.tensor(df.to_numpy())

    def get_data(self) -> torch.Tensor:
        return self.observations
