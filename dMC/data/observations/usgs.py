import logging
from pathlib import Path
import re
from typing import List

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch

from dMC.data.dates import Dates
from dMC.data.normalize import Normalize

log = logging.getLogger(__name__)


class USGS:
    def __init__(self, cfg: DictConfig, dates: Dates, normalize: Normalize):
        self.cfg = cfg
        self.dates = dates
        self.normalize = normalize
        self.observations = None
        self._read_observations()

    def _read_observations(self) -> None:
        output = None
        matching_files = self._find_file()
        if matching_files is False:
            return output
        matching_files = sorted(matching_files, key=lambda p: p.name)
        output = np.zeros([len(matching_files), len(self.dates.time_interval)])
        for i, file_path in enumerate(matching_files):
            df = pd.read_csv(file_path)
            date_column = None
            for potential_date_column in ["Date", "dates"]:
                if potential_date_column in df.columns:
                    date_column = potential_date_column
                    break

            if date_column is not None:
                df[date_column] = pd.to_datetime(df[date_column])
                start_time = pd.to_datetime(self.dates.time_interval[0])
                end_time = pd.to_datetime(self.dates.time_interval[-1])
                mask = (df[date_column] >= start_time) & (df[date_column] <= end_time)
                df_subset = df.loc[mask]
                output[i] = (
                    df_subset["v"].to_list()
                    if "v" in df.columns
                    else df_subset["V"].to_list()
                )
            else:
                log.error("No valid date column found for USGS data.")
        self.observations = torch.tensor(output.transpose(1, 0))

    def _find_file(self) -> List[Path]:
        usgs_dir = Path(self.cfg.observations.dir)
        log.info(f"loss gages: {self.cfg.observations.loss_nodes}")
        river_code_regex = [
            re.compile(r"{}-".format(node)) for node in self.cfg.observations.loss_nodes
        ]
        matching_files = []
        for file_path in usgs_dir.glob("*"):
            for regex in river_code_regex:
                if regex.search(str(file_path)):
                    matching_files.append(file_path)
                    break
        if not matching_files:
            print("No matching files found.")
            return False
        return matching_files

    def get_data(self) -> torch.Tensor:
        return self.observations
