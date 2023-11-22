from datetime import datetime
import logging
import time

import numpy as np
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError
import pandas as pd

log = logging.getLogger(__name__)


class Dates:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.time_interval = None
        self._large_interval = None
        self.read_dates()

    def apply_tau(self) -> np.ndarray:
        tau = self.cfg.time.tau
        time_steps = self.cfg.time.steps
        return self._large_interval[: time_steps + tau].to_numpy()

    def read_dates(self) -> None:
        """
        Reading the dates that we need to route streamflow for
        :return:
        """
        start_datetime = datetime.strptime(
            self.cfg.time.start, "%m/%d/%Y %H:%M:%S"
        )
        end_datetime = datetime.strptime(self.cfg.time.end, "%m/%d/%Y %H:%M:%S")
        self._large_interval = pd.date_range(
            start_datetime, end_datetime, freq="H"
        ).strftime("%m/%d/%Y %H:%M:%S")
        try:
            time_steps = self.cfg.time.steps
            self.time_interval = self._large_interval[:time_steps].to_numpy()
        except ConfigAttributeError:
            self.time_interval = self._large_interval.to_numpy()

    def unix_timestamp(self, dt_str) -> int:
        dt_obj = datetime.strptime(dt_str, "%m/%d/%Y %H:%M:%S")
        unix_timestamp_ = int(time.mktime(dt_obj.timetuple()))
        return unix_timestamp_
