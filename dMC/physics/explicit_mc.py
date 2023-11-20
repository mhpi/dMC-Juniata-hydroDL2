import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class ExplicitMC(nn.Module):
    def __init__(self, cfg: DictConfig, neural_network: nn.Module):
        super(ExplicitMC, self).__init__()
        self.cfg = cfg
        self.epoch = 0
        self.neural_network = neural_network
        self.p_spatial = torch.tensor(self.cfg.variables.p)
        self.t = torch.tensor(self.cfg.variables.t)
        self.x_storage = torch.tensor(self.cfg.variables.x)
        self.length = None
        self.slope = None
        self.velocity = None
        self._discharge_t = None
        self._discharge_t1 = None
        self._downstream_indexes = None
        self._q_t = None

    @property
    def n(self) -> torch.Tensor:
        return self.neural_network.n

    @property
    def q_spatial(self) -> torch.Tensor:
        return self.neural_network.q

    def get_length(self, attributes: torch.Tensor) -> torch.Tensor:
        length_idx = self.cfg.length.idx
        return attributes[:, length_idx].clone()

    def get_slope(self, attributes: torch.Tensor) -> torch.Tensor:
        slope_idx = self.cfg.slope.idx
        s0_ = attributes[:, slope_idx].clone()
        return torch.clamp(s0_, min=self.cfg.slope.min, max=self.cfg.slope.max)

    @staticmethod
    # @torch.jit.script
    def _get_velocity(q_t, _n, _q_spatial, _s0, p_spatial) -> torch.Tensor:
        """
        Since this is an explicit solver, we need to index the values that we're calculating
        :param downstream_indexes: Reach indexes
        """
        numerator = q_t * _n * (_q_spatial + 1)
        denominator = p_spatial * torch.pow(_s0, 0.5)
        depth = torch.pow(
            torch.div(numerator, denominator), torch.div(3.0, 5.0 + 3.0 * _q_spatial)
        )
        v = torch.div(1, _n) * torch.pow(depth, (2 / 3)) * torch.pow(_s0, (1 / 2))
        c_ = torch.clamp(v, 0.3, 15)
        c = c_ * 5 / 3
        return c

    def muskingum_cunge(
        self, i_t: torch.Tensor, i_t1: torch.Tensor, q_prime_segment: torch.Tensor,
    ) -> torch.Tensor:
        q_t = self._discharge_t[self._downstream_indexes]
        length = self.length[self._downstream_indexes]
        if self.n.dim() == 0:
            _n = self.n
        else:
            _n = self.n[self._downstream_indexes]
        if self.q_spatial.dim() == 0:
            _q_spatial = self.q_spatial
        else:
            _q_spatial = self.q_spatial[self._downstream_indexes]
        _s0 = self.slope[self._downstream_indexes]
        velocity = self._get_velocity(q_t, _n, _q_spatial, _s0, self.p_spatial)
        return self._muskingum_cunge(
            i_t, i_t1, q_t, q_prime_segment, length, velocity, self.x_storage, self.t
        )

    @staticmethod
    @torch.jit.script
    def _muskingum_cunge(
        i_t: torch.Tensor,
        i_t1: torch.Tensor,
        q_t: torch.Tensor,
        q_prime_segment: torch.Tensor,
        length: torch.Tensor,
        velocity: torch.Tensor,
        x_storage: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        An explicit Muskingum Cunge Solution
        :param downstream_indexes: Reach indexes
        :param i: timestep
        :param i_t: inflow at time t
        :param i_t1: discharge at time t
        :param q_prime: lateral inflow at reaches
        :return:
        """
        k = length / velocity
        denom = (2.0 * k * (1.0 - x_storage)) + t
        c_1 = (t - (2.0 * k * x_storage)) / denom
        c_2 = (t + (2.0 * k * x_storage)) / denom
        c_3 = (2.0 * k * (1.0 - x_storage) - t) / denom
        c_4 = (2.0 * t) / denom
        q_t1 = (c_1 * i_t1) + (c_2 * i_t) + (c_3 * q_t) + (c_4 * q_prime_segment)
        return torch.clamp(q_t1, min=1e-4)

    def read_inflow(self, upstream_indexes):
        return self._read_inflow(
            self._discharge_t, self._discharge_t1, upstream_indexes
        )

    @staticmethod
    @torch.jit.script
    def _read_inflow(discharge_t, discharge_t1, upstream_indexes):
        single_index_mask = (upstream_indexes != -1).sum(dim=1) == 1
        masked_indexes = upstream_indexes[single_index_mask]
        single_indices = masked_indexes[masked_indexes != -1]

        many_index_mask = (upstream_indexes != -1).sum(dim=1) > 1
        i_t = torch.zeros(len(upstream_indexes), dtype=torch.float64)
        i_t1 = torch.zeros(len(upstream_indexes), dtype=torch.float64)
        i_t[single_index_mask] = discharge_t[single_indices]
        i_t1[single_index_mask] = discharge_t1[single_indices]
        if many_index_mask.sum() > 0:
            masked_indexes = upstream_indexes[many_index_mask]
            actual_indices = torch.arange(len(many_index_mask))[many_index_mask]
            for i, row in enumerate(masked_indexes):
                valid_indices = row[row != -1]
                idx = actual_indices[i]
                i_t[idx] = discharge_t[valid_indices].sum()
                i_t1[idx] = discharge_t1[valid_indices].sum()
        return i_t, i_t1

    def forward(self, inputs):
        hydrofabric, observations = inputs
        attributes = hydrofabric.attributes
        q_prime = hydrofabric.forcings
        gage_indices = hydrofabric.network.gage_indices
        A = hydrofabric.network.explicit_network_matrix
        river_index_graph = torch.tensor(hydrofabric.network.index_graph)
        output = torch.zeros(
            [observations.shape[0], q_prime.shape[0]], dtype=torch.float64,
        )
        self._discharge_t = q_prime[0]
        self._discharge_t1 = torch.zeros(q_prime[0].shape, dtype=torch.float64)
        for i in range(len(gage_indices)):
            output[i, 0] = torch.sum(self._discharge_t[gage_indices[i]])
        self.length = self.get_length(attributes)
        self.slope = self.get_slope(attributes)
        for timestep in tqdm(
            range(1, len(q_prime)),
            desc=f"Epoch {self.epoch}: Explicit Muskingum Cunge Routing",
        ):
            for j in range(river_index_graph.shape[0]):
                self._downstream_indexes = river_index_graph[j][
                    river_index_graph[j] != -1
                ]
                if j == 0:
                    """No upstream nodes"""
                    i_t = torch.tensor(0, dtype=torch.float64)
                    i_t1 = torch.tensor(0, dtype=torch.float64)
                else:
                    upstream_indexes = A[self._downstream_indexes]
                    mask = upstream_indexes[:, 0] != -1
                    upstream_indexes_filtered = upstream_indexes[mask]
                    i_t, i_t1 = self.read_inflow(upstream_indexes_filtered)
                q_prime_segment = q_prime[timestep, self._downstream_indexes]
                q_t1 = self.muskingum_cunge(i_t, i_t1, q_prime_segment)
                self._discharge_t1[self._downstream_indexes] = q_t1
            for i in range(len(gage_indices)):
                output[i, timestep] = torch.sum(self._discharge_t1[gage_indices[i]])
            self._discharge_t = self._discharge_t1.clone()
        return output
