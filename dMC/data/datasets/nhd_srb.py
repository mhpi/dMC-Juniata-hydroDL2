import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from dMC.data.datasets import Hydrofabric, Network, Dataset
from dMC.data.dates import Dates
from dMC.data.normalize import Normalize

log = logging.getLogger(__name__)


class NHDSRB(Dataset):
    """
    A data for the Bindas et al. 2023 Muskingum Cunge Routing Paper

    Args:
    cfg (DictConfig): Configuration object containing all the necessary parameters.
    """

    def __init__(self, cfg: DictConfig, dates: Dates, normalize: Normalize):
        self.cfg = cfg
        self.dates = dates
        self.device = torch.device(self.cfg.device)
        self.end_node = self.cfg.end_node
        self.normalize = normalize
        self.attributes = None
        self.edges = None
        self.forcings = None
        self.network = None
        self.normalized_attributes = None
        self._nodes = None
        self._areas = None
        self._read_data()

    def _create_graph_data(self) -> None:
        """
        A function to create the edges and nodes within the dataset
        :return: None
        """
        NODE_ID = 0
        FROM_EDGE = 2
        FROM_NODE = 2
        TO_NODE = 3

        def _generate_sub_basin(node, edges_arr, nodes_arr):
            """
            A reverse tree-traversal algorithm to find all upstream connections of a given node
            :param node: the node we're on
            :param nodes_arr: all nodes upstream
            :param edges_arr: all edges upstream
            :param basin_edges_: the upstream basin's total edges
            :param basin_nodes_: the upstream basin's total nodes
            :return:
            """
            node = int(node)
            index = np.where(nodes_arr[:, NODE_ID] == node)[0]
            if index.size != 0:
                node_row = nodes_arr[index, :][0]
                upstream_edge_index = np.where(edges_arr[:, TO_NODE] == node)[0]
                if upstream_edge_index.size != 0:
                    for edge_id in upstream_edge_index:
                        edge_row = edges_arr[edge_id, :]
                        basin_edges.append(edge_row)
                        from_edge = node_row[FROM_EDGE]
                        # if from_edge != 0:
                        basin_nodes.append(node_row)
                        next_node = edge_row[FROM_NODE]
                        _generate_sub_basin(next_node, edges_arr, nodes_arr)
                else:
                    basin_nodes.append(node_row)

        def _process_df(basin_array, save_path):
            """
            a function to take your array of basin data and turn it
            into a format to save to disk
            :param basin_array: the basin data
            :param save_path: save path
            :return:
            """
            stack_ = np.stack(basin_array)
            df_ = pd.DataFrame(stack_)
            df_ = df_.sort_values(by=[0])
            df = df_.drop_duplicates()
            df.to_csv(save_path, index=False)
            return df

        try:
            self.edges = pd.read_csv(self.cfg.save_paths.edges).to_numpy()
            self._nodes = pd.read_csv(self.cfg.save_paths.nodes).to_numpy()
            self.areas = np.load(self.cfg.save_paths.areas)
        except FileNotFoundError:
            full_graph_edge = pd.read_csv(self.cfg.csv.edges)
            full_edges_numpy = full_graph_edge.to_numpy()
            full_graph_nodes = pd.read_csv(self.cfg.csv.nodes).to_numpy()
            basin_edges = []
            basin_nodes = []
            _generate_sub_basin(
                self.end_node,
                full_edges_numpy,
                full_graph_nodes,
            )
            # Adding extra edge for network
            _index = np.where(full_graph_nodes[:, 0] == self.end_node)[0]
            node_row = full_graph_nodes[_index, :][0]
            to_edge = node_row[3]  # MAGIC NUM: 3 is the toEdge index
            _index = np.where(full_graph_edge.iloc[:, 0] == to_edge)[0]
            edge_row = full_graph_edge.iloc[_index, :].to_numpy()[0]
            basin_edges.append(edge_row)

            df_edges = _process_df(basin_edges, self.cfg.save_paths.edges)
            df_edges.columns = full_graph_edge.columns
            self.edges = df_edges.to_numpy()
            self.areas = df_edges["TotDASqKM"].to_numpy().astype("float")
            np.save(self.cfg.save_paths.areas, self.areas)
            df_nodes = _process_df(basin_nodes, self.cfg.save_paths.nodes)
            self._nodes = df_nodes.to_numpy()

    def _create_network(self) -> None:
        node_ids = self._nodes[:, 0]
        graph_map = {id: i for i, id in enumerate(node_ids)}

        def _create_network_matrix(upstream_dictionary) -> torch.Tensor:
            network_path = Path(self.cfg.save_paths.network)
            if network_path.is_file():
                network_df = pd.read_csv(network_path).set_index(node_ids)
                network_df = network_df.drop(["Unnamed: 0"], axis=1)
                network_matrix = torch.tensor(
                    network_df.values, device=self.device, dtype=torch.float64
                )
            else:
                network_matrix = torch.zeros(
                    [len(node_ids), len(node_ids)], device=self.device
                )
                for key in upstream_dictionary:
                    try:
                        row = graph_map[key]
                        upstream_idx = upstream_dictionary[key]
                        network_matrix[row, upstream_idx] = 1
                    except KeyError:
                        log.debug(f"KeyError at key: {key}. Find a solution later but ignore")
                network_df = pd.DataFrame(
                    network_matrix.cpu().numpy(),
                    index=node_ids,
                    columns=node_ids,
                )
                network_df.to_csv(self.cfg.save_paths.network)
            return network_matrix

        def _prepare_river_creation(upstream_dictionary):
            graph_nodes = torch.from_numpy(
                np.fromiter(upstream_dictionary.keys(), dtype=int)
            )
            upstream_indexes = node_connections[:, 0].clone()
            combined = torch.cat((graph_nodes, upstream_indexes))
            uniques, counts = combined.unique(return_counts=True)
            upstream_edges = uniques[counts == 1].numpy()
            upstream_edges = upstream_edges[upstream_edges < self.end_node]
            return upstream_edges, node_connections.numpy()

        def _create_rivers(upstream_edges, node_connection_numpy, upstream_dictionary):
            edge_dict = {k: [] for k in upstream_edges}
            index_dict = {i: [] for i in range(len(edge_dict))}
            starting_indexes = [
                np.where(node_connection_numpy[:, 0] == edge)[0][0]
                for edge in upstream_edges
            ]
            indexes = starting_indexes
            downstream_edges = node_connection_numpy[indexes, 1]
            while sum(indexes) != (len(edge_dict) * -1):
                new_indexes = []
                new_edges = []
                for i in range(len(downstream_edges)):
                    current_index = indexes[i]
                    if current_index != -1:
                        river_index = i
                        river_edge_index = upstream_edges[i]
                        downstream_edge = downstream_edges[i]
                        edge_dict[river_edge_index].append(downstream_edge)
                        index_dict[river_index].append(current_index)
                        if len(upstream_dictionary[downstream_edge]) > 1:
                            upstream_dictionary[downstream_edge].remove(current_index)
                            new_indexes.append(-1)
                            new_edges.append(-1)
                        elif downstream_edge == self.end_node:
                            new_indexes.append(-1)
                            new_edges.append(-1)
                        else:
                            new_index = np.where(
                                node_connection_numpy[:, 0] == downstream_edge
                            )[0][0]
                            new_indexes.append(new_index)
                            new_edges.append(node_connection_numpy[new_index, 1])
                    else:
                        new_indexes.append(-1)
                        new_edges.append(-1)
                downstream_edges = new_edges
                indexes = new_indexes
            return edge_dict, index_dict

        def generate_explicit_network(node_connections, upstream_dictionary):
            max_length = max(len(v) for v in upstream_dictionary.values())
            A = torch.full((node_connections.shape[0], max_length), -1)
            for i, from_node in enumerate(node_connections[:, 0]):
                try:
                    values = upstream_dictionary[from_node.item()]
                    A[i][0 : len(values)] = torch.tensor(values)
                except KeyError:
                    pass
            return A

        node_connections = torch.tensor(
            self.edges[:, 2:4].astype("float64"), dtype=torch.long
        )
        upstream_dictionary = {}
        for i in range(len(self.edges[:, 0])):
            to_edge = node_connections[i, 1].item()
            try:
                upstream_dictionary[to_edge].append(i)
            except KeyError:
                upstream_dictionary[to_edge] = [i]
        upstream_dictionary_copy = copy.deepcopy(upstream_dictionary)
        upstream_edges, node_connection_numpy = _prepare_river_creation(
            upstream_dictionary_copy
        )
        edge_dict, index_dict = _create_rivers(
            upstream_edges, node_connection_numpy, upstream_dictionary_copy
        )

        longest_river = max(len(item) for item in edge_dict.values())
        node_graph = np.zeros([longest_river, len(edge_dict)], dtype=int)
        index_graph = np.zeros([longest_river, len(edge_dict)], dtype=int)
        node_graph[:] = -1
        index_graph[:] = -1
        for i in range(len(edge_dict)):
            river = edge_dict[upstream_edges[i]]
            river_indexes = index_dict[i]
            node_graph[: len(river), i] = river
            index_graph[: len(river_indexes), i] = river_indexes

        loss_nodes = self.cfg.observations.loss_nodes
        gage_inidices = [
            torch.nonzero(node_connections[:, 1] == loss_node, as_tuple=True)[0]
            for loss_node in loss_nodes
        ]
        explicit_network_matrix = generate_explicit_network(
            node_connections, upstream_dictionary
        )
        self.network = Network(
            gage_indices=gage_inidices,
            explicit_network_matrix=explicit_network_matrix.to(self.cfg.device),
            index_graph=index_graph,
        )

    def _read_attributes(self) -> None:
        self.attributes = torch.tensor(
            self.edges[:, 4:].astype("float64"), dtype=torch.float64
        )

    def _read_forcings(self) -> None:
        EDGE_ID = 0
        DATE_COLUMN = "dates"
        ALTERNATE_DATE_COLUMN = "Date"
        DATE_FORMAT = "%m/%d/%Y %H:%M"

        def _create_q_prime(_forcings: pd.DataFrame) -> np.ndarray:
            """
            Creating the q_prime data in the river network space
            :param forcings: The forcing data in HUC10 space
            :return: Q_prime data in the river network space
            """
            TM = pd.read_csv(self.cfg.csv.mass_transfer_matrix)
            edge_ids = self.edges[:, EDGE_ID]
            date_column = (
                DATE_COLUMN if DATE_COLUMN in _forcings else ALTERNATE_DATE_COLUMN
            )
            forcings_flow_subset = torch.tensor(
                _forcings.drop([date_column], axis=1).values, dtype=torch.float64
            )
            tm = torch.tensor(
                TM.iloc[edge_ids].values.T, dtype=torch.float64
            )  # Transpose directly
            q_prime_network = forcings_flow_subset @ tm
            q_prime = q_prime_network[self.cfg.time.tau :]
            return q_prime

        def _preprocess_forcings() -> pd.DataFrame:
            """
            a function to read in forcings,
            then sort based oon the time interval we're looking at
            :return: the forcings from our defined time interval
            """
            df = pd.read_csv(self.cfg.csv.q_prime)
            date_column = DATE_COLUMN if DATE_COLUMN in df else ALTERNATE_DATE_COLUMN
            if date_column == ALTERNATE_DATE_COLUMN:
                log.debug("KEY ERROR: using secondary key")
                df = df.drop(["Date Code"], axis=1)
            df[date_column] = pd.to_datetime(df[date_column], format=DATE_FORMAT)
            time_q_prime = self.dates.apply_tau()
            start_time = time_q_prime[0]
            end_time = time_q_prime[-1]
            mask = (df[date_column] >= start_time) & (df[date_column] <= end_time)
            df_subset = df.loc[mask]
            shift_amount = 1 if date_column == DATE_COLUMN else 2
            columns_ordered = np.roll(df_subset.columns.sort_values(), shift_amount)
            forcings_scaled = df_subset.reindex(columns=columns_ordered)
            return forcings_scaled

        start_date_str = self.dates.time_interval[0]
        end_date_str = self.dates.time_interval[-1]
        save_path = self.cfg.save_paths.q_prime.format(
            self.dates.unix_timestamp(start_date_str),
            self.dates.unix_timestamp(end_date_str),
        )
        log.info(f"Tau: {self.cfg.time.tau}")
        try:
            q_prime = pd.read_csv(save_path).to_numpy()
            self.forcings = torch.tensor(q_prime)
        except FileNotFoundError:
            _forcings = _preprocess_forcings()
            q_prime = _create_q_prime(_forcings)
            df = pd.DataFrame(q_prime)
            df.to_csv(save_path, index=False)
            self.forcings = q_prime

    def _read_data(self) -> None:
        self._create_graph_data()
        self._create_network()
        self._read_attributes()
        self._read_forcings()
        self.normalized_attributes = self.normalize(self.attributes)

    def get_data(self) -> Hydrofabric:
        return Hydrofabric(
            attributes=self.attributes.to(self.cfg.device),
            forcings=self.forcings.to(self.cfg.device),
            network=self.network,
            normalized_attributes=self.normalized_attributes.to(self.cfg.device),
        )
