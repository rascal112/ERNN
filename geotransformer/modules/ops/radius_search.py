import importlib
from IPython import embed
from torch_geometric.nn import radius_graph
import torch

ext_module = importlib.import_module('geotransformer.ext')


def radius_search_graph(q_points, q_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    # neighbor_indices = ext_module.radius_neighbors(q_points, q_points, q_lengths, q_lengths, radius)
    # if neighbor_limit > 0:
    #     neighbor_indices = neighbor_indices[:, :neighbor_limit]
    source = q_points[:q_lengths[0]]
    target = q_points[q_lengths[0]:]
    edge_index_source = radius_graph(source, r=radius, max_num_neighbors = neighbor_limit)
    edge_index_target = radius_graph(target, r=radius, max_num_neighbors = neighbor_limit) + q_lengths[0]
    
    edge_index = torch.cat([edge_index_source, edge_index_target], dim=1)
    # edge_index = edge_index.transpose(0, 1)
    return edge_index

def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    neighbor_indices = ext_module.radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius)
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    return neighbor_indices