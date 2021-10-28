from scipy.optimize import linear_sum_assignment

import numpy as np


def abs_dist_cost_matrix(points_1, points_2):
    """Generates the cost matrix suitable for scipy.optimize.linear_sum_assignment between two sets of points

    Args:
        points_1 (np.ndarray): A set of points with dimensions (point index, number of spatial dimensions)
        points_2 (np.ndarray): A set of points with dimensions (point index, number of spatial dimensions)

    Returns:
        (np.ndarray of shape (points_1.shape[0], points_2.shape[0])): The Euclidean distances between
        pairs of points in points_1 and points_2 such that the distance from pair points_1[i] points_2[j] occurs at
        position (j, i) in output

    """
    vect_dists = points_1 - points_2.reshape([points_2.shape[0], 1, points_2.shape[1]])
    return np.linalg.norm(vect_dists, axis=-1)


def match_indices(points_0, points_1):
    """Gets the zeroth-dimension indices in points_1 that match zeroth-dimension elements of points_2 such that the linear
    sum assignment of Euclidean distance costs are minimised

    Args:
        points_0 (np.ndarray): np.ndarray
        points_1 (np.ndarray): np.ndarray

    Returns:
        (tuple of two lists of ints): Each list will contain the integer indicies such that matching, ordered point
        sets are recovered via points_0[return_val[1]] closest matches are points_1[return_val[0]]

    """
    cost_mat = abs_dist_cost_matrix(points_0, points_1)
    return linear_sum_assignment(cost_mat)
