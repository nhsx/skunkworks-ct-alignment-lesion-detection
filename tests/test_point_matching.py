import pytest
import numpy as np
from ai_ct_scans import point_matching


@pytest.fixture()
def dummy_point_sets():
    dummy = np.ones([3, 3]) * np.array([1, 2, 3])
    dummy[1, :] += 1
    dummy[2, :] += 2
    dummy_2 = dummy.copy()
    dummy_2[0, 0] += 0.1
    dummy_2[1, 1] += 0.2
    dummy_2[2, 2] += 0.3
    return dummy, dummy_2


def test_abs_dist_cost_matrix_gets_expected(dummy_point_sets):
    dummy, dummy_2 = dummy_point_sets
    cost_mat = point_matching.abs_dist_cost_matrix(dummy, dummy_2)
    np.testing.assert_array_almost_equal(
        cost_mat[[0, 1, 2], [0, 1, 2]], np.array([0.1, 0.2, 0.3])
    )
    cost_mat_row_1_col_0_expected = np.linalg.norm(dummy[0] - dummy_2[1])
    assert np.abs(cost_mat_row_1_col_0_expected - cost_mat[1, 0]) < 1e-6


def test_abs_dist_cost_matrix_gets_expected_after_permutation(dummy_point_sets):
    dummy, dummy_2 = dummy_point_sets
    expected = [1, 2, 0]
    permuted_dummy_2 = dummy_2[expected]
    matched_indices = point_matching.match_indices(dummy, permuted_dummy_2)
    np.testing.assert_array_equal(matched_indices[0], [0, 1, 2])
    np.testing.assert_array_equal(matched_indices[1], expected)


@pytest.fixture()
def dummy_with_noise_point(dummy_point_sets):
    _, dummy_2 = dummy_point_sets
    dummy_with_noise_row = np.zeros([4, 3])
    dummy_with_noise_row[:2] = dummy_2[:2]
    dummy_with_noise_row[3] = dummy_2[2]
    return dummy_with_noise_row


def test_match_inds_handles_dissimilar_point_numbers(
    dummy_point_sets, dummy_with_noise_point
):
    dummy, _ = dummy_point_sets

    matched_inds = point_matching.match_indices(dummy, dummy_with_noise_point)
    np.testing.assert_array_equal(matched_inds[0], [0, 1, 3])
    np.testing.assert_array_equal(matched_inds[1], [0, 1, 2])


def test_match_inds_handles_dissimilar_and_permuted_point_numbers(
    dummy_point_sets, dummy_with_noise_point
):
    dummy, _ = dummy_point_sets
    dummy_w_noise_permuted = dummy_with_noise_point[[1, 0, 2, 3]]
    matched_indices = point_matching.match_indices(dummy, dummy_w_noise_permuted)
    np.testing.assert_array_equal(matched_indices[0], [0, 1, 3])
    np.testing.assert_array_equal(matched_indices[1], [1, 0, 2])
