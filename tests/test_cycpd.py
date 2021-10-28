import pytest
import numpy as np
import cycpd
import matplotlib.pyplot as plt
from ai_ct_scans import point_matching

plt.ion()


def test_cycpd_deformable_translation_exists():
    assert hasattr(cycpd, "deformable_registration")


@pytest.fixture()
def five_points_3d():
    np.random.seed(555)
    return np.random.rand(5, 3)


@pytest.fixture()
def five_points_3d_warped(five_points_3d):
    return five_points_3d + np.random.rand(5, 3) * 0.1


# Test not currently active
# def test_five_points_3d_matched(five_points_3d, five_points_3d_warped):
#     reg = cycpd.deformable_registration(**{'X': five_points_3d,
#                                            'Y': five_points_3d_warped,
#                                            'max_iterations': 100,
#                                            'alpha': 0.1,
#                                            'beta': 3
#                                            })
#     recovered_points, _ = reg.register()
#     prev_dif = np.abs(five_points_3d - five_points_3d_warped)
#     new_dif = np.abs(five_points_3d - recovered_points)
#     assert (new_dif < prev_dif).all()


@pytest.fixture()
def five_points_3d_warped_shuffled(five_points_3d_warped):
    return np.random.permutation(five_points_3d_warped)


def test_five_points_3d_shuffled_matched(
    five_points_3d, five_points_3d_warped_shuffled, five_points_3d_warped
):
    reg = cycpd.deformable_registration(
        **{
            "X": five_points_3d,
            "Y": five_points_3d_warped_shuffled,
            "max_iterations": 100,
            "alpha": 0.1,
            "beta": 3,
        }
    )
    recovered_points, _ = reg.register()
    only_warped_match_indices = point_matching.match_indices(
        five_points_3d, five_points_3d_warped
    )
    total_error_only_warped = np.linalg.norm(
        (
            five_points_3d[only_warped_match_indices[1]]
            - five_points_3d_warped[only_warped_match_indices[0]]
        ),
        axis=1,
    ).sum()
    matched_inds = point_matching.match_indices(five_points_3d, recovered_points)
    total_error_permuted_and_recovered = np.linalg.norm(
        (five_points_3d[matched_inds[1]] - recovered_points[matched_inds[0]]), axis=1
    ).sum()
    assert total_error_permuted_and_recovered < total_error_only_warped
