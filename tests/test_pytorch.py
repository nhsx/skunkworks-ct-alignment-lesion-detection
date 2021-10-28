import torch


def test_shape_of_simple_tensor():
    test_tensor = torch.rand(
        (
            2,
            5,
        )
    )
    assert test_tensor.shape == torch.Size([2, 5])
