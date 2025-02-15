import numpy as np

from ..variable import Variable

from .. import var_utils

class TestVarUtils:
    def test_sum_broadcast_dim1(self):
        a = np.array([[1, 2, 3], [1, 2, 3]])
        b = np.array([1, -2, 3])

        d = var_utils.sum_broadcast_dim(a, b)
        assert np.all(d.shape == b.shape)
        assert np.all(d == np.array([2, 4, 6]))

    def test_sum_broadcast_dim2(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array(1)

        d = var_utils.sum_broadcast_dim(a, b)
        assert np.all(d.shape == b.shape)
        assert np.all(d == np.array(21))

    def test_sum_broadcast_dim3(self):
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[1], [2]])

        d = var_utils.sum_broadcast_dim(a, b)
        assert np.all(d.shape == b.shape)
        assert np.all(d == np.array([[10], [26]]))

    def test_sum_broadcast_dim4(self):
        a = np.array([1, 2, 3])
        b = np.array([[1], [2], [3]])

        d = var_utils.sum_broadcast_dim(a, b)
        assert np.all(d.shape == b.shape)
        assert np.all(d == np.array([[1], [2], [3]]))

    def test_sum_broadcast_dim_invalid(self):
        a = np.array([1, 2, 3])
        b = np.array([[1, 2], [3, 4]])

        try:
            var_utils.sum_broadcast_dim(a, b)
        except ValueError as e:
            assert (
                str(e)
                == "Target shape (2, 2) cannot be broadcast to source shape (3,)"
            )
