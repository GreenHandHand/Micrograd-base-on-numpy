import numpy as np

from ..variable import Variable

from .. import var_utils


class TestVarUtils:
    def test_sum_broadcast_dim1(self):
        a = np.array([[1, 2, 3], [1, 2, 3]])
        b = np.array([1, -2, 3])

        d = var_utils.sum_broadcast_dim(a, b)
        assert np.all(d.shape == b.shape)

    def test_sum_broadcast_dim2(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array(1)

        d = var_utils.sum_broadcast_dim(a, b)
        assert np.all(d.shape == b.shape)
