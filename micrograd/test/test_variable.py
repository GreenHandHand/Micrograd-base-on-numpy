from ..variable import Variable

import pytest
import torch
import numpy as np


class TestVariable:
    @staticmethod
    def generate_compare_data(
        shape: tuple[int, ...] | list[tuple[int, ...]], cnt: int = 2
    ) -> list[tuple[Variable, torch.Tensor]]:
        if isinstance(shape, tuple):
            data = []
            for _ in range(cnt):
                x = np.random.uniform(-1, 1, shape)
                vx = Variable(x, _require_grad=True, dtype=np.float64)
                tx = torch.tensor(x, requires_grad=True, dtype=torch.float64)
                data.append((vx, tx))
            return data
        else:
            assert len(shape) == cnt, "Shape and count are not matched"
            data = []
            for i in range(cnt):
                x = np.random.uniform(-1, 1, shape[i])
                vx = Variable(x, _require_grad=True, dtype=np.float64)
                tx = torch.tensor(x, requires_grad=True, dtype=torch.float64)
                data.append((vx, tx))
            return data

    @staticmethod
    def compare_results(
        name: str,
        l_var: list[Variable] | Variable,
        l_tensor: list[torch.Tensor] | torch.Tensor,
        vx: list[Variable] | Variable,
        tx: list[torch.Tensor] | torch.Tensor,
    ):
        """断言数值和梯度的一致性"""
        if not isinstance(l_var, list):
            l_var = [l_var]
        if not isinstance(l_tensor, list):
            l_tensor = [l_tensor]
        if not isinstance(vx, list):
            vx = [vx]
        if not isinstance(tx, list):
            tx = [tx]

        for l_v, l_t, v, t in zip(l_var, l_tensor, vx, tx):
            assert t.grad is not None, f"{name} Gradient is not computed"
            assert np.allclose(l_v, l_t.detach().numpy(), equal_nan=True), (
                f"{name} is not correct"
            )
            assert np.allclose(v.grad, t.grad.numpy(), equal_nan=True), (
                f"{name} Gradient is not correct"
            )

    @pytest.mark.add
    def test_add(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx + vy
        l_var.backward()

        l_tensor = tx + ty
        l_tensor.sum().backward()

        self.compare_results("Addition", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.add
    def test_add_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx + vy
        l_var.backward()

        l_tensor = tx + ty
        l_tensor.sum().backward()

        self.compare_results("Addition Broadcast", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.mul
    def test_mul(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx * vy
        l_var.backward()

        l_tensor = tx * ty
        l_tensor.sum().backward()

        self.compare_results("Multiplication", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.mul
    def test_mul_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx * vy
        l_var.backward()

        l_tensor = tx * ty
        l_tensor.sum().backward()

        self.compare_results(
            "Multiplication Broadcast", l_var, l_tensor, [vx, vy], [tx, ty]
        )

    @pytest.mark.sub
    def test_sub(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx - vy
        l_var.backward()

        l_tensor = tx - ty
        l_tensor.sum().backward()

        self.compare_results("Subtraction", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.sub
    def test_sub_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx - vy
        l_var.backward()

        l_tensor = tx - ty
        l_tensor.sum().backward()

        self.compare_results(
            "Subtraction Broadcast", l_var, l_tensor, [vx, vy], [tx, ty]
        )

    def _test_pow(self, p):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = vx**p
        l_var.backward()

        l_tensor = tx**p
        l_tensor.sum().backward()

        self.compare_results(f"Power {p}", l_var, l_tensor, vx, tx)

    @pytest.mark.pow
    def test_pow(self):
        p_list = [-1, 1, 0.5, 100, -0.5, 1.5]
        for p in p_list:
            self._test_pow(p)

    @pytest.mark.div
    def test_div(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx / vy
        l_var.backward()

        l_tensor = tx / ty
        l_tensor.sum().backward()

        self.compare_results("Division", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.div
    def test_div_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx / vy
        l_var.backward()

        l_tensor = tx / ty
        l_tensor.sum().backward()

        self.compare_results("Division Broadcast", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.exp
    def test_exp(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.exp(vx)
        l_var.backward()

        l_tensor = torch.exp(tx)
        l_tensor.sum().backward()

        self.compare_results("Exp Broadcast", l_var, l_tensor, vx, tx)

    @pytest.mark.matmul
    def test_matmul(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(10, 100), (100, 10)])

        l_var = vx @ vy
        l_var.backward()

        l_tensor = tx @ ty
        l_tensor.sum().backward()

        self.compare_results("Matmul", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.sum
    def test_sum_all(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.sum(vx)
        l_var.backward()

        l_tensor = tx.sum()
        l_tensor.backward()

        self.compare_results("Sum all", l_var, l_tensor, vx, tx)

    @pytest.mark.sum
    def test_sum_slice_single(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.sum(vx, axis=1)
        l_var.backward()

        l_tensor = tx.sum(dim=1)
        l_tensor.sum().backward()

        self.compare_results("Sum slice single", l_var, l_tensor, vx, tx)

    @pytest.mark.sum
    def test_sum_slice_multiple(self):
        ((vx, tx),) = self.generate_compare_data((2, 3, 4, 5), cnt=1)

        l_var = Variable.sum(vx, axis=(1, 2))
        l_var.backward()

        l_tensor = tx.sum(dim=(1, 2))
        l_tensor.sum().backward()

        self.compare_results("Sum slice multiple", l_var, l_tensor, vx, tx)

    @pytest.mark.log
    def test_log(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)
        l_var = Variable.log(vx)
        l_var.backward()

        l_tensor = torch.log(tx)
        l_tensor.sum().backward()

        self.compare_results("log", l_var, l_tensor, vx, tx)

    @pytest.mark.slice
    def test_slice_simple(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)
        l_var = vx[:, 1:]
        l_var.backward()

        l_tensor = tx[:, 1:]
        l_tensor.sum().backward()

        self.compare_results("Slice simple", l_var, l_tensor, vx, tx)

    @pytest.mark.slice
    def test_slice_complex(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((2, 3, 4, 5))
        l_var = vx[:, 1:3, 2:4, :]
        l_var = l_var**2 + 3 * vy[:, 1:3, 2:4, :]
        l_var.backward()

        l_tensor = tx[:, 1:3, 2:4, :]
        l_tensor = l_tensor**2 + 3 * ty[:, 1:3, 2:4, :]
        l_tensor.sum().backward()

        self.compare_results("Slice complex", l_var, l_tensor, [vx, vy], [tx, ty])

    @pytest.mark.slice
    def test_slice_variable(self):
        ((vx, tx),) = self.generate_compare_data((2, 3, 4, 5), cnt=1)
        slice_tensor = torch.tensor([1, 0], dtype=torch.long)
        l_tensor = tx[slice_tensor, ...]
        l_tensor.sum().backward()

        slice_var = Variable([1, 0], dtype=np.long)
        l_var = vx[slice_var, ...]
        l_var.backward()

        self.compare_results("Slice variable", l_var, l_tensor, vx, tx)

    @pytest.mark.integration
    def test_cross_entropy_sample(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)
        y = np.random.randint(0, 2, 3)
        vy = Variable(y, dtype=np.long)
        ty = torch.tensor(y, dtype=torch.long)

        ax_t = tx[torch.arange(tx.shape[0]), ty]
        ax_v = vx[Variable(np.arange(vx.shape[0])), vy]

        assert np.allclose(ax_t.detach().numpy(), ax_v.value), (
            "CrossEntropy Sample(getitem) is not correct"
        )

        exp_t = torch.exp(ax_t)
        exp_v = Variable.exp(ax_v)

        assert np.allclose(exp_t.detach().numpy(), exp_v.value), (
            "CrossEntropy Sample(exp) is not correct"
        )

        sum_exp_t = torch.sum(torch.exp(tx), dim=1)
        sum_exp_v = Variable.sum(Variable.exp(vx), axis=1)

        assert np.allclose(sum_exp_t.detach().numpy(), sum_exp_v.value), (
            "CrossEntropy Sample(sum) is not correct"
        )

        cross_entropy_t = -torch.log(exp_t / sum_exp_t)
        cross_entropy_v = -Variable.log(exp_v / sum_exp_v)

        assert np.allclose(cross_entropy_t.detach().numpy(), cross_entropy_v.value), (
            "CrossEntropy Sample is not correct"
        )

        loss_t = torch.sum(cross_entropy_t)
        loss_v = Variable.sum(cross_entropy_v)
        loss_t.backward()
        loss_v.backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(loss_t.detach().numpy(), loss_v.value), (
            "CrossEntropy Sample Loss is not correct"
        )
        assert np.allclose(tx.grad.numpy(), vx.grad), (
            "CrossEntropy Sample Gradient is not correct"
        )

    @pytest.mark.mean
    def test_mean_sample(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.mean(vx)
        l_var.backward()

        l_tensor = torch.mean(tx)
        l_tensor.backward()

        self.compare_results("Mean", l_var, l_tensor, vx, tx)

    @pytest.mark.mean
    def test_mean_complex(self):
        ((vx, tx),) = self.generate_compare_data((2, 3, 4, 5), cnt=1)

        l_var = Variable.mean(vx, axis=(1, 2))
        l_var.backward()

        l_tensor = torch.mean(tx, dim=(1, 2))
        l_tensor.sum().backward()

        self.compare_results("Mean Complex", l_var, l_tensor, vx, tx)

    @pytest.mark.relu
    def test_relu(self):
        ((vx, tx),) = self.generate_compare_data((100, 10), cnt=1)

        l_var = Variable.relu(vx)
        l_var.backward()

        l_tensor = torch.relu(tx)
        l_tensor.sum().backward()

        self.compare_results("Relu", l_var, l_tensor, vx, tx)

    @pytest.mark.tanh
    def test_tanh(self) -> None:
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        l_var = Variable.tanh(vx)
        l_var.backward()

        l_tensor = torch.tanh(tx)
        l_tensor.sum().backward()

        self.compare_results("Tanh", l_var, l_tensor, vx, tx)

    @pytest.mark.reshape
    def test_reshape(self) -> None:
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        l_var = vx.reshape(1, 6)
        l_var.backward()

        l_tensor = tx.reshape(1, 6)
        l_tensor.sum().backward()

        self.compare_results("Reshape", l_var, l_tensor, vx, tx)

    @pytest.mark.max
    def test_max_axis_0(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        # NumPy 计算 max
        l_var = vx.max(0)
        l_var.backward()

        # PyTorch 计算 max
        l_tensor, _ = torch.max(tx, 0)
        l_tensor.sum().backward()

        self.compare_results("Max", l_var, l_tensor, vx, tx)

    @pytest.mark.max
    def test_max_axis_1(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = vx.max(1)
        l_var.backward()

        l_tensor, _ = torch.max(tx, 1)
        l_tensor.sum().backward()

        self.compare_results("Max", l_var, l_tensor, vx, tx)

    @pytest.mark.max
    def test_max_global(self):
        """max(axis=None) 全局最大值"""
        ((vx, tx),) = self.generate_compare_data((3, 3), cnt=1)

        l_var = vx.max()
        l_var.backward()

        l_tensor = torch.max(tx)
        l_tensor.backward()

        self.compare_results("Max", l_var, l_tensor, vx, tx)

    @pytest.mark.max
    def test_max_axis_0_keepdims(self):
        ((vx, tx),) = self.generate_compare_data((4, 2), cnt=1)

        l_var = vx.max(0, keepdims=True)
        l_var.backward()

        l_tensor, _ = torch.max(tx, 0)
        l_tensor = l_tensor.unsqueeze(0)  # 模拟 keepdims
        l_tensor.sum().backward()

        self.compare_results("Max", l_var, l_tensor, vx, tx)

    @pytest.mark.max
    def test_max_axis_1_keepdims(self):
        """max(axis=1, keepdims=True)"""
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        l_var = vx.max(1, keepdims=True)
        l_var.backward()

        l_tensor, _ = torch.max(tx, 1)
        l_tensor = l_tensor.unsqueeze(1)  # 模拟 keepdims
        l_tensor.sum().backward()

        self.compare_results("Max", l_var, l_tensor, vx, tx)

    @pytest.mark.min
    def test_min_axis_0(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        # NumPy 计算 min
        l_var = vx.min(0)
        l_var.backward()

        # PyTorch 计算 min
        l_tensor, _ = torch.min(tx, 0)
        l_tensor.sum().backward()

        self.compare_results("Min", l_var, l_tensor, vx, tx)

    @pytest.mark.min
    def test_min_axis_1(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = vx.min(1)
        l_var.backward()

        l_tensor, _ = torch.min(tx, 1)
        l_tensor.sum().backward()

        self.compare_results("Min", l_var, l_tensor, vx, tx)

    @pytest.mark.min
    def test_min_global(self):
        """min(axis=None) 全局最大值"""
        ((vx, tx),) = self.generate_compare_data((3, 3), cnt=1)

        l_var = vx.min()
        l_var.backward()

        l_tensor = torch.min(tx)
        l_tensor.backward()

        self.compare_results("Min", l_var, l_tensor, vx, tx)

    @pytest.mark.min
    def test_min_axis_0_keepdims(self):
        ((vx, tx),) = self.generate_compare_data((4, 2), cnt=1)

        l_var = vx.min(0, keepdims=True)
        l_var.backward()

        l_tensor, _ = torch.min(tx, 0)
        l_tensor = l_tensor.unsqueeze(0)  # 模拟 keepdims
        l_tensor.sum().backward()

        self.compare_results("Min", l_var, l_tensor, vx, tx)

    @pytest.mark.min
    def test_min_axis_1_keepdims(self):
        """min(axis=1, keepdims=True)"""
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        l_var = vx.min(1, keepdims=True)
        l_var.backward()

        l_tensor, _ = torch.min(tx, 1)
        l_tensor = l_tensor.unsqueeze(1)  # 模拟 keepdims
        l_tensor.sum().backward()

        self.compare_results("Min", l_var, l_tensor, vx, tx)
