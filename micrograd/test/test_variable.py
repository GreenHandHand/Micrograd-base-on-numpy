from ..variable import Variable

import torch
import numpy as np


class TestVariable:
    def generate_compare_data(
        self, shape: tuple[int, ...] | list[tuple[int, ...]], cnt: int = 2
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

    def test_add(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l1 = vx + vy
        l1.backward()

        l2 = tx + ty
        l3 = torch.sum(l2)
        l3.backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l1.value, l2.detach().numpy()), "Addition is not correct"

    def test_add_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx + vy
        l_var.backward()

        l_tensor = tx + ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Broadcast addition is not correct"
        )

    def test_mul(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx * vy
        l_var.backward()

        l_tensor = tx * ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Multiplication is not correct"
        )

    def test_mul_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx * vy
        l_var.backward()

        l_tensor = tx * ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Broadcast multiplication is not correct"
        )

    def test_sub(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx - vy
        l_var.backward()

        l_tensor = tx - ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Subtraction is not correct"
        )

    def test_sub_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx - vy
        l_var.backward()

        l_tensor = tx - ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Broadcast subtraction is not correct"
        )

    def _test_pow(self, p):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = vx**p
        l_var.backward()

        l_tensor = tx**p
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy(), equal_nan=True), (
            "Gradient is not correct"
        )
        assert np.allclose(l_var, l_tensor.detach().numpy(), equal_nan=True), (
            "Power is not correct"
        )

    def test_pow(self):
        p_list = [-1, 1, 0.5, 100, -0.5, 1.5]
        for p in p_list:
            self._test_pow(p)

    def test_div(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((3, 2))

        l_var = vx / vy
        l_var.backward()

        l_tensor = tx / ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Division is not correct"

    def test_div_broadcast(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(3, 2), (2,)])

        l_var = vx / vy
        l_var.backward()

        l_tensor = tx / ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Broadcast division is not correct"
        )

    def test_exp(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.exp(vx)
        l_var.backward()

        l_tensor = torch.exp(tx)
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Exponential is not correct"
        )
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"

    def test_matmul(self):
        (vx, tx), (vy, ty) = self.generate_compare_data([(10, 100), (100, 10)])

        l_var = vx @ vy
        l_var.backward()

        l_tensor = tx @ ty
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "Matrix multiplication is not correct"
        )

    def test_sum_all(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.sum(vx)
        l_var.backward()

        l_tensor = tx.sum()
        l_tensor.backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Sum is not correct"

    def test_sum_slice_single(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.sum(vx, axis=1)
        l_var.backward()

        l_tensor = tx.sum(dim=1)
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Sum is not correct"

    def test_sum_slice_multiple(self):
        ((vx, tx),) = self.generate_compare_data((2, 3, 4, 5), cnt=1)

        l_var = Variable.sum(vx, axis=(1, 2))
        l_var.backward()

        l_tensor = tx.sum(dim=(1, 2))
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Sum is not correct"

    def test_log(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)
        l_var = Variable.log(vx)
        l_var.backward()

        l_tensor = torch.log(tx)
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(vx.grad, tx.grad.numpy(), equal_nan=True), (
            "Gradient is not correct"
        )
        assert np.allclose(l_var, l_tensor.detach().numpy(), equal_nan=True), (
            "Log is not correct"
        )

    def test_slice_smaple(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)
        l_var = vx[:, 1:]
        l_var.backward()

        l_tensor = tx[:, 1:]
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var.value, l_tensor.detach().numpy()), (
            "Simple Slice is not correct"
        )
        assert np.allclose(vx.grad, tx.grad.numpy()), "Simple Gradient is not correct"

    def test_slice_complex(self):
        (vx, tx), (vy, ty) = self.generate_compare_data((2, 3, 4, 5))
        l_var = vx[:, 1:3, 2:4, :]
        l_var = l_var**2 + 3 * vy[:, 1:3, 2:4, :]
        l_var.backward()

        l_tensor = tx[:, 1:3, 2:4, :]
        l_tensor = l_tensor**2 + 3 * ty[:, 1:3, 2:4, :]
        l_tensor.sum().backward()

        assert tx.grad is not None and ty.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var.value, l_tensor.detach().numpy()), (
            "Complex Slice is not correct"
        )
        assert np.allclose(vx.grad, tx.grad.numpy()) and np.allclose(
            vy.grad, ty.grad.numpy()
        ), "Complex Slice Gradient is not correct"

    def test_slice_variable(self):
        ((vx, tx),) = self.generate_compare_data((2, 3, 4, 5), cnt=1)
        slice_tensor = torch.tensor([1, 0], dtype=torch.long)
        l_tensor = tx[slice_tensor, ...]
        l_tensor.sum().backward()

        slice_var = Variable([1, 0], dtype=np.long)
        l_var = vx[slice_var, ...]
        l_var.backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), (
            "VariableIndex Slice is not correct"
        )
        assert np.allclose(vx.grad, tx.grad.numpy()), (
            "VariableIndex Slice Gradient is not correct"
        )

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

    def test_mean_sample(self):
        ((vx, tx),) = self.generate_compare_data((3, 2), cnt=1)

        l_var = Variable.mean(vx)
        l_var.backward()

        l_tensor = torch.mean(tx)
        l_tensor.backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Mean is not correct"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"

    def test_mean_complex(self):
        ((vx, tx),) = self.generate_compare_data((2, 3, 4, 5), cnt=1)

        l_var = Variable.mean(vx, axis=(1, 2))
        l_var.backward()

        l_tensor = torch.mean(tx, dim=(1, 2))
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Mean is not correct"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"

    def test_relu(self):
        ((vx, tx),) = self.generate_compare_data((2, 3), cnt=1)

        l_var = Variable.relu(vx)
        l_var.backward()

        l_tensor = torch.relu(tx)
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "ReLU is not correct"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"

    def test_tanh(self) -> None:
        ((vx, tx), ) = self.generate_compare_data((2, 3), cnt = 1)

        l_var = Variable.tanh(vx)
        l_var.backward()

        l_tensor = torch.tanh(tx)
        l_tensor.sum().backward()
        
        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Tanh is not correct"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"

    def test_reshape(self) -> None:
        ((vx, tx), ) = self.generate_compare_data((2, 3), cnt = 1)

        l_var = vx.reshape(1, 6)
        l_var.backward()
        
        l_tensor = tx.reshape(1, 6)
        l_tensor.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(l_var, l_tensor.detach().numpy()), "Reshape is not correct"
        assert np.allclose(vx.grad, tx.grad.numpy()), "Gradient is not correct"