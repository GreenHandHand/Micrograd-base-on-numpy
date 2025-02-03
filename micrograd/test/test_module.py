from ..variable import Variable

from ..module import Module, Linear, SGD, CrossEntropyLoss

import numpy as np
import torch
from torch import nn


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


class TestModule:
    def test_MLP(self):
        class MLP(Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()

                self.linear1 = Linear(input_size, hidden_size)
                self.linear2 = Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        from sklearn.datasets import make_moons
        import matplotlib.pyplot as plt

        x, y = make_moons(n_samples=100, noise=0.1)
        x, y = Variable(x), Variable(y, dtype=np.long)
        # plt.scatter(x[:, 0], x[:, 1], c=y)
        # plt.show()

        EPOCH = 5
        print(x.shape, y.shape)

        net = MLP(input_size=2, hidden_size=10, output_size=2)

        print(len(net.parameters()))

        optimizer = SGD(net.parameters(), lr=1e-5)
        loss_fn = CrossEntropyLoss(reduction="sum")

        for epoch in range(EPOCH):
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.value)

    def test_cross_entropy_reduce_none(self):
        ((vx, tx),) = generate_compare_data((3, 2), cnt=1)
        y = np.random.randint(0, 2, 3)
        vy = Variable(y, dtype=np.long)
        ty = torch.tensor(y, dtype=torch.long)

        v_loss_fn = CrossEntropyLoss(reduction="none")
        t_loss_fn = nn.CrossEntropyLoss(reduction="none")

        v_loss = v_loss_fn(vx, vy)
        t_loss = t_loss_fn(tx, ty)

        v_loss.backward()
        t_loss.sum().backward()

        assert tx.grad is not None, "Gradient is not computed"
        assert np.allclose(v_loss.value, t_loss.detach().numpy()), (
            f"CrossEntropyLoss is not correct, diff is {np.linalg.norm(v_loss - t_loss.detach().numpy())}"
        )
        assert np.allclose(vx.grad, tx.grad.numpy()), (
            f"Gradient is not correct, diff is {vx.grad - tx.grad.numpy()}"
        )
