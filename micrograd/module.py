from collections import OrderedDict
from typing import Any, Iterable, Literal
from abc import abstractmethod

import numpy as np

from .variable import Variable


class Module:
    def __init__(self) -> None:
        self._module = OrderedDict()
        self._parameters = OrderedDict()

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def __str__(self) -> str:
        s: list[str] = []
        for name, m in self._parameters.items():
            m_str = str(m).replace("\n", "\n  ")
            s.append(f"  {name}: " + m_str)
        return f"""{self.__class__.__name__}(\n{"\n".join(s)}\n)"""

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def parameters(self) -> list[Variable]:
        def generate():
            for p in self._parameters.values():
                if p._require_grad:
                    yield p

            for m in self._module.values():
                yield from m.parameters()

        return [p for p in generate()]

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self._module[name] = value

        super().__setattr__(name, value)


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self._w = Variable(np.random.uniform(-1, 1, size=(input_size, output_size)))
        self._b = Variable(np.zeros(output_size))

        self._w._require_grad = True
        self._b._require_grad = True
        self._parameters = OrderedDict(_w=self._w, _b=self._b)

    def forward(self, x: Variable):
        return x @ self._w + self._b

    def __str__(self) -> str:
        return f"Linear(input_size={self.input_size}, output_size={self.output_size})"

    def __repr__(self) -> str:
        return self.__str__()


class SGD:
    def __init__(
        self,
        parameters: list[Variable],
        lr: float = 1e-3,
        *,
        weight_decay: float = 0,
        momentum: float = 0,
        dampening: float = 0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = 1 - dampening

        self._momentum = None if momentum == 0 else {}

    def zero_grad(self):
        for p in self.parameters:
            p.grad.fill(0)

    def step(self):
        for p in self.parameters:
            delta = p.grad
            if self.weight_decay != 0:
                delta += self.weight_decay * p.value
            if self._momentum is not None:
                if self._momentum.get(id(p)) is None:
                    self._momentum[id(p)] = delta
                else:
                    self._momentum[id(p)] = (
                        self.momentum * self._momentum[id(p)] + self.dampening * delta
                    )
                delta = self._momentum[id(p)]

            p.value -= self.lr * delta


class CrossEntropyLoss:
    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        self.reduction = reduction

    def __call__(self, y_pred: Variable, y_true: Variable) -> Variable:
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size, )
        assert y_pred.shape[0] == y_true.shape[0], (
            f"Batch size is not matched, expected {y_pred.shape[0]} but got {y_true.shape[0]}"
        )
        assert y_true.dtype in (np.int32, np.int64, np.long), (
            f"Data type is not matched, expected long but got {y_true.dtype}"
        )
        cross_entropy = -Variable.log(
            Variable.exp(y_pred[np.arange(y_pred.shape[0]), y_true])
            / Variable.sum(Variable.exp(y_pred), axis=1)
        )

        if self.reduction == "mean":
            return Variable.mean(cross_entropy)
        elif self.reduction == "sum":
            return Variable.sum(cross_entropy)
        elif self.reduction == "none":
            return cross_entropy
        else:
            raise ValueError(f"Reduction method {self.reduction} is not supported")
