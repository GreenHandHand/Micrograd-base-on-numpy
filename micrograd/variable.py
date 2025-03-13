from __future__ import annotations

import os
from typing import Callable, Self

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from .var_utils import sum_broadcast_dim


class Variable:
    def __init__(
        self,
        value: ArrayLike | int | float,
        dtype: DTypeLike | None = None,
        *,
        _children: tuple[Variable, ...] = (),
        _op: str = "",
        _require_grad: bool = False,
    ) -> None:
        self.value: np.ndarray = np.array(value, dtype=dtype)
        self.grad: np.ndarray = np.zeros_like(value, dtype=self.value.dtype)
        self.dtype = self.value.dtype

        self._require_grad = _require_grad
        self._backward: Callable = lambda: None
        self._prev = set(_children)
        self._op = _op

        self.shape = self.value.shape
        self.size = self.value.size

    def __str__(self) -> str:
        item = {"value": self.value}
        if self._require_grad:
            item["grad"] = self.grad

        if self.value.ndim >= 2:
            return f"""Variable(\n{",\n".join([f"{key}:\n{value}" for key, value in item.items()])}\n)"""
        else:
            return f"""Variable({", ".join([f"{key}:{value}" for key, value in item.items()])})"""

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return self.value.size

    def __getitem__(self, key) -> Variable:
        if isinstance(key, Variable):
            assert key.dtype in (np.long, np.int32, np.int64), "Index must be integer"
            key = key.value
        elif isinstance(key, tuple):
            key = tuple(
                np.astype(k.value, np.int32) if isinstance(k, Variable) else k
                for k in key
            )

        out = Variable(
            self.value[key],
            dtype=self.dtype,
            _children=(self,),
            _op="silce",
            _require_grad=self._require_grad,
        )

        def _backward():
            if self._require_grad:
                self.grad[key] += out.grad

        out._backward = _backward
        return out

    def __setitem__(self, key, value) -> Self:
        if isinstance(key, Variable):
            key = key.value
        if isinstance(value, Variable):
            self.value[key] = value.value
        else:
            self.value[key] = value
        return self

    def __array__(self, dtype: DTypeLike | None = None) -> np.ndarray:
        return self.value if dtype is None else self.value.astype(dtype)

    def reshape(self, *shape: int) -> Variable:
        reshaped_var = self.copy()
        reshaped_var.value = self.value.reshape(*shape)
        reshaped_var.grad = self.grad.reshape(*shape)
        reshaped_var.shape = shape
        return reshaped_var

    def copy(self) -> Variable:
        return Variable(
            self.value.copy(),
            dtype=self.dtype,
            _children=tuple(self._prev),
            _op=self._op,
        )

    # ============================================================
    # 比较重要的运算符重载, 包含了加法、乘法、指数
    # 这些重载实现了反向传播
    # ============================================================
    def __add__(self, other: Variable | int | float) -> Variable:
        """
        已知 x + y = out, d(loss)/d(out) = out.grad
        求 d(loss)/d(x) 和 d(loss)/d(y)

        由于 d(out) = dx + dy => d(loss) = out.grad * (dx + dy)
        于是 d(loss)/d(x) = out.grad, d(loss)/d(y) = out.grad
        """
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value, _children=(self, other), _op="+")
        out._require_grad = self._require_grad or other._require_grad

        def _backward():
            if self._require_grad:
                self.grad += sum_broadcast_dim(out.grad, self.grad)
            if other._require_grad:
                other.grad += sum_broadcast_dim(out.grad, other.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: Variable | int | float) -> Variable:
        """
        已知 X * Y = out, d(loss)/d(out) = out.grad
        求 d(loss)/d(x) 和 d(loss)/d(y)

        由于 d(out) = d(x * y) = xdy + ydx => d(loss) = out.grad * (xdy + ydx)
        于是 d(loss)/d(x) = out.grad * y, d(loss)/d(y) = out.grad * x
        """
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(
            self.value * other.value, dtype=self.dtype, _children=(self, other), _op="*"
        )
        out._require_grad = self._require_grad or other._require_grad

        def _backward():
            """
            处理 broad casting
            - a(n, m) * b(n, 1) => a(n, m) * b(n, m) = out(n, m)
            - grad_out(n, m) * b(n, 1) => grad_a(n, m) (broad cast)
            - grad_out(n, m) * a(n, m) => grad_b(n, m).sum(axis=1) => grad_b(n, 1)

            因此, 对于被广播的维度, 需要对其求和
            """
            if self._require_grad:
                _grad = out.grad * other.value
                self.grad += sum_broadcast_dim(_grad, self.grad)
            if other._require_grad:
                _grad = out.grad * self.value
                other.grad += sum_broadcast_dim(_grad, other.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other: Variable) -> Variable:
        """已知 X @ Y = out, d(loss)/d(out) = out.grad
        求 d(loss)/d(x) 和 d(loss)/d(y)

        这是一个复杂的问题, 用简单的两行无法表示，这里简要描述一下
        dA[i, j] = sum(dC[i, :] * B[j, :]) => dA = dC @ B.T
        dB[j, k] = sum(dC[:, k] * A[:, j]) => dB = A.T @ dC
        """
        out = Variable(
            self.value @ other.value, self.dtype, _children=(self, other), _op="@"
        )
        out._require_grad = self._require_grad or other._require_grad

        def _backward():
            if self._require_grad:
                self.grad += out.grad @ other.value.T
            if other._require_grad:
                other.grad += self.value.T @ out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: int | float) -> Variable:
        """
        已知 X ** p = out, d(loss)/d(out) = out.grad
        求 d(loss)/d(X)

        由于 d(out) = d(x ** p) = p * x ** (p - 1) * dx => d(loss) = out.grad * p * x ** (p - 1) * dx
        于是 d(loss)/d(x) = out.grad * p * x ** (p-1) = out.grad * p * out / x
        """
        out = Variable(
            self.value**power,
            self.dtype,
            _children=(self,),
            _op=f"^{power}",
            _require_grad=self._require_grad,
        )

        def _backward():
            if self._require_grad:
                self.grad += out.grad * power * out.value / self.value

        out._backward = _backward
        return out

    @staticmethod
    def exp(var: Variable | int | float) -> Variable:
        """
        已知 exp(x) = out, d(out)/d(x) = exp(x), d(loss)/d(out) = out.grad
        求 d(loss)/d(x)

        由于 d(out) = exp(x) * dx => d(loss) = out.grad * exp(x) * dx
        于是 d(loss)/d(x) = out.grad * exp(x) = out.grad * out
        """
        var = var if isinstance(var, Variable) else Variable(var)
        out = Variable(np.exp(var.value), var.dtype, _children=(var,), _op="exp")
        out._require_grad = var._require_grad

        def _backward():
            if var._require_grad:
                var.grad += out.grad * out.value

        out._backward = _backward
        return out

    # ===============================================================
    # 下面的重载可以基于之前的反向传播实现，因此不需要再手动实现反向传播
    # ===============================================================

    @staticmethod
    def tanh(var: Variable | int | float) -> Variable:
        """
        已知 tanh(x) = out, d(out)/d(x) = 1 - out^2, d(loss)/d(out) = out.grad
        求 d(loss)/d(x)

        由于 d(out) = (1 - out^2) * dx => d(loss) = out.grad * (1 - out^2) * dx
        于是 d(loss)/d(x) = out.grad * (1 - out^2)
        """
        var = var if isinstance(var, Variable) else Variable(var)
        out = Variable(np.tanh(var.value), var.dtype, _children=(var,), _op="tanh")
        out._require_grad = var._require_grad

        def _backward():
            if var._require_grad:
                var.grad += (1 - out.value**2) * out.grad

        out._backward = _backward
        return out

    @staticmethod
    def relu(var: Variable | int | float) -> Variable:
        var = var if isinstance(var, Variable) else Variable(var)
        out = Variable(var.value, var.dtype, _children=(var,), _op="relu")
        out._require_grad = var._require_grad
        out.value[out.value < 0] = 0

        def _backward():
            if var._require_grad:
                var.grad[out.value > 0] += out.grad[out.value > 0]

        out._backward = _backward
        return out

    @staticmethod
    def sum(var: Variable, axis: tuple[int, ...] | int | None = None) -> Variable:
        """
        由于 dS/dx = 1, 故 d(loss)/d(x_i) = d(loss)/d(S) * d(S)/d(x_i) = d(loss)/d(S)
        """
        out = Variable(
            np.sum(var.value, axis=axis), var.dtype, _children=(var,), _op="sum"
        )
        out._require_grad = var._require_grad

        def _backward():
            if not var._require_grad:
                return
            if axis is None:
                var.grad += out.grad
            elif isinstance(axis, int):
                grad = np.expand_dims(out.grad, axis=axis).repeat(
                    var.grad.shape[axis], axis=axis
                )
                var.grad += grad
            else:
                # Assume out.shape = (2, 3, 4, 5)
                # Sum at axis == [1, 2], out.shape(2, 3, 4, 5) => (2, 5)
                # out.shape(2, 5).expand(axis=[1, 2]) => (2, 1, 1, 4)
                # out.shape(2, 1, 1, 4).repeat(3, axis=1) => (2, 3, 1, 4)
                grad = np.expand_dims(out.grad, axis=axis)
                for ax in axis:
                    grad = grad.repeat(var.grad.shape[ax], axis=ax)
                var.grad += grad

        out._backward = _backward
        return out

    @staticmethod
    def mean(var: Variable, axis: int | tuple[int, ...] | None = None) -> Variable:
        if axis is None:
            return Variable.sum(var) / var.size
        elif isinstance(axis, int):
            return Variable.sum(var, axis=axis) / var.shape[axis]
        elif isinstance(axis, tuple):
            prob = 1
            for ax in axis:
                prob *= var.shape[ax]
            return Variable.sum(var, axis=axis) / prob

    @staticmethod
    def log(var: Variable) -> Variable:
        """
        已知 log(x) = out, d(out)/d(x) = 1/x, d(loss)/d(out) = out.grad
        求 d(loss)/d(x)

        由于 d(loss)/d(x) * d(x)/d(out) = out.grad => d(loss)/d(x) = out.grad * d(out)/d(x)
        于是 d(loss)/d(x) = out.grad / x
        """
        out = Variable(np.log(var.value), var.dtype, _children=(var,), _op="log")
        out._require_grad = var._require_grad

        def _backward():
            if var._require_grad:
                var.grad += out.grad / var.value

        out._backward = _backward
        return out

    def __rmul__(self, other: Variable | int | float) -> Variable:
        return self.__mul__(other)

    def __neg__(self) -> Variable:
        return self.__mul__(-1)

    def __sub__(self, other: Variable | int | float) -> Variable:
        return self.__add__(-other)

    def __truediv__(self, other: Variable | int | float) -> Variable:
        return self.__mul__(other**-1)

    def max(self, axis: int | None = None, keepdims: bool = False) -> Variable:
        idx, out = (
            np.argmax(self.value, axis=axis, keepdims=True),
            np.max(self.value, axis=axis, keepdims=keepdims),
        )
        out = Variable(out, self.dtype, _children=(self,), _op="max")
        out._require_grad = self._require_grad

        def _backward():
            if self._require_grad:
                if axis is None:
                    self.grad.reshape(-1)[idx] = out.grad
                else:
                    temp = np.take_along_axis(self.grad, idx, axis=axis)
                    np.put_along_axis(
                        self.grad, idx, temp + out.grad.reshape(temp.shape), axis=axis
                    )

        out._backward = _backward
        return out
    
    def min(self, axis: int | None = None, keepdims: bool = False) -> Variable:
        idx, out = (
            np.argmin(self.value, axis=axis, keepdims=True),
            np.min(self.value, axis=axis, keepdims=keepdims),
        )
        out = Variable(out, self.dtype, _children=(self,), _op="min")
        out._require_grad = self._require_grad

        def _backward():
            if self._require_grad:
                if axis is None:
                    self.grad.reshape(-1)[idx] = out.grad
                else:
                    temp = np.take_along_axis(self.grad, idx, axis=axis)
                    np.put_along_axis(
                        self.grad, idx, temp + out.grad.reshape(temp.shape), axis=axis
                    )

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def sort_topo(node: Variable):
            if node in visited or not node._require_grad:
                return
            visited.add(node)

            if len(node._prev) == 0:
                topo.append(node)
            else:
                for prev in node._prev:
                    sort_topo(prev)
                topo.append(node)

        sort_topo(self)

        self.grad.fill(1.0)
        for v in reversed(topo):
            v._backward()

    def draw(self, save_dir: str = "test_output", save_name: str = "round_table"):
        try:
            from graphviz import Digraph  # type: ignore

            visited = set()

            def add_edge(dot: Digraph, node: Variable):
                if node in visited:
                    return
                else:
                    visited.add(node)
                if len(node._prev) == 0:
                    dot.node(
                        str(id(node)),
                        label=f"{str(node)}",
                        shape="record",
                    )
                else:
                    for prev in node._prev:
                        add_edge(dot, prev)
                        dot.node(
                            str(id(node)),
                            label="{%s | %s}" % (node._op, str(node)),
                            shape="record",
                        )
                        dot.edge(str(id(prev)), str(id(node)))

            dot = Digraph()
            dot.attr(rankdir="LR")
            add_edge(dot, self)

            os.makedirs(save_dir, exist_ok=True)
            dot.render(os.path.join(save_dir, save_name), format="png", view=True)
        except ImportError as e:
            print(e)
            print("Please install graphviz first")
