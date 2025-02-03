import numpy as np
from numpy.typing import NDArray
from typing import Any


def sum_broadcast_dim(source: NDArray[Any], target: NDArray[Any]) -> NDArray[Any]:
    # 检查目标数组是否可以广播到源数组的形状
    try:
        np.broadcast_shapes(source.shape, target.shape)
    except ValueError as e:
        raise ValueError(
            f"Target shape {target.shape} cannot be broadcast to source shape {source.shape}"
        ) from e

    s_ndim = source.ndim
    t_ndim = target.ndim
    max_ndim = max(s_ndim, t_ndim)

    # 对齐维度，前面补1
    s_padded = (1,) * (max_ndim - s_ndim) + source.shape
    t_padded = (1,) * (max_ndim - t_ndim) + target.shape

    # 验证每个维度是否符合广播规则
    for s_dim, t_dim in zip(s_padded, t_padded):
        if not (t_dim == 1 or s_dim == t_dim or s_dim == 1):
            raise ValueError(
                f"Target shape {target.shape} cannot be broadcast to source shape {source.shape}"
            )

    current = source.copy()

    # 遍历每个维度，处理需要求和的轴
    for i in range(max_ndim):
        t_dim = t_padded[i]
        s_dim = s_padded[i]
        if t_dim == 1 and s_dim != 1:
            current = np.sum(current, axis=i, keepdims=True)

    # 去除前面补的1的维度
    if max_ndim > t_ndim:
        squeeze_axes = tuple(range(max_ndim - t_ndim))
        current = current.squeeze(axis=squeeze_axes)

    # 确保最终形状与目标一致
    if current.shape != target.shape:
        raise RuntimeError(
            f"Resulting shape {current.shape} does not match target shape {target.shape}"
        )

    return current
