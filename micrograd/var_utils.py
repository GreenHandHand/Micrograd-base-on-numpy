from typing import Any

import numpy as np
from numpy.typing import NDArray


def sum_broadcast_dim(source: NDArray[Any], target: NDArray[Any]) -> NDArray[Any]:
    # 检查目标数组是否可以广播到源数组的形状
    try:
        shape = np.broadcast_shapes(source.shape, target.shape)
    except ValueError as e:
        raise ValueError(
            f"Target shape {target.shape} cannot be broadcast to source shape {source.shape}"
        ) from e

    # 可以广播，创建一个副本
    current = source.copy()
    res_shape = target.shape

    # 将目标的维度扩展到与源相同
    for i in range(len(shape)):
        if len(target.shape) <= i:
            target = target[np.newaxis]
        elif target.shape[i] != shape[i]:
            target = np.expand_dims(target, axis=i)

    # 将对应的维度进行求和
    for i, (source_dim, target_dim) in enumerate(zip(source.shape, target.shape)):
        if source_dim != 1 and target_dim == 1:
            current = current.sum(axis=i, keepdims=True)
        elif source_dim != target_dim:
            raise ValueError(
                f"Dimension {i} mismatch: source {source_dim} and target {target_dim}"
            )

    return current.reshape(res_shape)
