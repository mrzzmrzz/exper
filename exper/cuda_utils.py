from collections.abc import Mapping, Sequence
from typing import TypeGuard

import torch
from torch import Tensor


def is_tensor_list(objs: Sequence[object]) -> TypeGuard[list[Tensor]]:
    return isinstance(objs[0], Tensor)


def is_sequence_list(objs: Sequence[object]) -> TypeGuard[list[Sequence]]:
    return isinstance(objs[0], Sequence)


def is_mapping_list(objs: Sequence[object]) -> TypeGuard[list[Mapping]]:
    return isinstance(objs[0], Mapping)


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    raise TypeError("Can't transfer object type `%s`" % type(obj))


def mean(obj, *args, **kwargs):
    """
    Compute mean of tensors in any nested container.
    """
    if hasattr(obj, "mean"):
        return obj.mean(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: mean(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(mean(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't perform mean over object type `%s`" % type(obj))


def cat(objs: list[object], *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    if is_tensor_list(objs):
        return torch.cat(objs, *args, **kwargs)
    elif is_sequence_list(objs):
        return type(objs[0])(
            {k: cat([x[k] for x in objs], *args, **kwargs) for k in objs[0]}
        )
    elif is_sequence_list(objs):
        return type(objs[0])(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(objs[0]))


def stack(objs: Sequence[object], *args, **kwargs):
    """
    Stack a list of nested containers with the same structure.
    """
    if is_tensor_list(objs):
        return torch.stack(objs, *args, **kwargs)
    elif is_mapping_list(objs):
        return type(objs[0])(
            {k: stack([x[k] for x in objs], *args, **kwargs) for k in objs[0]}
        )
    elif is_sequence_list(objs):
        return type(objs[0])(stack(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform stack over object type `%s`" % type(objs[0]))
