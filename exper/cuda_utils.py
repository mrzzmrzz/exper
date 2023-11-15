from collections.abc import Mapping, Sequence
import torch


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


def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, Mapping):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, Sequence):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(obj))


def stack(objs, *args, **kwargs):
    """
    Stack a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.stack(objs, *args, **kwargs)
    elif isinstance(obj, Mapping):
        return {k: stack([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, Sequence):
        return type(obj)(stack(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform stack over object type `%s`" % type(obj))