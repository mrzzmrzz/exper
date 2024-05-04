import inspect
from typing import Any, Dict, List

import torch
from torch.nn.modules.loss import _Loss


def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolver(
    classes: List[Any],
    class_dict: Dict[str, Any],
    query: str | Any,
    base_cls: type | None,
    base_cls_repr: str | None,
    *args: Any,
    **kwargs: Any,
):
    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)

    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ""
    base_cls_repr = normalize_string(base_cls_repr)
    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                return obj
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, "")]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                return obj
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


def loss_function_resolver(query: str | Any, *args, **kwargs):
    base_cls = _Loss
    loss_function = [
        loss_func
        for loss_func in vars(torch.nn.modules.loss).values()
        if isinstance(loss_func, type) and issubclass(loss_func, base_cls)
    ]

    return resolver(loss_function, {}, query, base_cls, None, *args, **kwargs)
