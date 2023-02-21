import torch
import typing
import functools

# Device to be used for computation (GPU if available, else CPU).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, torch.nn.Module, typing.List[torch.Tensor],
                 tuple, dict, typing.Generator]


def device(x: T, dev: str = DEVICE, throw: bool = True) -> T:
    """Convenience method to move a tensor/module/other structure containing tensors to the device.
    Args:
        x (T): the tensor (or strucure containing tensors)
        dev (str, optional): the device to move the tensor to. Defaults to DEVICE.
        throw (bool, optional): whether to throw an error if the type is not a compatible tensor. Defaults to True.
    Raises:
        TypeError: if the type was not a compatible tensor
    Returns:
        T: the input tensor moved to the device
    """

    to = functools.partial(device, dev=dev, throw=False)
    if isinstance(x, (torch.Tensor, torch.nn.Module)):
        return x.to(dev)
    elif isinstance(x, list):
        return list(map(to, x))
    elif isinstance(x, tuple):
        # Check if it is a named tuple
        if hasattr(x, "_fields") and hasattr(x, "_asdict"):
            return type(x)(**to(x._asdict()))
        return tuple(map(to, x))
    elif isinstance(x, dict):
        return {k: to(v) for k, v in x.items()}
    elif isinstance(x, typing.Iterable):
        return map(to, x)
    elif throw:
        raise TypeError(f"Cannot move type {type(x).__name__} to {dev}.")
    return x
