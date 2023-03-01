import torch
import typing
import functools

# Device to be used for computation (GPU if available, else CPU).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = typing.Union[torch.Tensor, torch.nn.Module, typing.List[torch.Tensor],
                 tuple, dict, typing.Generator]

def apply(x: T, /,
          tensor_func: typing.Callable[[torch.Tensor], torch.Tensor], *,
          module_func: typing.Callable[[torch.nn.Module], torch.nn.Module] = None,
          throw: bool = True, 
          **kwargs) -> T:
    if module_func is None:
        module_func = tensor_func
    recurse = functools.partial(apply, tensor_func=tensor_func, module_func=module_func, throw=False, **kwargs)
    if isinstance(x, torch.Tensor):
        return tensor_func(x, **kwargs)
    elif isinstance(x, torch.nn.Module):
        return module_func(x, **kwargs)
    elif isinstance(x, list):
        return list(map(recurse, x))
    elif isinstance(x, tuple):
        # Check if it is a named tuple
        if hasattr(x, "_fields") and hasattr(x, "_asdict"):
            return type(x)(**recurse(x._asdict()))
        return tuple(map(recurse, x))
    elif isinstance(x, dict):
        return {k: recurse(v) for k, v in x.items()}
    elif isinstance(x, typing.Iterable):
        return map(recurse, x)
    elif throw:
        raise TypeError(f"Cannot apply {tensor_func:r} to {type(x).__name__}.")
    return x

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
    return apply(x,
                 tensor_func=functools.partial(torch.Tensor.to, device=dev),
                 module_func=functools.partial(torch.nn.Module.to, device=dev),
                 throw=throw)

detach = functools.partial(apply, tensor_func=torch.detach)