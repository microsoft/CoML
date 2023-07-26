import inspect
import itertools
from types import FrameType
from typing import List

def nearest_user_frame() -> FrameType:
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            return frame_info.frame
        if not (module.__name__.startswith("coml") or module.__name__.startswith("inspect")):
            return frame_info.frame
    raise RuntimeError("No user frame found.")

def reproduce_function_parameters(*args, **kwargs) -> str:
    """Get the variable names that user has passed to us.
    
    The hacks below are inspired by https://stackoverflow.com/q/2749796
    """
    frame = nearest_user_frame()
    parameters: List[str] = []
    for arg in args:
        for k, v in itertools.chain(frame.f_locals.items(), frame.f_globals.items()):
            if v is arg:
                parameters.append(k)
                break
        else:
            parameters.append(repr(arg))  # literal?

    for keyword, arg in kwargs.items():
        for k, v in itertools.chain(frame.f_locals.items(), frame.f_globals.items()):
            if v is arg:
                parameters.append(f"{keyword}={k}")
                break
        else:
            parameters.append(f"{keyword}={repr(arg)}")

    return ', '.join(parameters)


class ContextGrapser:
    pass
