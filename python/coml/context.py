import inspect
from types import FrameType

def nearest_user_frame() -> FrameType:
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            return frame_info.frame
        if not (module.__name__.startswith("coml") or module.__name__.startswith("inspect")):
            return frame_info.frame
    raise RuntimeError("No user frame found.")

class ContextGrapser:
    pass
