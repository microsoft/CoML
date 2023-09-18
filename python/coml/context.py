import ast
import inspect
import itertools
from types import FrameType, FunctionType
from typing import List, Any, Sequence, Mapping, Tuple

class _unknown_type:
    pass

_unknown = _unknown_type()

def _eval_expr(namespace: Mapping[str, Any], e: ast.expr) -> Any:
    if isinstance(e, ast.Constant):
        return e.value
    elif isinstance(e, ast.Name):
        return namespace.get(e.id, _unknown)
    else:
        # could be expression, function call...
        return _unknown

def _args_equal(
    args1: Sequence[Any],
    kwargs1: Mapping[str, Any],
    args2: Sequence[Any],
    kwargs2: Mapping[str, Any]
) -> bool:
    if len(args1) != len(args2):
        return False
    if len(kwargs1) != len(kwargs2):
        return False
    for arg1, arg2 in zip(args1, args2):
        if arg1 is not arg2:
            return False
    for k, v in kwargs1.items():
        if k not in kwargs2:
            return False
        if v is not kwargs2[k]:
            return False
    return True

def nearest_user_frame() -> Tuple[inspect.FrameInfo, FrameType]:
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            return frame_info, frame_info.frame
        if not (module.__name__.startswith("coml") or module.__name__.startswith("inspect")):
            return frame_info, frame_info.frame
    raise RuntimeError("No user frame found.")

def frame_namespace(frame: FrameType) -> Mapping[str, Any]:
    frame_namespace = frame.f_globals.copy()
    frame_namespace.update(frame.f_locals)
    return frame_namespace

def reproduce_function_parameters(expected_func: FunctionType, *args: Any, **kwargs: Any) -> str:
    """Get the variable names that user has passed to us.
    
    The hacks below are inspired by https://stackoverflow.com/q/2749796
    """
    frame = nearest_user_frame()
    inspect.getinnerframes()
    source, lineno = inspect.findsource(frame)
    print(frame.f_lineno)
    print()
    print(source)
    print(lineno)

    nodes = ast.parse("".join(source))
    namespace = frame_namespace(frame)

    for node in ast.walk(nodes):
        if isinstance(node, ast.Call):
            if node.lineno <= frame.f_lineno and (
                node.end_lineno is None or frame.f_lineno <= node.end_lineno
            ):
                func = _eval_expr(namespace, node.func)
                arguments = [_eval_expr(namespace, arg) for arg in node.args]
                keywords = {kw.arg: _eval_expr(namespace, kw.value) for kw in node.keywords if isinstance(kw.arg, str)}
                print(node.args)
                print(node.keywords)
                print(func, arguments, keywords)
                # if func is expected_func and _args_equal(arguments, keywords, args, kwargs):

    # for (i, node) in enumerate(nodes.body):
    #     if hasattr(node, 'value') and isinstance(node.value, ast.Call)
    #         and hasattr(node.value.func, 'id') and node.value.func.id == 'foo'  # Here goes name of the function:
    #         i_expr = i
    #         break


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
