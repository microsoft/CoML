from coml.context import nearest_user_frame, reproduce_function_parameters

def test_reproduce_function_parameters():
    assert reproduce_function_parameters(1, 2, 3) == "1, 2, 3"

    a = {"a": 1}
    b = {"b": 2}
    assert reproduce_function_parameters(x=a, y=b) == "x=a, y=b"
