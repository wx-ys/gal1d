
import inspect




def get_required_params_num(func):
    require_num = 0
    for name,param in inspect.signature(func).parameters.items():
        if param.default == inspect.Parameter.empty:
            require_num = require_num +1
    return require_num
    