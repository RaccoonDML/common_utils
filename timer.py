import time
import functools
from loguru import logger


def timeit(func):
    """
    A decorator that logs the execution time of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get the class name if this is a method
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            logger.info(change_str_color(f'{class_name}.{func.__name__} executed in {execution_time:.2f}s', 'green'))
        else:
            logger.info(change_str_color(f'{func.__name__} executed in {execution_time:.2f}s', 'green'))
        
        return result
    return wrapper 


def change_str_color(str, color):
    color_dict = {
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'purple': 35,
        'cyan': 36,
        'white': 37,
    }
    color = color_dict[color] if color in color_dict else color
    return f'\033[{color}m{str}\033[0m'