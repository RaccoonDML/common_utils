"""
MyPackage: a toolkit for image processing
"""

__version__ = "2.0.0"
__author__ = "Beta"
__email__ = "2684813584@qq.com"


# 导出核心接口
from .file import setup_save_dir, gp2linux_path
from .seed import set_seed
from .timer import timeit, change_str_color

__all__ = ["setup_save_dir", "set_seed", "timeit", "change_str_color", "gp2linux_path"]