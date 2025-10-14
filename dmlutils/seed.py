import random
import torch
import numpy as np

def set_seed(seed):
    # 控制所有可能的随机源
    random.seed(seed)        # Python内置
    np.random.seed(seed)     # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # 关闭cuDNN非确定性算法
    torch.backends.cudnn.benchmark = False     # 关闭自动优化
