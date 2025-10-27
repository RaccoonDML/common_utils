import os
import shutil

def setup_save_dir(save_dir, clean=True):
    if clean and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
