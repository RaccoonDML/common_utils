import os
import shutil

def setup_save_dir(save_dir, clean=True):
    if clean and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def gp2linux_path(path):
    ### r"\\files.miguocomics.co\内容空间\☆暗恋对象是只猫\03线稿\暗恋喵-003-线稿" => 
    item_list = [x for x in path.split('\\') if x != '' and x!='files.miguocomics.co']
    prefix = "/home/daimingliang/workspace/gongpan"
    return os.path.join(prefix, *item_list)