from typing import Dict
from PIL import Image
from psd_tools.api.psd_image import PSDImage
from psd_tools.api.layers import PixelLayer, Group

"""
usage:
from dmlutils.psd_operations import save_image_dict_to_psd, extract_psd_layer   
input_psd_path = "/home/daimingliang/workspace/Lineart/dev/大小姐-005-5-分镜.psd"
image_dict = extract_psd_layer(input_psd_path, "精草")
psd = save_image_dict_to_psd(image_dict, "线稿-模型输出", output_path="output.psd")
"""


def save_image_dict_to_psd(image_dict: Dict[str, Image.Image], group_name: str, output_path: str=None) -> PSDImage:
    width, height = list(image_dict.values())[0].size
    psd = PSDImage.new(mode="RGBA", size=(width, height), color=(255, 255, 255, 0))
    layer_list = []
    for layer_name, img in image_dict.items():
        # 使用 PixelLayer.frompil 创建像素图层
        print(layer_name, img.size)
        pixel_layer = PixelLayer.frompil(
            img,
            psd,  # 父 PSD
            name=str(layer_name)
        )
        pixel_layer.name = str(layer_name)
        layer_list.append(pixel_layer)

    group_layer = Group.group_layers(layer_list, name=group_name)
    psd.append(group_layer)
    if output_path:
        psd.save(output_path, encoding="utf-8")
    return psd



def extract_psd_layer(psd_path: str, group_name: str) -> Dict[str, Image.Image]:
    """
    从PSD文件中提取指定图层组内的所有图层，
    保留每个图层在画布中的原始大小和位置。
    
    Args:
        psd_path (str): PSD文件路径
        group_name (str): 目标图层组名称
        
    Returns:
        Dict[str, Image.Image]: {图层名: 带原始位置的完整画布 PIL Image}
    """
    psd = PSDImage.open(psd_path)
    canvas_size = (psd.width, psd.height)

    # 查找目标组
    target_group = next((layer for layer in psd if layer.is_group() and layer.name == group_name), None)
    if target_group is None:
        raise ValueError(f"未找到名为 '{group_name}' 的图层组。")

    result: Dict[str, Image.Image] = {}

    for layer in target_group:
        if layer.is_group():  # 跳过子组（可选：可递归）
            continue

        img = layer.composite()
        if img is None:
            continue

        x1, y1, x2, y2 = layer.bbox
        # 在整个PSD画布上创建透明底
        full_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        # 粘贴到正确位置
        full_canvas.paste(img, (x1, y1), img if img.mode == "RGBA" else None)
        result[layer.name] = full_canvas

    if not result:
        raise ValueError(f"图层组 '{group_name}' 中没有有效图层。")

    return result
