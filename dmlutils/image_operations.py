# %%
# 工具函数v3 ===========================
"""
Usage1:
x = rgba_to_whitebg(Image.open(x_path))
original_size = x.size
box = get_box(x)
x, pad_list = crop_pad_resize(x, box, size=1024, pad_value=255)
y = ops(x)
y = paste_image_to_origin_image(y, pad_list, box, original_size, mode='RGB')

Usage2:
x = rgba_to_whitebg(Image.open(x_path))
original_size = x.size
box = get_box(x)
x = crop_image(x, box)
y = ops(x)
y = paste_image_to_origin_image_without_unpad(y, box, original_size, mode='RGB')
"""
# =======================
# ==========================
# 系统与文件操作
# ==========================
import os
import shutil
import time
import copy
import math
import random
from io import BytesIO

# ==========================
# 数值计算与数组操作
# ==========================
import numpy as np

# ==========================
# 图像处理与计算机视觉
# ==========================
from PIL import Image, ImageOps
import cv2
import cairosvg
from skimage import morphology

# ==========================
# PyTorch & torchvision
# ==========================
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

# ==========================
# 进度条与日志
# ==========================
from tqdm import tqdm
from loguru import logger




def normalize_image(image):
    image_array = np.array(image)
    if image_array.max() > image_array.min():
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    return Image.fromarray((image_array * 255).astype(np.uint8))


def binarize_image(image, thresh=128):
    image = image.convert('L')
    arr = np.array(image)
    binary = np.where(arr > thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def pil_to_bytes(img: Image, format: str = 'PNG') -> bytes:
    "for diffusers dataset"
    byte_arr = BytesIO()
    img.save(byte_arr, format=format)
    byte_data = byte_arr.getvalue()
    return byte_data

def pil_to_bytesio(image):
    """Convert PIL Image to BytesIO"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

def bytesio_to_pil(buffer):
    """Convert BytesIO to PIL Image"""
    buffer.seek(0)
    return Image.open(buffer)


def l_rgb_to_rgba(l_image):
    l_image = ImageOps.invert(normalize_image(l_image)).convert('L')
    black_image = Image.new('RGBA', l_image.size, (0,0,0,0))
    black_image.putalpha(l_image)
    return black_image


def rgba_to_whitebg(img, bg=(255, 255, 255, 255)):
    img = img.convert('RGBA')
    white_bg = Image.new('RGBA', img.size, bg)
    result = Image.alpha_composite(white_bg, img)
    return result.convert('RGB')


def image_grid(imgs, rows, cols=None):
    # assert len(imgs) == rows * cols
    if cols is None:
        cols = math.ceil(len(imgs)/rows)
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def pad_short_side_to_size(img: Image.Image, size: int, color=(255,255,255)):
    """中心 pad"""
    w, h = img.size
    new_w = max(w, size)
    new_h = max(h, size)
    if (new_w, new_h) == (w, h):
        return img

    mode = img.mode
    if mode in ("RGBA", "LA") and (len(color) == 3):
        bg_color = tuple(list(color) + [255])
    elif mode == "L" and isinstance(color, tuple):
        bg_color = color[0]
    else:
        bg_color = color

    bg = Image.new(mode, (new_w, new_h), bg_color)
    offset = ((new_w - w) // 2, (new_h - h) // 2)
    bg.paste(img, offset)
    return bg


def resize_short_side_to_size(img: Image.Image, size: int, resample=Image.LANCZOS):
    """将短边缩放到指定大小，保持比例"""
    w, h = img.size
    min_size = min(w, h)
    if min_size <= size:
        return img

    if w < h:
        scale = size / w
        new_w = size
        new_h = math.ceil(h * scale)
    else:
        scale = size / h
        new_h = size
        new_w = math.ceil(w * scale)

    new_w = max(new_w, size)
    new_h = max(new_h, size)

    return img.resize((new_w, new_h), resample=resample)


def crop_image(image_path, box):
    if isinstance(image_path, str):
        image = rgba_to_whitebg(Image.open(image_path))
    elif isinstance(image_path, Image.Image):
        image = rgba_to_whitebg(image_path)
    else:
        raise TypeError("image_path must be either a string or a PIL Image object")
    image = image.crop(box)
    return image



def crop_mod_resize(image, box, size, pad_value, mod_value=32):
    long_side_size = size
    x1, y1, x2, y2 = box
    image = rgba_to_whitebg(image)
    
    # Crop the image according to box
    image = image.crop(box)
    image_np = np.array(image)
    
    # Get original dimensions
    h, w = image_np.shape[:2]
    max_side = max(h, w)
    
    if max_side > long_side_size:
        # Resize if longest side exceeds target size
        scale = long_side_size / max_side
        new_h = int(h * scale)
        new_w = int(w * scale)
        image = Image.fromarray(image_np)
        image = image.resize((new_w, new_h))
        image_np = np.array(image)
        pad_list = [0, 0, 0, 0]
    else:
        # Pad to long_side_size if needed
        if h < w:
            pad_top = 0
            pad_bottom = 0
            pad_left = (long_side_size - w) // 2
            pad_right = long_side_size - w - pad_left
        else:
            pad_top = (long_side_size - h) // 2
            pad_bottom = long_side_size - h - pad_top
            pad_left = 0
            pad_right = 0
            
        image_np = np.pad(image_np,
                         ((pad_top, pad_bottom),
                          (pad_left, pad_right),
                          (0, 0)),
                         mode='constant',
                         constant_values=pad_value)
        pad_list = [pad_left, pad_top, pad_right, pad_bottom]
    
    # Calculate padding needed to make dimensions divisible by mod_value
    h, w = image_np.shape[:2]
    pad_h = (mod_value - h % mod_value) % mod_value
    pad_w = (mod_value - w % mod_value) % mod_value
    
    # Pad if necessary
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        image_np = np.pad(image_np,
                         ((pad_top, pad_bottom),
                          (pad_left, pad_right),
                          (0, 0)),
                         mode='constant',
                         constant_values=pad_value)
        pad_list = [x+y for x,y in zip(pad_list, [pad_left, pad_top, pad_right, pad_bottom])]
    
    return Image.fromarray(image_np), pad_list


def crop_pad_resize(image, box, size, pad_value):
    x1, y1, x2, y2 = box
    is_rgba = image.mode == 'RGBA'
    if is_rgba:
        alpha_channel = image.split()[-1].crop(box)
    else:
        alpha_channel = None
    rgb_image = rgba_to_whitebg(image).crop(box)
    
    # Process RGB image
    def process_channel(img_array, pad_val):
        h, w = img_array.shape[:2]
        max_side = max(h, w)
        
        # Resize if needed
        if max_side > size:
            scale = size / max_side
            new_h, new_w = int(h * scale), int(w * scale)
            img = Image.fromarray(img_array)
            img_array = np.array(img.resize((new_w, new_h)))
            h, w = img_array.shape[:2]

        # Pad to square
        if h > w:
            pad_left, pad_right = (h - w) // 2, h - w - (h - w) // 2
            pad_top = pad_bottom = 0
        else:
            pad_top, pad_bottom = (w - h) // 2, w - h - (w - h) // 2
            pad_left = pad_right = 0
        
        # Apply square padding
        if len(img_array.shape) == 3:  # RGB
            img_array = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                             mode='constant', constant_values=pad_val)
        else:  # Grayscale (Alpha)
            img_array = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                             mode='constant', constant_values=pad_val)
        
        square_pad_list = [pad_left, pad_top, pad_right, pad_bottom]

        # Pad to target size if needed
        h, w = img_array.shape[:2]
        if h < size:
            pad_h = (size - h) // 2
            pad_h_extra = size - h - 2*pad_h
            pad_w = (size - w) // 2 
            pad_w_extra = size - w - 2*pad_w
            
            if len(img_array.shape) == 3:  # RGB
                img_array = np.pad(img_array, ((pad_h, pad_h+pad_h_extra), (pad_w, pad_w+pad_w_extra), (0, 0)), 
                                 mode='constant', constant_values=pad_val)
            else:  # Grayscale (Alpha)
                img_array = np.pad(img_array, ((pad_h, pad_h+pad_h_extra), (pad_w, pad_w+pad_w_extra)), 
                                 mode='constant', constant_values=pad_val)
            
            square_pad_list = [x+y for x,y in zip(square_pad_list, [pad_w, pad_h, pad_w+pad_w_extra, pad_h+pad_h_extra])]
        
        return img_array, square_pad_list

    # Process RGB channels
    rgb_np, square_pad_list = process_channel(np.array(rgb_image), pad_value)
    
    # Process Alpha channel if exists
    if is_rgba:
        alpha_pad_value = 0 if pad_value == 255 else 255
        alpha_np, _ = process_channel(np.array(alpha_channel), alpha_pad_value)
        # Combine RGB and Alpha
        result_np = np.dstack((rgb_np, alpha_np))
        return Image.fromarray(result_np, mode='RGBA'), square_pad_list
    else:
        return Image.fromarray(rgb_np), square_pad_list


def unpad_image(image, square_pad_list):
    """Removes padding according to square_pad_list [x1 y1 x2 y2]"""
    pad_left, pad_top, pad_right, pad_bottom = square_pad_list
    w, h = image.size
    return image.crop((pad_left, pad_top, w - pad_right, h - pad_bottom))


def get_box(mask_path):
    if isinstance(mask_path, str):
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask_path, Image.Image):
        image = np.array(mask_path)
    elif isinstance(mask_path, np.ndarray):
        image = mask_path
    else:
        raise TypeError("mask_path must be either a string or a PIL Image object or a numpy array")
    a0 = np.mean(255-image, axis=0)
    a1 = np.mean(255-image, axis=1)
    x = np.where(a0>0)[0]
    y = np.where(a1>0)[0]
    return  x[0], y[0], x[-1], y[-1]


def paste_image_to_origin_image(overlay_shade, pad_list, box, original_size, mode='RGB', bg=(255,255,255), background_image=None):
    overlay_shade = unpad_image(overlay_shade, pad_list)
    x1, y1, x2, y2 = box
    box_original_size = (x2 - x1, y2 - y1)
    overlay_shade = overlay_shade.resize(box_original_size)
    if background_image is None:
        if mode == 'RGBA':
            transparent_image = Image.new('RGBA', original_size, (0, 0, 0, 0))
        else:
            transparent_image = Image.new('RGB', original_size, bg)
    else:
        transparent_image = background_image.copy()

    transparent_image.paste(overlay_shade, (x1, y1))
    return transparent_image


def paste_image_to_origin_image_without_unpad(cropped_image, box, original_size, mode='RGB', bg=(255,255,255), background_image=None):
    x1, y1, x2, y2 = box
    box_original_size = (x2 - x1, y2 - y1)
    cropped_image = cropped_image.resize(box_original_size)
    if background_image is None:
        if mode == 'RGBA':
            transparent_image = Image.new('RGBA', original_size, (0, 0, 0, 0))
        else:
            transparent_image = Image.new('RGB', original_size, bg)
    else:
        transparent_image = background_image.copy()

    transparent_image.paste(cropped_image, (x1, y1))
    return transparent_image

def zpdd_multiply(stage, overlay):
    return stage * overlay

def xxjd_lineardodge(stage, overlay):
    return torch.clamp(stage + overlay, 0, 1)