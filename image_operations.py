# %%
# 工具函数v3 ===========================
"""
Usage:
original_lc_image = Image.open(lc_path)
original_size = original_lc_image.size
box = get_box(lc_path)
lc_image, pad_list = crop_pad_resize(lc_path, box, size=1024, pad_value=255)
shade = get_shade(lc_image)
shade_final = paste_image_to_origin_image(shade, pad_list, box, original_size, mode='RGB')
"""
# =======================
import os
import cv2
import copy
import math
import torch
import time
import shutil
import numpy as np
from PIL import Image, ImageOps
from skimage import morphology
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm
from loguru import logger
from io import BytesIO
import random
import cairosvg
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY


# blacklevel 黑色阈值    turdsize 过滤污点力度   alphamax 最大角度   opttolerance 曲线简化容忍度
def mask_to_svg(img: Image.Image, blacklevel=0.8):
    """
    usage:
    import cairosvg
    from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY

    img = Image.open(img_path)
    svg_content = mask_to_svg(img)
    with open('output.svg', 'w') as f:
        f.write(svg_content.decode('utf-8'))

    mask = ImageOps.invert(mask).convert('L') # white bg, black line(object)
    svg = mask_to_svg(mask)
    png = Image.open(BytesIO(cairosvg.svg2png(bytestring=svg,output_width=boxw,output_height=boxh)))
    """
    bm = Bitmap(img, blacklevel=blacklevel)  # bigger blacklevel, more object, white bg, black line(object)
    plist = bm.trace(
        turdsize=2, turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=1.0, opticurve=True, opttolerance=0.2)
    parts = [f"M{c.start_point.x},{c.start_point.y}" + 
            "".join(f"L{s.c.x},{s.c.y}L{s.end_point.x},{s.end_point.y}" if s.is_corner 
                   else f"C{s.c1.x},{s.c1.y} {s.c2.x},{s.c2.y} {s.end_point.x},{s.end_point.y}"
                   for s in c.segments) + "z" 
            for c in plist]
    svg_content = f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="{img.width}" height="{img.height}" viewBox="0 0 {img.width} {img.height}"><path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/></svg>'''
    return svg_content.encode('utf-8')



def image_grid(imgs, rows, cols=None):
    # assert len(imgs) == rows * cols
    if cols is None:
        cols = math.ceil(len(imgs)/rows)
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


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


def rgba_to_whitebg(img, bg=(255, 255, 255, 255)):
    img = img.convert('RGBA')
    white_bg = Image.new('RGBA', img.size, bg)
    result = Image.alpha_composite(white_bg, img)
    return result.convert('RGB')


# use_8_neighbors: True使用八邻域, False使用四邻域
def get_neighbors(use_8_neighbors=False):
    if use_8_neighbors:
        return np.array([1, -1, 0, 0, 1, -1, 1, -1]).reshape(1,-1), \
                np.array([0, 0, 1, -1, 1, -1, -1, 1]).reshape(1,-1), 8
    else:
        return np.array([1, -1, 0, 0]).reshape(1,-1), \
                np.array([0, 0, 1, -1]).reshape(1,-1), 4


def label_propagation(input_img, unknown_mask, end_lines, fill_endlines=True, max_iter=500, use_maxvalue_fill_endlines=False):
    res_img  = input_img.copy()
    unknown_mask = unknown_mask.copy()
    height, width = input_img.shape[:2]
    assert unknown_mask.shape == input_img.shape[:2], unknown_mask.shape
    assert end_lines.shape == input_img.shape[:2], unknown_mask.shape
    
    dx, dy, n_neighbors = get_neighbors(use_8_neighbors=False)  # 默认使用四邻域（使用八邻域有问题）
    kernel = np.ones((3,3),np.uint8)

    for iter_index in range(1, max_iter):
        # logger.debug(f'iter: {iter_index}')
        # Perform dilation operation
        dilated_mask = cv2.dilate(unknown_mask, kernel, iterations=1)  # h w

        # Find edge pixels by comparing dilated mask with the original mask
        edge_pixels = dilated_mask - unknown_mask   # h w
        edge_pixels_coordinates = np.argwhere(edge_pixels > 0)  # n 2
        repeat_edge_pixels_coordinates = edge_pixels_coordinates.repeat(n_neighbors, axis=0)  # n*neighbors 2

        # Use numpy broadcasting instead of for loop
        # dx: (1,neighbors), repeat_edge_pixels_coordinates[:,1]: (n,) -> (n,neighbors)
        nx = dx + repeat_edge_pixels_coordinates[:,1].reshape((-1,n_neighbors))  # (n,neighbors)
        nx = nx.reshape(-1)  # (n*neighbors,)
        # dy: (1,neighbors), repeat_edge_pixels_coordinates[:,0]: (n,) -> (n,neighbors) 
        ny = dy + repeat_edge_pixels_coordinates[:,0].reshape((-1,n_neighbors))  # (n,neighbors)
        ny = ny.reshape(-1)  # (n*neighbors,)
        # Boolean indexing to filter nx and ny based on the first four conditions
        filtered_indices = np.where((0 <= nx) & (nx < width) & (0 <= ny) & (ny < height))[0] # N: pre-unknown-index
        # Apply the additional conditions to the filtered indices
        valid_thin    = (end_lines[ny[filtered_indices], nx[filtered_indices]] == 0)
        valid_unknown = (unknown_mask[ny[filtered_indices], nx[filtered_indices]] != 0)
        valid_indices = filtered_indices[valid_thin & valid_unknown] # M: unknown-index
        
        if valid_indices.shape[0] > 0:
            valid_nx = nx[valid_indices] 
            valid_ny = ny[valid_indices]
            unknown_mask[valid_ny, valid_nx] = 0
            newedge_x = repeat_edge_pixels_coordinates[valid_indices][:, 1]
            newedge_y = repeat_edge_pixels_coordinates[valid_indices][:, 0]
            res_img[valid_ny, valid_nx] = res_img[newedge_y, newedge_x] 
        else:
            break

    if fill_endlines:
        # 获取所有端点像素的位置
        end_points = np.argwhere(end_lines > 0)  # (N, 2) array of [y, x] coordinates
        if use_maxvalue_fill_endlines:
            values = cv2.dilate(res_img, np.ones((3,3), np.float32), iterations=1)
        else:
            values = cv2.erode(res_img, np.ones((3,3), np.float32), iterations=1)
        res_img[end_points[:, 0], end_points[:, 1]] = values[end_points[:, 0], end_points[:, 1]]
        unknown_mask[end_points[:, 0], end_points[:, 1]] = 0

    return unknown_mask, res_img


def close_line_gaps(base_image, lineart_image, dilation_iter=1, line_threshold=50, use_maxvalue_fill_endlines=False):
    lines_array = np.array(lineart_image)
    unknown_mask = ((lines_array > line_threshold)*255).astype(np.uint8)
    endpoints = morphology.skeletonize((lines_array > 128))

    if dilation_iter > 0:
        # 使用十字形核，减少对角线方向的膨胀
        cross_kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)
        dilated_mask = cv2.dilate(unknown_mask, cross_kernel, iterations=dilation_iter)
    else:
        dilated_mask = unknown_mask

    remain_unknown_mask, filled_image = label_propagation(np.array(base_image), copy.deepcopy(dilated_mask), copy.deepcopy(endpoints), use_maxvalue_fill_endlines=use_maxvalue_fill_endlines)
    # 填充剩余的未知区域
    _, filled_image = label_propagation(filled_image, copy.deepcopy(remain_unknown_mask), np.zeros_like(remain_unknown_mask), use_maxvalue_fill_endlines=use_maxvalue_fill_endlines)

    return to_pil_image(filled_image), to_pil_image(dilated_mask), to_pil_image((endpoints*255).astype(np.uint8))
    # return to_pil_image(filled_image)


def inpaint_line_gaps(base_image: Image.Image, lineart_image: Image.Image, dilation_iter: int = 1) -> Image.Image:
    # Convert lineart to mask
    lines_array = np.array(lineart_image)
    unknown_mask = ((lines_array > 0)*255).astype(np.uint8)
    
    # Dilate mask if specified
    if dilation_iter > 0:
        cross_kernel = np.array([
            [0, 1, 0],
            [1, 1, 1], 
            [0, 1, 0]
        ], dtype=np.uint8)
        unknown_mask = cv2.dilate(unknown_mask, cross_kernel, iterations=dilation_iter)

    # Convert base image to numpy array
    base_array = np.array(base_image)
    
    # Apply inpainting
    filled_image = cv2.inpaint(base_array, unknown_mask, 5, cv2.INPAINT_TELEA)
    
    return Image.fromarray(filled_image)


# %%
# Unet: lc, lcshade -> shade
# ==============
def zpdd_multiply(stage, overlay):
    return stage * overlay

def xxjd_lineardodge(stage, overlay):
    return torch.clamp(stage + overlay, 0, 1)


def pil_to_buffer(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer