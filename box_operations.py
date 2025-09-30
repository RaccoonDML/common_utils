from loguru import logger
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
from torchvision.transforms.functional import to_tensor, to_pil_image
from .timer import change_str_color
import numpy as np
import cv2



def create_square_mask(image_size, square_size_ratio=0.9, blur_radius=10):
    mask = Image.new('L', image_size, 0)
    size_inner = int(min(image_size) * square_size_ratio)
    left = (image_size[0] - size_inner) // 2
    top = (image_size[1] - size_inner) // 2
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, left + size_inner, top + size_inner], fill=255)
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return blurred_mask



def create_mask(box, pad_list, blur_radius=10):
    x1, y1, x2, y2 = box
    pad_x1, pad_y1, pad_x2, pad_y2 = pad_list
    inner_x1 = pad_x1
    inner_y1 = pad_y1
    inner_x2 = pad_x1 + (x2 - x1 - pad_x2 - pad_x1)
    inner_y2 = pad_y1 + (y2 - y1 - pad_y2 - pad_y1)
    # 输入：mask 为二值化的 numpy 数组（0 和 255）
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cv2.rectangle(mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 255, -1)  # 绘制纯白矩形
    # 高斯模糊（sigma 控制模糊程度，与 radius 对应）
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_radius, sigmaY=blur_radius)
    mask = Image.fromarray(blurred_mask)
    return mask


def create_mask_by_2box(outer_box, inner_box, blur_radius=10):
    x1, y1, x2, y2 = outer_box
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    print(f'outer_box: {outer_box}, inner_box: {inner_box}')
    inner_x1 = x1_inner - x1
    inner_y1 = y1_inner - y1
    inner_x2 = x1_inner + (x2_inner - x1_inner) - x1
    inner_y2 = y1_inner + (y2_inner - y1_inner) - y1
    print(f'inner_x1: {inner_x1}, inner_y1: {inner_y1}, inner_x2: {inner_x2}, inner_y2: {inner_y2}')
    # 输入：mask 为二值化的 numpy 数组（0 和 255）
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cv2.rectangle(mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 255, -1)  # 绘制纯白矩形
    # 高斯模糊（sigma 控制模糊程度，与 radius 对应）
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_radius, sigmaY=blur_radius)
    mask = Image.fromarray(blurred_mask)
    return mask




def is_overlap(a, b):
    # 判断两个框是否相交
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

def merge_boxes(boxes):
    merged = []
    while len(boxes) > 0:
        current = boxes.pop(0)
        to_merge = [current]
        # 寻找所有相交的框
        for box in boxes[:]:
            if is_overlap(current, box):
                boxes.remove(box)
                to_merge.append(box)
        # 计算合并后的坐标
        x1 = min(b[0] for b in to_merge)
        y1 = min(b[1] for b in to_merge)
        x2 = max(b[2] for b in to_merge)
        y2 = max(b[3] for b in to_merge)
        merged.append([x1, y1, x2, y2])
    return merged


def combine_and_filter_head_boxes(head_boxes, filter_range, max_head_boxes=5):
    if len(head_boxes) > 1:
        head_boxes = merge_boxes(head_boxes)
    # filter out small and big boxes
    if len(head_boxes) > max_head_boxes:
        logger.warning(f'Too many head boxes: {len(head_boxes)}, filter out all')
        return []
    res = []
    for idx, box in enumerate(head_boxes):
        w, h = box[2] - box[0], box[3] - box[1]
        minwh = min(w, h)
        maxwh = max(w, h)
        if minwh > filter_range[0] and maxwh < filter_range[1]:
            res.append(box)
            logger.info(f'Keep box {idx}, w: {w}, h: {h}')
        else:
            logger.info(f'Filter box {idx}, w: {w}, h: {h}')
    return res





def pad_head_box(head_box, image_size, padi=50):
    x1, y1, x2, y2 = head_box
    img_w, img_h = image_size
    
    # Calculate center and current size
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    current_size = max(x2 - x1, y2 - y1) + 2 * padi
    
    # Calculate ideal square bounds
    half_size = current_size // 2
    ideal_x1, ideal_y1 = cx - half_size, cy - half_size
    ideal_x2, ideal_y2 = cx + half_size, cy + half_size
    
    # Clamp to image boundaries
    padded_x1 = max(0, ideal_x1)
    padded_y1 = max(0, ideal_y1)
    padded_x2 = min(img_w, ideal_x2)
    padded_y2 = min(img_h, ideal_y2)
    
    # Adjust to maintain square if possible
    width, height = padded_x2 - padded_x1, padded_y2 - padded_y1
    if width != height:
        size = min(width, height)
        padded_x2 = padded_x1 + size
        padded_y2 = padded_y1 + size
    
    padded_head_box = [padded_x1, padded_y1, padded_x2, padded_y2]
    pad_list = [x1-padded_x1, y1-padded_y1, padded_x2-x2, padded_y2-y2]
    
    return padded_head_box, pad_list


def resize_or_pad_to_1024(image_list, box, target_size=1024, max_content_size=960):
    """
    'box' is the box of the image to be processed, in the format of [x1, y1, x2, y2]
    'image_list' is a list of images to be processed, in the format of [image1, image2, ...]

    longside = max(box[2] - box[0], box[3] - box[1])
    if longside < max_content_size:
        pad box to 1024x1024 (ref function pad_head_box)
        crop images in image_list
        construct inpainting mask according to box and pad_list
        record key params for restore
    else:
        recursively call this function with larger max_content_size and target_size
        resize the return image to 1024x1024
        record key params for restore, such as scale, pad_list, original_size
    """
    img_w, img_h = image_list[0].size
    x1, y1, x2, y2 = box
    longside = max(x2 - x1, y2 - y1)
    
    if longside < max_content_size:
        logger.info(change_str_color('Content fits within max_content_size, pad to target_size', 'red'))
        # Case 1: Content fits within max_content_size, pad to target_size
        
        # Calculate target_size×target_size crop region centered on box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_size = target_size // 2
        
        # Calculate crop bounds centered on box
        crop_x1 = max(0, cx - half_size)
        crop_y1 = max(0, cy - half_size)
        crop_x2 = min(img_w, crop_x1 + target_size)
        crop_y2 = min(img_h, crop_y1 + target_size)
        crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
        assert crop_x2 - crop_x1 == target_size and crop_y2 - crop_y1 == target_size, 'hit boundary, crop_box: ' + str(crop_box)
        
        # Crop images to exactly target_size×target_size
        crop_image_list = []
        for image in image_list:
            cropped_image = image.crop(crop_box)
            crop_image_list.append(cropped_image)
        
        # Create mask for the head box area
        mask = create_mask_by_2box(crop_box, box)
        
        # Store minimal parameters for restoration
        params = {
            'method': 'pad',
            'crop_box': crop_box,
            'original_box': box
        }
        
    else:
        # Case 2: Content too large, scale down to fit target_size
        logger.info(change_str_color('Content too large, scale down to fit target_size', 'red'))
        
        # Calculate scale factor so that longside becomes target_size
        scale = max_content_size / longside  # 1024/1500 = 0.6826666666666667
        
        # Calculate crop region centered around the box
        crop_size = int(target_size / scale)
        box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        crop_x1 = int(max(0, box_center_x - crop_size / 2))
        crop_y1 = int(max(0, box_center_y - crop_size / 2))
        crop_x2 = int(min(img_w, crop_x1 + crop_size))
        crop_y2 = int(min(img_h, crop_y1 + crop_size))
        
        # Adjust if we hit boundaries
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = int(max(0, crop_x2 - crop_size))
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = int(max(0, crop_y2 - crop_size))
            
        crop_region = [crop_x1, crop_y1, crop_x2, crop_y2]
        
        # Crop and resize images
        crop_image_list = []
        for image in image_list:
            cropped_image = image.crop(crop_region)
            cropped_image = cropped_image.resize((target_size, target_size))
            crop_image_list.append(cropped_image)
        
        # Calculate box position in target_size coordinate system
        crop_scale = target_size / (crop_x2 - crop_x1)
        new_x1 = round((x1 - crop_x1) * crop_scale)
        new_y1 = round((y1 - crop_y1) * crop_scale)
        new_x2 = round((x2 - crop_x1) * crop_scale)
        new_y2 = round((y2 - crop_y1) * crop_scale)
        
        # Create inpainting mask
        mask = Image.new('L', (target_size, target_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([new_x1, new_y1, new_x2, new_y2], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Store minimal parameters for restoration
        params = {
            'method': 'scale',
            'original_box': box,
            'box_in_1024': [new_x1, new_y1, new_x2, new_y2]
        }
    
    return crop_image_list, mask, params


def restore_from_1024(image, restore_params):
    """
    restore image from 1024x1024 back to original size using stored parameters
    """
    method = restore_params['method']
    
    if method == 'pad':
        # Case 1: No scaling needed, just crop the box area
        x1, y1, x2, y2 = restore_params['original_box']
        crop_x1, crop_y1, crop_x2, crop_y2 = restore_params['crop_box']
        
        # Calculate box position in cropped coordinate system
        box_in_crop_x1 = x1 - crop_x1
        box_in_crop_y1 = y1 - crop_y1
        box_in_crop_x2 = x2 - crop_x1
        box_in_crop_y2 = y2 - crop_y1
        
        # Crop the box area directly (no resize needed)
        content_image = image.crop((box_in_crop_x1, box_in_crop_y1, box_in_crop_x2, box_in_crop_y2))
        
    elif method == 'scale':
        # Case 2: Image was cropped and scaled, need to restore back
        x1, y1, x2, y2 = restore_params['original_box']
        
        # Crop and resize to original box size
        content_image = image.crop(restore_params['box_in_1024'])
        content_image = content_image.resize((x2 - x1, y2 - y1))
    
    else:
        raise ValueError(f"Unknown restoration method: {method}")
    
    return content_image