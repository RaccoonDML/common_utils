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



def normalize_image(image):
    image_array = np.array(image)
    if image_array.max() > image_array.min():
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    return Image.fromarray((image_array * 255).astype(np.uint8))



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