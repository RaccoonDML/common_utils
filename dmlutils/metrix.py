from PIL import Image
import numpy as np
import cv2

def chamfer_distance_pil(img1: Image.Image, img2: Image.Image) -> float:
    """
    计算两张 PIL Image 线稿图像的 Chamfer Distance。
    
    参数:
        img1, img2: PIL.Image
            输入的线稿或边缘图像，任意模式 (RGB/L/LA)
    
    返回:
        float: Chamfer Distance（越小越相似）
    """

    # --- 转灰度 ---
    img1_gray = img1.convert("L")
    img2_gray = img2.convert("L")

    # --- 调整到相同大小 ---
    if img1_gray.size != img2_gray.size:
        img2_gray = img2_gray.resize(img1_gray.size, Image.BILINEAR)

    # --- 转 numpy 数组 ---
    arr1 = np.array(img1_gray, dtype=np.uint8)
    arr2 = np.array(img2_gray, dtype=np.uint8)

    # --- 二值化，前景为1 ---
    # 自动判断线条是黑线还是白线：线条应为1，背景为0
    mean1, mean2 = arr1.mean(), arr2.mean()
    invert1 = mean1 > 127  # 如果背景更亮，说明线条是黑的
    invert2 = mean2 > 127

    bin1 = (arr1 < 128).astype(np.uint8) if invert1 else (arr1 > 128).astype(np.uint8)
    bin2 = (arr2 < 128).astype(np.uint8) if invert2 else (arr2 > 128).astype(np.uint8)

    # --- 距离变换 ---
    dist1 = cv2.distanceTransform(1 - bin1, cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform(1 - bin2, cv2.DIST_L2, 3)

    # --- 双向 Chamfer 距离 ---
    d1 = np.mean(dist1[bin2 == 1]) if np.any(bin2 == 1) else 0
    d2 = np.mean(dist2[bin1 == 1]) if np.any(bin1 == 1) else 0

    chamfer = (d1 + d2) / 2.0
    return float(chamfer)
