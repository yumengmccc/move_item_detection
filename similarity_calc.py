import numpy as np
from skimage.metrics import structural_similarity as ssim

# 计算图片相似度（后续主要改进部分）
def similarity_calc(img1, img2):
    """
    计算两张图片之间的相似度，使用结构相似性（SSIM）。

    参数:
        img1: PIL.Image 或 NumPy 数组，第一张图片
        img2: PIL.Image 或 NumPy 数组，第二张图片

    返回:
        similarity: float，结构相似性分数，范围 [0, 1]，越接近 1 相似度越高
    """
    # 将输入图片转换为 NumPy 数组
    np1 = np.array(img1)
    np2 = np.array(img2)
    # print(np1.dtype)
    # 确保两张图片的大小一致
    if np1.shape != np2.shape:
        raise ValueError("两张图片的大小必须相同！")

    # 如果是彩色图片，则转换为灰度图进行 SSIM 计算
    if len(np1.shape) == 3:  # 彩色图片具有 (height, width, channels)
        from skimage.color import rgb2gray
        np1 = rgb2gray(np1)
        np2 = rgb2gray(np2)

    # 计算结构相似性（SSIM）
    similarity, _ = ssim(np1, np2, full=True, data_range=255)
    return similarity
