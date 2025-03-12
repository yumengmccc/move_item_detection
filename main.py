from PIL import Image
import numpy as np
import cv2
import os
from similarity_calc import similarity_calc

# Image 转换为 numpy数组
def trans_img(img, target_size):
    return img.resize(target_size)

# 提取视频的帧
def extract_frames(cap):
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    return frames, frame_count

target_size = (400, 300)

cap = cv2.VideoCapture('demo1.mp4')
frames, frames_count = extract_frames(cap)
print(frames_count)

# 保存每一帧
for i, frame in enumerate(frames):
    cv2.imwrite(f"./Demo/frame_{i}.jpg", frame)

# 读取每一帧
imgs = os.listdir(f'./Demo')

# 选择第[0]项作为基础项，后续图片和第[0]项计算相似度
base_img = trans_img(Image.open('./Demo/frame_0.jpg'), target_size)

for i, elem in enumerate(imgs[1:]):
    img = Image.open(os.path.join(f'./Demo', elem))
    img = trans_img(img, target_size)
    print(f'similarity between 0 and {i+1} : {similarity_calc(base_img, img)}')

forest_img = trans_img(Image.open('./forest.png'), target_size)
print(f'similarity 对照组 between 0 and Forest : {similarity_calc(base_img, forest_img)}')

abird_img = trans_img(Image.open('./a bird.png'), target_size)
print(f'similarity 对照组 between 0 and A bird : {similarity_calc(base_img, abird_img)}')