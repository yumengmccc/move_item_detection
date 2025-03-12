import cv2
import numpy as np
import os


def optimized_motion_detection(video_path, result_path, scale=0.25, graphic=False):
    """
    优化版移动目标检测
    改进点：
    1. 使用KNN背景减法器更好处理动态背景
    2. 改进形态学处理流程
    3. 增加轮廓合并机制
    4. 添加目标筛选条件
    5. 自动填补缺失的文件索引
    """
    # 创建KNN背景减法器
    bg_subtractor = cv2.createBackgroundSubtractorKNN(
        history=100,
        dist2Threshold=2000,
        detectShadows=True
    )

    # 形态学操作核
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    os.makedirs(result_path, exist_ok=True)

    # 获取现有文件索引
    existing_indices = set()
    for filename in os.listdir(result_path):
        if filename.startswith('target_') and filename.endswith('.png'):
            num_str = filename[len('target_'):-4]
            if num_str.isdigit():
                existing_indices.add(int(num_str))

    # 创建索引生成器
    def index_generator():
        # 生成所有缺失的中间索引
        if existing_indices:
            max_idx = max(existing_indices)
            all_possible = set(range(max_idx + 1))
            available = sorted(all_possible - existing_indices)
            for idx in available:
                yield idx
            # 继续生成后续索引
            current = max_idx + 1
            while True:
                yield current
                current += 1
        else:
            # 没有现存文件时从0开始
            current = 0
            while True:
                yield current
                current += 1

    gen = index_generator()

    # 获取视频基本信息
    min_area = 2000  # 动态计算最小区域

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 1. 预处理
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        original = frame.copy()
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # 2. 背景建模
        fg_mask = bg_subtractor.apply(blurred)

        # 3. 后处理
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)

        if frame_count < 8:
            continue

        # 4. 目标检测
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 收集候选矩形
        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append([x, y, x + w, y + h])

        # 处理最终目标
        for rect in rects:
            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1

            # 排除细长型目标
            aspect_ratio = w / (h + 1e-5)
            if aspect_ratio < 1/3 or aspect_ratio > 3:
                continue

            # 保存目标区域
            target = original[y1:y2, x1:x2]
            if target.size > 0:
                current_idx = next(gen)
                cv2.imwrite(os.path.join(result_path, f'target_{current_idx}.png'), target)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示结果
        if graphic:
            cv2.imshow('Optimized Detection', frame)
            cv2.imshow('Processed Mask', fg_mask)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "./Camera/IMAG2024-06-24_12-16-33_04.MP4"
    optimized_motion_detection(
        video_path=video_path,
        result_path='./optimized_results',
        scale=0.25,
        graphic=True
    )
