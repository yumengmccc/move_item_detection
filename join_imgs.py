import os
import shutil

root_path = './datasets/birds/CUB_200_2011/CUB_200_2011/images'
target_root = './datasets/birds'  # 直接存放到这个目录

# 创建目标目录（如果不存在）
os.makedirs(target_root, exist_ok=True)

for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.lower().endswith('.jpg'):
            src_path = os.path.join(root, file)

            # 直接使用文件名作为目标路径（去掉了子目录部分）
            dest_path = os.path.join(target_root, file)

            # 处理可能出现的重名文件（可选）
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(file)
                dest_path = os.path.join(target_root, f"{name}_{counter}{ext}")
                counter += 1

            shutil.copy(src_path, dest_path)
            print(f'Copied: {src_path} -> {dest_path}')

print("所有图片已扁平化复制完成！")
