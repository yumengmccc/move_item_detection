import os
import shutil

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm


# 定义与训练时相同的模型结构
class BirdClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b3')
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        features = self.base.extract_features(x)
        return self.classifier(nn.functional.adaptive_avg_pool2d(features, 1).flatten(1))


# 加载训练好的模型
def load_model(model_path, device):
    model = BirdClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    return model.to(device)


# 图像预处理转换
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 递归查找所有图片文件
def find_images(root_dir):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths


def predict_images(model, transform, device, root_dir='./result'):
    # 获取所有图片路径
    image_paths = find_images(root_dir)
    print(f"Found {len(image_paths)} images to predict")

    results = []
    for img_path in tqdm(image_paths):
        try:
            # 加载和预处理图像
            image = Image.open(img_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)

            # 执行预测
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.softmax(outputs, dim=1)

            # 获取预测结果
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][pred_class].item()
            class_name = 'bird' if pred_class == 0 else 'plant'

            results.append({
                'path': img_path,
                'class': class_name,
                'confidence': f"{confidence:.4f}"
            })

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

    return results


if __name__ == '__main__':
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = load_model('best_model.pth', device)

    # 执行预测
    predictions = predict_images(model, val_transform, device)

    # 打印结果
    print("\nPrediction Results:")
    idx = 0
    bird_idx = 0
    for result in predictions:
        print(f"Image: {result['path']}")
        print(f"  Predicted: {result['class']} (confidence: {result['confidence']})")
        src_path = result['path']
        dest_path = f'./final_result/all_img/all_figure_{idx}.png'
        idx += 1
        shutil.copy(src_path, dest_path)
        if result['class'] == 'bird' and float(result['confidence']) > 0.9:
            dest_path = f'./final_result/birds_img/bird_figure_{bird_idx}.png'
            bird_idx += 1
            shutil.copy(src_path, dest_path)
            print(f'Copied: {src_path} -> {dest_path}')
        print("-" * 60)