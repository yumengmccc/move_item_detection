# 2025/3/31/10:00 课堂任务：写好pytorch框架，完成分类

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from PIL import Image
from efficientnet_pytorch import EfficientNet

# 数据增强定义（保持不变）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义数据集类（处理不同来源的数据）
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_bird=True):
        self.classes = ['bird', 'plant']
        self.img_paths = []
        self.labels = []

        # 遍历目录加载图像路径
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    self.img_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(0 if is_bird else 1)  # 鸟类标签0，植物标签1

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 分别为鸟类图片和植物(草、木、树枝)图片
bird_dataset = CustomDataset('/datasets/birds', transform=train_transform, is_bird=True)
plant_dataset = CustomDataset('/datasets/plants', transform=train_transform, is_bird=False)

# 合并数据集并划分训练集
full_dataset = ConcatDataset([bird_dataset, plant_dataset])
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=full_dataset.labels)

# 创建训练集和验证集
train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

# 获取训练集标签
train_labels = [full_dataset.datasets[0].labels[i] if i < len(bird_dataset)
                else full_dataset.datasets[1].labels[i - len(bird_dataset)]
                for i in train_idx]
train_labels = torch.tensor(train_labels)

# 类平衡采样器
class_weights = 1. / torch.bincount(train_labels)
samples_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

# 创建数据加载器
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,  # 使用采样器
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# 模型定义（保持不变）
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


# 训练配置（保持不变）
model = BirdClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
