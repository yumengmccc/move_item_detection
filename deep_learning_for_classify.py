import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# 数据增强定义
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, is_bird=True):
        self.classes = ['bird', 'plant']
        self.img_paths = []
        self.labels = []
        # print(os.listdir(root_dir))

        for img_name in os.listdir(root_dir):
            self.img_paths.append(os.path.join(root_dir, img_name))
            self.labels.append(0 if is_bird else 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        return image, self.labels[idx]


# 数据增强包装类
class TransformDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# 模型定义
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


if __name__ == '__main__':
    # 数据准备
    bird_dataset = CustomDataset('./datasets/birds', is_bird=True)
    plant_dataset = CustomDataset('./datasets/plants', is_bird=False)

    print(len(bird_dataset), len(plant_dataset))
    all_labels = bird_dataset.labels + plant_dataset.labels

    # 数据集划分
    train_idx, val_idx = train_test_split(
        range(len(all_labels)),
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    # 创建最终数据集
    full_dataset = ConcatDataset([bird_dataset, plant_dataset])
    train_dataset = TransformDataset(Subset(full_dataset, train_idx), train_transform)
    val_dataset = TransformDataset(Subset(full_dataset, val_idx), val_transform)

    # 类平衡采样器
    train_labels = [all_labels[i] for i in train_idx]
    class_weights = 1. / torch.bincount(torch.tensor(train_labels))
    samples_weights = class_weights[torch.tensor(train_labels)]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    # 数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 训练配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BirdClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练循环
    num_epochs = 7
    best_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_dataset)
        train_acc = correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct / val_total

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # 打印日志
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')
        print('-' * 50)

    print(f'Training completed. Best validation accuracy: {best_acc:.4f}')

    # model = BirdClassifier()
    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()