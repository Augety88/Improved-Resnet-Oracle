import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import json
import csv
from torchvision import transforms
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# ===================== 配置部分 =====================
class Config:
    # 路径配置
    DATASET_ROOT = "/media/wn/实验备份1/（最终）甲骨文拓片转手写_增广+甲骨文手写体_增广"
    MODEL_SAVE_ROOT = "./saved_models"
    RESULT_CSV_PATH = "./ablation_results.csv"
    
    # 硬件/模型配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = (224, 224)
    NUM_CLASSES = None
    
    # 训练配置
    EPOCHS = 30
    LR_CANDIDATES = [5e-4]
    WEIGHT_DECAY_CANDIDATES = [1e-4]
    BATCH_SIZE_CANDIDATES = [64]
    
    # 消融实验配置
    ABLATION_CONFIGS = [
        {"use_cbam": True, "use_edge_enhance": True, "name": "FullImproved"},
        {"use_cbam": False, "use_edge_enhance": True, "name": "NoCBAM"},
        {"use_cbam": True, "use_edge_enhance": False, "name": "NoEdgeEnhance"},
        {"use_cbam": False, "use_edge_enhance": False, "name": "OriginalResNet18"}
    ]

config = Config()

# ===================== CBAM 注意力模块 =====================
class CBAM(nn.Module):
    """通道 + 空间注意力模块，聚焦古文字笔画特征"""
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

# ===================== 残差块 =====================
class BasicBlock(nn.Module):
    """改进的残差块，可选嵌入 CBAM"""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_cbam:
            out = self.cbam(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ===================== OracleResNet18 模型 =====================
class OracleResNet18(nn.Module):
    """适配古文字的改进 ResNet18"""
    def __init__(self, num_classes, use_cbam=True, use_edge_enhance=True):
        super(OracleResNet18, self).__init__()
        self.in_channels = 64
        self.use_edge_enhance = use_edge_enhance
        
        # 边缘增强：替换 7×7 卷积为 3 个 3×3 卷积
        if self.use_edge_enhance:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:
            # 原始 ResNet18 浅层
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, use_cbam=False)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, use_cbam=False)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, use_cbam=False)
        
        # 输出层优化：Dropout 抑制小样本过拟合
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1, use_cbam=True):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, use_cbam))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_cbam=use_cbam))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_edge_enhance:
            x = self.conv1(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ===================== 评估指标计算 =====================
def calculate_metrics(outputs, labels, num_classes):
    """
    计算多分类任务的核心指标：
    返回：acc, f1, auc, top5_acc
    """
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    
    # 准确率
    acc = torch.eq(preds, labels).sum().item() / len(labels)
    
    # Top5 准确率
    _, top5 = torch.topk(probs, 5, dim=1)
    top5_acc = sum([labels.cpu().numpy()[i] in top5.cpu().numpy()[i] for i in range(len(labels))]) / len(labels)
    
    # F1 分数（加权平均）
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
    
    # AUC（多分类 OVR）
    try:
        labels_one_hot = label_binarize(labels.cpu().numpy(), classes=range(num_classes))
        auc = roc_auc_score(labels_one_hot, probs.cpu().numpy(), multi_class='ovr')
    except Exception:
        auc = 0.0
    
    return acc, f1, auc, top5_acc

# ===================== 数据集类 =====================
class OracleDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_idx=None, transform=None, is_tuopian=False):
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_idx = label_to_idx  # 添加标签到索引的映射
        self.transform = transform
        self.is_tuopian = is_tuopian

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label_str = self.labels[idx]
        # 如果有映射关系，将字符串标签转为数字索引
        if self.label_to_idx:
            label = self.label_to_idx[label_str]
        else:
            label = label_str
        if self.transform:
            img = self.transform(img)
        return img, label

# ===================== 超参数随机选择 =====================
def random_hyperparams():
    lr = random.choice(config.LR_CANDIDATES)
    weight_decay = random.choice(config.WEIGHT_DECAY_CANDIDATES)
    batch_size = random.choice(config.BATCH_SIZE_CANDIDATES)
    return lr, weight_decay, batch_size

# ===================== 数据增强函数 =====================
def oracle_image_augment(img_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

# ===================== 主训练函数 =====================
def train_model():
    print(f"Using device: {config.DEVICE}")
    os.makedirs(config.MODEL_SAVE_ROOT, exist_ok=True)

    # 1. 加载并合并数据集、划分
    all_image_paths = []
    all_labels = []
    train_root = os.path.join(config.DATASET_ROOT, "train")
    test_root = os.path.join(config.DATASET_ROOT, "test")

    # 获取所有图片路径和标签
    for root, _, files in os.walk(train_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_image_paths.append(os.path.join(root, file))
                all_labels.append(root.split('/')[-1])  # 假设文件夹名为标签

    for root, _, files in os.walk(test_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_image_paths.append(os.path.join(root, file))
                all_labels.append(root.split('/')[-1])

    config.NUM_CLASSES = len(set(all_labels))
    # 确保 idx_to_class 的键是整数，值是类别名
    idx_to_class = {i: k for i, k in enumerate(set(all_labels))}
    class_to_idx = {k: i for i, k in idx_to_class.items()}
    print(f"Number of classes: {config.NUM_CLASSES}")
    print("Sample mapping:", dict(list(idx_to_class.items())[:10]))

    # 保存映射关系，方便推理时使用
    os.makedirs(config.MODEL_SAVE_ROOT, exist_ok=True)
    mapping_path = os.path.join(config.MODEL_SAVE_ROOT, "classnames.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, ensure_ascii=False, indent=2)
    print("Saved class mapping:", mapping_path)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=0.3,  
        stratify=all_labels,  
        random_state=42
    )
    print(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")
    
    # 2. 初始化 CSV 结果文件
    with open(config.RESULT_CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = [
            'ablation_type', 'epoch', 'train_loss', 'train_acc', 
            'lr', 'weight_decay', 'batch_size',
            'test_loss', 'test_acc', 'test_f1', 'test_auc', 'test_top5_acc'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 3. 遍历消融实验配置
        for abl_config in config.ABLATION_CONFIGS:
            print(f"\n===== Ablation Experiment: {abl_config['name']} =====")
            lr, weight_decay, batch_size = random_hyperparams()
            print(f'Hyperparams: lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}')
            nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
            
            # 构建数据集加载器
            data_transform = oracle_image_augment(config.IMG_SIZE)
            train_dataset = OracleDataset(
                train_paths, train_labels, label_to_idx=class_to_idx, transform=data_transform,
                is_tuopian="拓片" in config.DATASET_ROOT
            )
            test_dataset = OracleDataset(
                test_paths, test_labels, label_to_idx=class_to_idx, transform=data_transform,
                is_tuopian="拓片" in config.DATASET_ROOT
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
            
            # 4. 初始化模型
            net = OracleResNet18(
                num_classes=config.NUM_CLASSES,
                use_cbam=abl_config["use_cbam"],
                use_edge_enhance=abl_config["use_edge_enhance"]
            ).to(config.DEVICE)
            
            # 5. 优化器/损失函数
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
            
            # 6. 训练循环 + 每轮输出详细 test 指标
            save_path = os.path.join(config.MODEL_SAVE_ROOT, f'OracleResNet18_{abl_config["name"]}.pth')
            
            for epoch in range(config.EPOCHS):
                # ===== 训练阶段 =====
                net.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                train_bar = tqdm(train_loader, file=sys.stdout)
                for step, (images, labels) in enumerate(train_bar):
                    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * images.size(0)  # 按样本数累加损失
                    _, preds = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (preds == labels).sum().item()
                    
                    # 训练进度条显示（实时看训练集准确率）
                    train_acc_current = train_correct / train_total if train_total != 0 else 0
                    train_bar.desc = f"[Abl:{abl_config['name']}] Epoch[{epoch+1}/{config.EPOCHS}] " \
                                     f"Train Loss:{train_loss/train_total:.3f} " \
                                     f"Train Acc:{train_acc_current:.3f}"
                
                # 训练集指标汇总
                train_loss_epoch = train_loss / train_total if train_total != 0 else 0
                train_acc_epoch = train_correct / train_total if train_total != 0 else 0
                
                # ===== 测试集评估阶段 =====
                net.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = 0
                all_test_outputs = []
                all_test_labels = []
                with torch.no_grad():
                    test_bar = tqdm(test_loader, file=sys.stdout, desc="[Test Eval]")
                    for test_images, test_labels_ in test_bar:
                        test_images, test_labels_ = test_images.to(config.DEVICE), test_labels_.to(config.DEVICE)
                        outputs = net(test_images)
                        loss = loss_fn(outputs, test_labels_)
                        
                        test_loss += loss.item() * test_images.size(0)
                        _, preds = torch.max(outputs, 1)
                        test_total += test_labels_.size(0)
                        test_correct += (preds == test_labels_).sum().item()
                        
                        all_test_outputs.append(outputs)
                        all_test_labels.append(test_labels_)
                
                # 测试集指标计算
                test_loss_epoch = test_loss / test_total if test_total != 0 else 0
                test_acc_epoch = test_correct / test_total if test_total != 0 else 0
                test_f1_epoch, test_auc_epoch, test_top5_acc_epoch = 0.0, 0.0, 0.0
                if test_total > 0 and len(all_test_outputs) > 0:
                    test_outputs_cat = torch.cat(all_test_outputs)
                    test_labels_cat = torch.cat(all_test_labels)
                    test_acc_epoch,test_f1_epoch, test_auc_epoch, test_top5_acc_epoch = calculate_metrics(
                        test_outputs_cat, test_labels_cat, config.NUM_CLASSES
                    )
                
                # ===== 终端打印每轮完整指标 =====
                print(f"\nEpoch {epoch+1} Summary - Ablation: {abl_config['name']}")
                print(f"  Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f}")
                print(f"  Test  Loss: {test_loss_epoch:.4f} | Test  Acc: {test_acc_epoch:.4f}")
                print(f"  Test  F1: {test_f1_epoch:.4f} | Test  AUC: {test_auc_epoch:.4f} | Test Top5 Acc: {test_top5_acc_epoch:.4f}")
                
                # ===== 写入 CSV =====
                writer.writerow({
                    'ablation_type': abl_config['name'],
                    'epoch': epoch+1,
                    'train_loss': round(train_loss_epoch, 3),
                    'train_acc': round(train_acc_epoch, 3),
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'test_loss': round(test_loss_epoch, 3),
                    'test_acc': round(test_acc_epoch, 3),
                    'test_f1': round(test_f1_epoch, 3),
                    'test_auc': round(test_auc_epoch, 3),
                    'test_top5_acc': round(test_top5_acc_epoch, 3)
                })
                
                # 学习率调度
                scheduler.step()
            
            # 保存最终模型和映射关系到 checkpoint（权重 + 类别映射）
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'ablation': abl_config,
                'config': {
                    'img_size': config.IMG_SIZE,
                    'num_classes': config.NUM_CLASSES,
                    'epochs': config.EPOCHS
                }
            }
            torch.save(checkpoint, save_path)
            print(f'Model saved to: {save_path}')
    
    print('Training finished! Results saved to:', config.RESULT_CSV_PATH)

if __name__ == '__main__':
    train_model()
