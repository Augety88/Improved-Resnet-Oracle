# 古文字适配的改进ResNet18 + CBAM注意力模块
import torch
import torch.nn as nn

class CBAM(nn.Module):
    """通道+空间注意力模块，聚焦古文字笔画特征"""
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

class BasicBlock(nn.Module):
    """改进的残差块，可选嵌入CBAM"""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=True):
        """
        初始化 BasicBlock 残差块。
        
        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int): 卷积步长，默认为 1
            downsample (nn.Module): 下采样模块，用于残差连接时的维度匹配，默认为 None
            use_cbam (bool): 是否使用 CBAM 注意力机制，默认为 True
        
        返回:
            None
        """
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层：进行特征提取，可调整通道数和空间维度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        # 第一批归一化层：稳定训练过程
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # ReLU 激活函数：引入非线性
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层：保持通道数不变，进一步提取特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 第二批归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 保存下采样模块引用
        self.downsample = downsample
        
        # 保存 CBAM 使用标志
        self.use_cbam = use_cbam
        
        # 如果使用 CBAM，则创建注意力模块
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

class OracleResNet18(nn.Module):
    """适配古文字的改进ResNet18"""
    def __init__(self, num_classes, use_cbam=True, use_edge_enhance=True):
        super(OracleResNet18, self).__init__()
        self.in_channels = 64
        self.use_edge_enhance = use_edge_enhance
        
        # 边缘增强：替换7×7卷积为3个3×3卷积
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
            # 原始ResNet18浅层
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, use_cbam=False)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, use_cbam=False)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, use_cbam=False)
        
        # 输出层优化：Dropout抑制小样本过拟合
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