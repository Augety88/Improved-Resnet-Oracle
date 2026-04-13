import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
from torchvision import transforms

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

# ===================== 数据增强函数 =====================
def oracle_image_augment(img_size):
    """图像预处理和数据增强"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

# ===================== 加载模型函数 =====================
def load_checkpoint(checkpoint_path, device):
    """
    加载 checkpoint 文件，支持新旧两种格式
    
    新格式（train_idx.py 生成）：
    - 包含 model_state_dict, idx_to_class, ablation, config
    - 无需外部配置文件即可加载
    
    旧格式：
    - 仅包含 state_dict
    - 需要 classnames.json 辅助
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    idx_to_class = None
    class_to_idx = None

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式：包含映射、配置、模型权重
        model_state = checkpoint['model_state_dict']
        idx_to_class = checkpoint.get('idx_to_class')
        class_to_idx = checkpoint.get('class_to_idx')

    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        # 旧格式：仅 state_dict（键为字符串，如 'conv1.weight'）
        model_state = checkpoint
        # 尝试从同目录查找 classnames.json
        json_path = os.path.join(os.path.dirname(checkpoint_path), 'classnames.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            idx_to_class = mapping.get('idx_to_class')
            class_to_idx = mapping.get('class_to_idx')
        else:
            raise RuntimeError(f'旧格式 checkpoint 需要 classnames.json 文件，但未找到：{json_path}')

    else:
        raise RuntimeError('checkpoint format error')

    # 处理 idx_to_class 格式问题：确保键是整数，值是类别名
    if idx_to_class is not None:
        # 检查格式是否正确
        sample_key = list(idx_to_class.keys())[0]
        if isinstance(sample_key, str):
            # 格式反了：{"类别名": 索引} -> {索引："类别名"}
            print("警告：检测到 idx_to_class 格式错误，正在自动修复...")
            idx_to_class = {int(v): k for k, v in idx_to_class.items()}
    
    # 如果 idx_to_class 仍然为 None，尝试从 class_to_idx 转换
    if idx_to_class is None and class_to_idx is not None:
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    if idx_to_class is None:
        raise RuntimeError('idx_to_class missing in checkpoint')

    num_classes = len(idx_to_class)
    
    # 根据 checkpoint 中的 ablation 信息重建模型结构
    use_cbam = False
    use_edge_enhance = False
    if isinstance(checkpoint, dict) and 'ablation' in checkpoint:
        ablation = checkpoint['ablation']
        use_cbam = bool(ablation.get('use_cbam', False))
        use_edge_enhance = bool(ablation.get('use_edge_enhance', False))

    model = OracleResNet18(num_classes=num_classes, use_cbam=use_cbam, use_edge_enhance=use_edge_enhance)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(f"Error(s) in loading state_dict for OracleResNet18:\nMissing keys: {missing}\nUnexpected keys: {unexpected}\n(请检查模型结构参数 use_cbam/use_edge_enhance 是否与 checkpoint 一致)")
    model.to(device).eval()
    
    # 打印调试信息
    print(f"模型加载成功！")
    print(f"  类别数：{num_classes}")
    print(f"  idx_to_class 前 3 个：{dict(list(idx_to_class.items())[:3])}")
    
    return model, idx_to_class


def infer_image(model, idx_to_class, image_path, img_size, device):
    """对单张图像进行推理"""
    transform = oracle_image_augment(img_size)
    img = Image.open(image_path).convert('RGB')  # 改为 RGB 以匹配训练时的输入
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred].item())
    return idx_to_class.get(pred, 'UNKNOWN'), conf


class OracleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('甲骨文识别界面')
        self.root.geometry('700x500')

        self.checkpoint_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.user_msg = tk.StringVar(value='请先选择模型权重（.pth）和图像文件或文件夹')

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.idx_to_class = None
        self.loaded_img_size = 128

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text='模型 checkpoint:').grid(row=0, column=0, sticky=tk.W, pady=2)
        entry_chk = ttk.Entry(frame, textvariable=self.checkpoint_path, width=70)
        entry_chk.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Button(frame, text='选择', command=self.choose_checkpoint).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text='图片或文件夹:').grid(row=1, column=0, sticky=tk.W, pady=2)
        entry_img = ttk.Entry(frame, textvariable=self.image_path, width=70)
        entry_img.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Button(frame, text='选择', command=self.choose_input).grid(row=1, column=2, padx=5)

        ttk.Label(frame, text='图像大小（训练时用相同尺寸）:').grid(row=2, column=0, sticky=tk.W, pady=2)
        self.img_size_spin = ttk.Spinbox(frame, from_=64, to=512, increment=32, width=10)
        self.img_size_spin.set('224')  # 默认改为 224，与训练一致
        self.img_size_spin.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Button(frame, text='加载模型', command=self.load_model).grid(row=3, column=0, pady=10)
        ttk.Button(frame, text='开始识别', command=self.run_inference).grid(row=3, column=1, pady=10)

        ttk.Label(frame, textvariable=self.user_msg, foreground='blue').grid(row=4, column=0, columnspan=3, sticky=tk.W)

        self.result_text = tk.Text(frame, height=10, wrap=tk.NONE)
        self.result_text.grid(row=5, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        frame.rowconfigure(5, weight=1)
        frame.columnconfigure(1, weight=1)

        # 添加图片显示区域
        self.image_label = ttk.Label(frame, text='图片预览')
        self.image_label.grid(row=6, column=0, columnspan=3, pady=5)

    def choose_checkpoint(self):
        path = filedialog.askopenfilename(title='选择 checkpoint 文件', filetypes=[('PyTorch files', '*.pth'), ('All files', '*.*')])
        if path:
            self.checkpoint_path.set(path)

    def choose_input(self):
        if messagebox.askyesno('选择模式', '是否选择文件夹（是）或单图（否）？'):
            path = filedialog.askdirectory(title='选择图像目录')
        else:
            path = filedialog.askopenfilename(title='选择图像文件', filetypes=[('Image files', '*.png *.jpg *.jpeg *.bmp'), ('All files', '*.*')])
        if path:
            self.image_path.set(path)

    def load_model(self):
        cp = self.checkpoint_path.get().strip()
        if not cp or not os.path.exists(cp):
            messagebox.showerror('错误', '请先选择有效的 checkpoint 文件')
            return
        try:
            self.model, self.idx_to_class = load_checkpoint(cp, self.device)
            self.loaded_img_size = int(self.img_size_spin.get())
            
            # 从 checkpoint 中获取训练时的图像尺寸（如果有）
            checkpoint = torch.load(cp, map_location=self.device)
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                train_img_size = checkpoint['config'].get('img_size', (224, 224))
                if isinstance(train_img_size, tuple):
                    self.loaded_img_size = train_img_size[0]
                    self.img_size_spin.set(str(train_img_size[0]))
            
            self.user_msg.set(f'已加载模型：{os.path.basename(cp)}，类别数 {len(self.idx_to_class)}，设备 {self.device}')
        except Exception as e:
            messagebox.showerror('加载失败', str(e))

    def run_inference(self):
        inp = self.image_path.get().strip()
        if self.model is None or self.idx_to_class is None:
            messagebox.showerror('错误', '请先加载模型')
            return
        if not inp or not os.path.exists(inp):
            messagebox.showerror('错误', '请先选择图像或文件夹')
            return

        self.result_text.delete(1.0, tk.END)
        paths = []
        if os.path.isdir(inp):
            for root, _, files in os.walk(inp):
                for fn in sorted(files):
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        paths.append(os.path.join(root, fn))
        else:
            paths = [inp]

        if not paths:
            messagebox.showwarning('提示', '未找到图片文件')
            return

        self.user_msg.set(f'开始识别 {len(paths)} 张图像 ...')
        for p in paths:
            try:
                # 显示图片
                img = Image.open(p).convert('RGB')
                img_resized = img.resize((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                self.image_label.config(image=photo)
                self.image_label.image = photo  # 保持引用

                # 推理
                label, conf = infer_image(self.model, self.idx_to_class, p, self.loaded_img_size, self.device)
                self.result_text.insert(tk.END, f'{p} -> {label} (conf={conf:.4f})\n')
                self.result_text.see(tk.END)  # 滚动到末尾
                self.root.update()  # 刷新界面
            except Exception as e:
                self.result_text.insert(tk.END, f'{p} -> ERROR: {e}\n')

        self.user_msg.set('识别完成')


if __name__ == '__main__':
    root = tk.Tk()
    app = OracleGUI(root)
    root.mainloop()
