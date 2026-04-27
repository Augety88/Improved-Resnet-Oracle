# 甲骨文识别系统 - 网站部署

## 项目概述

这是一个基于 Flask 的甲骨文识别网站，使用改进的 ResNet18+CBAM 深度学习模型进行甲骨文图像识别。

## 功能特性

- 📤 图像上传：支持拖拽或点击上传甲骨文图像
- 🔮 智能识别：基于深度学习模型的自动识别
- 📊 结果展示：显示识别结果和置信度
- 📜 历史记录：自动保存和管理识别历史
- 🖼️ 演示样本：提供预设的演示样本快速测试

## 目录结构

```
website/
├── app.py              # Flask 主应用
├── requirements.txt    # Python 依赖
├── static/
│   ├── css/
│   │   └── style.css   # 样式文件
│   ├── js/
│   │   └── app.js       # 前端脚本
│   └── uploads/         # 上传文件目录
└── templates/
    └── index.html       # 主页面
```

## 安装与运行

### 1. 安装依赖

```bash
cd website
pip install -r requirements.txt
```

### 2. 运行服务器

```bash
python app.py
```

### 3. 访问网站

打开浏览器访问: http://localhost:5000

## 模型文件

系统需要模型权重文件才能进行识别。有两种方式提供模型：

### 方式一：使用已训练的模型
将训练好的 `.pth` 文件放到 `saved_models/` 目录，命名为 `best_model.pth`

### 方式二：使用 classnames.json
如果只有 classnames.json 文件，系统将使用模拟模式进行演示

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/classify` | POST | 上传图像进行识别 |
| `/api/history` | GET | 获取历史记录 |
| `/api/history/<id>` | DELETE | 删除单条记录 |
| `/api/history/clear` | POST | 清空历史记录 |

## 硬件要求

- CPU 或 NVIDIA GPU (CUDA)
- 内存: 4GB+
- 磁盘空间: 500MB+

## 技术栈

- **后端**: Flask, PyTorch, TorchVision
- **前端**: HTML5, CSS3, JavaScript
- **数据库**: SQLite (内置)
- **模型**: 改进 ResNet18 + CBAM 注意力机制