import base64
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "甲骨文手写体+甲骨文拓片"
DEMO_DIR = ROOT_DIR / "演示数据"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "oracle_history.db"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MODEL_EXTENSIONS = {".pth"}
MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_MODEL_UPLOAD_BYTES = 120 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_float(name, default, minimum=None, maximum=None):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


load_env_file(ROOT_DIR / ".env")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MODEL_FOLDER"] = str(MODEL_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MODEL_UPLOAD_BYTES
app.config["DATABASE"] = str(DB_PATH)
app.config["ADMIN_TOKEN"] = os.environ.get("ORACLE_ADMIN_TOKEN", "")
app.config["AI_API_KEY"] = os.environ.get("ORACLE_AI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
app.config["AI_BASE_URL"] = os.environ.get("ORACLE_AI_BASE_URL", "").strip()
app.config["AI_MODEL"] = os.environ.get("ORACLE_AI_MODEL", "").strip()
app.config["AI_PROVIDER_NAME"] = os.environ.get("ORACLE_AI_PROVIDER", "OpenAI-compatible").strip() or "OpenAI-compatible"
app.config["AI_WIRE_API"] = os.environ.get("ORACLE_AI_WIRE_API", "chat_completions").strip().lower()
app.config["AI_TIMEOUT"] = env_int("ORACLE_AI_TIMEOUT", 35, minimum=5, maximum=120)
app.config["AI_MAX_TOKENS"] = env_int("ORACLE_AI_MAX_TOKENS", 800, minimum=128, maximum=4000)
app.config["AI_TEMPERATURE"] = env_float("ORACLE_AI_TEMPERATURE", 0.4, minimum=0.0, maximum=1.5)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def is_admin_request():
    token = app.config.get("ADMIN_TOKEN")
    if token:
        return request.headers.get("X-Admin-Token") == token
    return request.remote_addr in {"127.0.0.1", "::1", "localhost"}


def reject_non_admin():
    return jsonify({"success": False, "error": "该操作仅限管理员或本机开发环境使用"}), 403


def safe_upload_name(filename, fallback="upload"):
    raw_name = secure_filename(filename or "") or fallback
    stem = Path(raw_name).stem[:80] or fallback
    suffix = Path(filename or "").suffix.lower()
    return f"{stem}{suffix}"


def validate_image_file(file_storage):
    filename = file_storage.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ValueError("仅支持 PNG、JPG、JPEG、BMP、WEBP 图片")

    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError("图片不能超过 10MB")

    try:
        image = Image.open(file_storage.stream)
        image.verify()
        width, height = image.size
    except Exception as exc:
        raise ValueError("图片文件无法解析，请更换图片") from exc
    finally:
        file_storage.stream.seek(0)

    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("图片分辨率过大，请压缩后再上传")


def validate_model_file(file_storage):
    suffix = Path(file_storage.filename or "").suffix.lower()
    if suffix not in MODEL_EXTENSIONS:
        raise ValueError("仅支持 .pth 模型文件")

    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    if size > MAX_MODEL_UPLOAD_BYTES:
        raise ValueError("模型文件不能超过 120MB")


CATEGORY_DESCRIPTIONS = {
    "动物": "以兽、鸟、虫、鱼等为原型的象形字，常保留头角、足爪或身体轮廓。",
    "人物": "与人、身体、亲属和身份有关的字，多从人的站立、跪坐、侧身形象演变而来。",
    "自然": "记录日月山川、风雨雷电、草木土石等自然现象与物象。",
    "方位": "表示方向、位置、行进关系的字，常通过上下、内外、背向等构形表达。",
    "农牧": "与田猎、畜养、作物、器具和祭祀用牲相关，反映商代生产生活。",
    "器物": "由工具、兵器、车船、容器等实物形态抽象而来。",
    "行为": "表示动作、状态或社会活动，常由手、足、口、人等部件会意。",
    "时间天象": "与日月、旬日、干支、时令和占卜日期有关。",
    "未分类": "数据集中暂未归入明确主题的字，可先从字形和样本来源观察。"
}

CURATED_KNOWLEDGE = {
    "牛": {
        "category": "动物",
        "meaning": "牛字的甲骨文字形多突出牛角和头部轮廓，是典型的象形字。",
        "origin": "商代祭祀、畜牧和占卜材料中常见牛，字形常以正面牛头或侧面轮廓表现。",
        "tip": "观察这个字时，可以先找两只弯角，再看中部是否像牛头。"
    },
    "羊": {
        "category": "动物",
        "meaning": "羊字常强调对称的羊角，整体像羊头。",
        "origin": "羊在祭祀和畜养中很重要，甲骨文常用简洁线条保留角、头等特征。",
        "tip": "与牛相比，羊的角往往更卷曲、上部更对称。"
    },
    "日": {
        "category": "时间天象",
        "meaning": "日字源于太阳形象，早期常写成圆形或近圆形，中间加一点或短画。",
        "origin": "甲骨文中日既可表示太阳，也常参与日期、时令等占卜记录。",
        "tip": "如果字形中有封闭轮廓和中心点画，可联想到太阳。"
    },
    "月": {
        "category": "时间天象",
        "meaning": "月字像一弯月亮，保留弧形外廓。",
        "origin": "月相与祭祀、历法和占卜时间密切相关，因此在卜辞中很常见。",
        "tip": "月的字形通常比日更狭长，像弯月或半月。"
    },
    "山": {
        "category": "自然",
        "meaning": "山字像并列的山峰，通常由几个高低不同的竖向峰形组成。",
        "origin": "从群峰形象抽象而来，后来逐渐线条化为今天的山字结构。",
        "tip": "看见连续峰峦一样的轮廓时，可优先联想到山。"
    },
    "水": {
        "category": "自然",
        "meaning": "水字像流动的水脉，中间为主流，两侧有支流或水纹。",
        "origin": "从河流曲折流动的形象演变，后世逐渐变为三点水和水字结构。",
        "tip": "曲线、分叉和流动感是辨认水字的重要线索。"
    },
    "人": {
        "category": "人物",
        "meaning": "人字像侧身站立的人，保留身体和腿部的轮廓。",
        "origin": "甲骨文的人常以简化侧影表示，是很多会意字的基础部件。",
        "tip": "先看是否有头身和两腿分开的姿态。"
    },
    "大": {
        "category": "人物",
        "meaning": "大字像正面站立、两臂张开的人，表示成年人或宏大的意思。",
        "origin": "由人的正面形象演变，手臂展开使字形显得舒展。",
        "tip": "张臂站立的姿态是大字最醒目的特征。"
    },
    "王": {
        "category": "人物",
        "meaning": "王字与权力、首领相关，后世形成三横一竖的结构。",
        "origin": "甲骨文中王常见于王命、祭祀和征伐记录，是卜辞核心人物。",
        "tip": "如果识别到王，可以结合卜辞背景理解为商王或王权。"
    },
    "田": {
        "category": "农牧",
        "meaning": "田字像被阡陌分割的田地，是农耕空间的图示化表达。",
        "origin": "由田块边界和分区线条抽象而来。",
        "tip": "封闭方框中带交叉分割，往往与田地有关。"
    },
    "车": {
        "category": "器物",
        "meaning": "车字源于车轮、车轴和车厢的俯视或侧视形象。",
        "origin": "商代车马用于战争、礼仪和交通，字形常保留轮轴特征。",
        "tip": "圆形轮、轴线和车厢结构是识别重点。"
    },
    "舟": {
        "category": "器物",
        "meaning": "舟字像一条小船，常见长条形船身。",
        "origin": "由船体轮廓抽象而成，反映水上交通和渡涉活动。",
        "tip": "细长、两端翘起的轮廓很像船。"
    }
}

KEYWORD_CATEGORY_RULES = [
    ("动物", "牛羊犬虎鹿豕鱼鸟隹马兔龟象龙蛇鼠"),
    ("人物", "人女子母父兄弟王臣身自目耳口心"),
    ("自然", "水火山川土石木禾雨雷电云雪风"),
    ("方位", "上下中东西南北出入至"),
    ("农牧", "田年禾牧牢狩牡"),
    ("器物", "车舟刀弓矢戈鼎贝玉"),
    ("时间天象", "日月旦旬夕昃春夏秋冬"),
]


def get_db():
    conn = sqlite3.connect(app.config["DATABASE"])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    existing_columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(oracle_characters)").fetchall()
    }
    if existing_columns and "display_name" not in existing_columns:
        conn.execute("DROP TABLE IF EXISTS oracle_variants")
        conn.execute("DROP TABLE IF EXISTS oracle_characters")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            top_predictions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS oracle_characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            char_name TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            category TEXT,
            category_desc TEXT,
            meaning TEXT,
            origin TEXT,
            tip TEXT,
            sample_count INTEGER DEFAULT 0,
            train_count INTEGER DEFAULT 0,
            test_count INTEGER DEFAULT 0,
            first_image TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS oracle_variants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            char_name TEXT NOT NULL,
            folder_name TEXT NOT NULL,
            source_split TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_ext TEXT NOT NULL,
            rel_path TEXT NOT NULL UNIQUE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_variants_char ON oracle_variants(char_name)")
    conn.commit()
    conn.close()


def normalize_class_name(name):
    if not name:
        return ""
    clean = str(name).strip()
    clean = clean.replace("（", "(").replace("）", ")")
    clean = re.sub(r"\s+", "", clean)
    return clean.split("(")[0]


def category_for_char(char_name):
    if char_name in CURATED_KNOWLEDGE:
        return CURATED_KNOWLEDGE[char_name]["category"]
    for category, chars in KEYWORD_CATEGORY_RULES:
        if any(ch in char_name for ch in chars):
            return category
    if re.fullmatch(r"[0-9A-Fa-f]{5}", char_name or ""):
        return "未分类"
    return "未分类"


def build_default_knowledge(char_name, sample_count=0, train_count=0, test_count=0):
    curated = CURATED_KNOWLEDGE.get(char_name, {})
    category = curated.get("category") or category_for_char(char_name)
    category_desc = CATEGORY_DESCRIPTIONS.get(category, CATEGORY_DESCRIPTIONS["未分类"])
    meaning = curated.get(
        "meaning",
        f"数据集中收录了“{char_name}”的多种甲骨文字形，可从笔画走向、轮廓结构和拓片差异观察它的演变线索。"
    )
    origin = curated.get(
        "origin",
        "甲骨文常把具体物象、动作或观念压缩成高度概括的线条。同一字在不同甲骨片、摹写和拓片中会出现方向、繁简、残损程度的差异。"
    )
    tip = curated.get(
        "tip",
        f"本库当前包含 {sample_count} 张相关样本，其中训练集 {train_count} 张、测试集 {test_count} 张。可以点击“查看其他形体”对比手写体与拓片。"
    )
    return category, category_desc, meaning, origin, tip


def scan_dataset():
    characters = {}
    if not DATA_DIR.exists():
        return characters

    for split in ("train", "test"):
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            char_name = normalize_class_name(class_dir.name)
            if not char_name:
                continue
            entry = characters.setdefault(
                char_name,
                {
                    "display_name": char_name,
                    "sample_count": 0,
                    "train_count": 0,
                    "test_count": 0,
                    "first_image": "",
                    "variants": []
                }
            )
            for image_file in sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
                rel_path = f"{split}/{class_dir.name}/{image_file.name}"
                entry["sample_count"] += 1
                entry[f"{split}_count"] += 1
                if not entry["first_image"]:
                    entry["first_image"] = rel_path
                entry["variants"].append(
                    {
                        "char_name": char_name,
                        "folder_name": class_dir.name,
                        "source_split": split,
                        "filename": image_file.name,
                        "file_ext": image_file.suffix.lower(),
                        "rel_path": rel_path,
                    }
                )
    return characters


def rebuild_oracle_database():
    characters = scan_dataset()
    conn = get_db()
    conn.execute("DELETE FROM oracle_variants")
    conn.execute("DELETE FROM oracle_characters")

    for char_name, entry in characters.items():
        category, category_desc, meaning, origin, tip = build_default_knowledge(
            char_name,
            entry["sample_count"],
            entry["train_count"],
            entry["test_count"],
        )
        conn.execute(
            """
            INSERT INTO oracle_characters
                (char_name, display_name, category, category_desc, meaning, origin, tip,
                 sample_count, train_count, test_count, first_image, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                char_name,
                entry["display_name"],
                category,
                category_desc,
                meaning,
                origin,
                tip,
                entry["sample_count"],
                entry["train_count"],
                entry["test_count"],
                entry["first_image"],
            ),
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO oracle_variants
                (char_name, folder_name, source_split, filename, file_ext, rel_path)
            VALUES (:char_name, :folder_name, :source_split, :filename, :file_ext, :rel_path)
            """,
            entry["variants"],
        )

    conn.commit()
    conn.close()
    return len(characters)


def ensure_oracle_database():
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) AS count FROM oracle_characters").fetchone()
    conn.close()
    if row["count"] == 0:
        rebuild_oracle_database()


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        hidden = max(channel // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channel))
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        x = x * self.sigmoid(avg_out + max_out)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.cbam = CBAM(out_channels) if use_cbam else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam is not None:
            out = self.cbam(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class OracleResNet18(nn.Module):
    def __init__(self, num_classes, use_cbam=True, use_edge_enhance=True):
        super().__init__()
        self.in_channels = 64
        self.use_edge_enhance = use_edge_enhance
        if use_edge_enhance:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, use_cbam=False)
        self.layer2 = self._make_layer(128, 2, stride=2, use_cbam=False)
        self.layer3 = self._make_layer(256, 2, stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(512, 2, stride=2, use_cbam=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1, use_cbam=True):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample, use_cbam)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, use_cbam=use_cbam))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_edge_enhance:
            x = self.conv1(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(self.dropout(x))


class OracleClassifier:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.idx_to_class = None
        self.img_size = 224
        self.classnames = self._load_classnames()
        self.has_model = False

    def _load_classnames(self):
        json_path = ROOT_DIR / "saved_models" / "classnames.json"
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): v for k, v in data.get("idx_to_class", {}).items()}
        except Exception:
            return {}

    def load_model(self, checkpoint_path=None):
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else ROOT_DIR / "saved_models" / "OracleResNet18_FullImproved.pth"
        if not checkpoint_path.exists():
            self.has_model = False
            return False
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
                if "idx_to_class" in checkpoint:
                    self.idx_to_class = {int(k): v for k, v in checkpoint["idx_to_class"].items()}
                if "config" in checkpoint and "img_size" in checkpoint["config"]:
                    self.img_size = checkpoint["config"]["img_size"]
                    if isinstance(self.img_size, (tuple, list)):
                        self.img_size = self.img_size[0]
                use_cbam = bool(checkpoint.get("ablation", {}).get("use_cbam", False))
                use_edge_enhance = bool(checkpoint.get("ablation", {}).get("use_edge_enhance", False))
            else:
                model_state = checkpoint
                self.idx_to_class = self.classnames
                use_cbam = False
                use_edge_enhance = True
            if not self.idx_to_class:
                raise RuntimeError("无法确定模型类别映射")
            self.model = OracleResNet18(len(self.idx_to_class), use_cbam=use_cbam, use_edge_enhance=use_edge_enhance)
            self.model.load_state_dict(model_state, strict=False)
            self.model.to(self.device).eval()
            self.has_model = True
            print(f"模型加载成功：{checkpoint_path}，类别数：{len(self.idx_to_class)}，设备：{self.device}")
            return True
        except Exception as exc:
            print(f"模型加载失败：{exc}")
            self.has_model = False
            return False

    def preprocess_image(self, image):
        transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()])
        return transform(image.convert("RGB")).unsqueeze(0)

    def predict(self, image_path=None, image_data=None):
        if not self.has_model:
            return {
                "prediction": "未加载模型",
                "confidence": 0.0,
                "top_predictions": [{"rank": 1, "class": "请先加载 .pth 模型", "confidence": 0.0}],
            }
        try:
            if image_path:
                img = Image.open(image_path).convert("RGB")
            elif image_data:
                payload = image_data.split(",", 1)[-1]
                try:
                    decoded = base64.b64decode(payload, validate=True)
                except Exception as exc:
                    raise ValueError("画板图片数据无效") from exc
                if len(decoded) > MAX_IMAGE_UPLOAD_BYTES:
                    raise ValueError("画板图片数据过大")
                img = Image.open(BytesIO(decoded)).convert("RGB")
            else:
                return None
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError("图片分辨率过大，请压缩后再识别")

            x = self.preprocess_image(img).to(self.device)
            with torch.no_grad():
                probs = F.softmax(self.model(x), dim=1)
                top_k, indices = torch.topk(probs, min(5, probs.shape[1]))
                top_predictions = []
                for rank, (prob, idx) in enumerate(zip(top_k[0], indices[0]), start=1):
                    class_name = self.idx_to_class.get(int(idx.item()), "UNKNOWN")
                    top_predictions.append(
                        {
                            "rank": rank,
                            "class": class_name,
                            "display_class": normalize_class_name(class_name),
                            "confidence": round(float(prob.item()) * 100, 2),
                        }
                    )
                pred_idx = int(indices[0, 0].item())
                pred_class = self.idx_to_class.get(pred_idx, "UNKNOWN")
            return {
                "prediction": pred_class,
                "display_prediction": normalize_class_name(pred_class),
                "confidence": round(float(probs[0, pred_idx].item()) * 100, 2),
                "top_predictions": top_predictions,
            }
        except Exception as exc:
            print(f"识别失败：{exc}")
            return {"prediction": "识别失败", "display_prediction": "识别失败", "confidence": 0.0, "top_predictions": []}


init_db()
ensure_oracle_database()
classifier = OracleClassifier()
classifier.load_model()
user_models = {}


def row_to_knowledge(row):
    if not row:
        return {}
    return {
        "char_name": row["char_name"],
        "id": row["id"],
        "display_name": row["display_name"],
        "category": row["category"],
        "category_desc": row["category_desc"],
        "meaning": row["meaning"],
        "origin": row["origin"],
        "tip": row["tip"],
        "sample_count": row["sample_count"],
        "train_count": row["train_count"],
        "test_count": row["test_count"],
        "first_image": f"/api/variant_image/{row['first_image']}" if row["first_image"] else "",
    }


def find_character(char_name):
    clean_name = normalize_class_name(char_name)
    conn = get_db()
    row = conn.execute("SELECT * FROM oracle_characters WHERE char_name = ?", (clean_name,)).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM oracle_characters WHERE char_name LIKE ? OR display_name LIKE ? LIMIT 1",
            (f"{clean_name}%", f"{clean_name}%"),
        ).fetchone()
    conn.close()
    return row_to_knowledge(row)


def get_assistant_status():
    has_endpoint = bool(app.config.get("AI_BASE_URL") and app.config.get("AI_MODEL"))
    model = app.config.get("AI_MODEL") or ""
    return {
        "enabled": bool(has_endpoint),
        "has_endpoint": bool(app.config.get("AI_BASE_URL")),
        "has_api_key": bool(app.config.get("AI_API_KEY")),
        "model": model or "未配置",
        "base_url": app.config.get("AI_BASE_URL") or "",
        "provider_name": app.config.get("AI_PROVIDER_NAME") or "OpenAI-compatible",
        "wire_api": app.config.get("AI_WIRE_API") or "chat_completions",
    }


def normalize_ai_base_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        return ""
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def normalize_responses_base_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        return ""
    if base_url.endswith("/responses"):
        return base_url
    return f"{base_url}/responses"


def build_assistant_context(context):
    if not isinstance(context, dict):
        context = {}

    parts = []
    prediction = context.get("prediction")
    if prediction:
        parts.append(f"当前识别结果：{prediction}")

    confidence = context.get("confidence")
    if confidence is not None:
        parts.append(f"当前置信度：{confidence}%")

    knowledge = context.get("knowledge")
    if isinstance(knowledge, dict):
        for label, key in (
            ("字形含义", "meaning"),
            ("类别", "category"),
            ("类别说明", "category_desc"),
            ("字源观察", "origin"),
            ("观察提示", "tip"),
        ):
            value = knowledge.get(key)
            if value:
                parts.append(f"{label}：{value}")

    top_predictions = context.get("top_predictions")
    if isinstance(top_predictions, list) and top_predictions:
        preview = []
        for item in top_predictions[:5]:
            if not isinstance(item, dict):
                continue
            name = item.get("display_class") or item.get("class") or ""
            conf = item.get("confidence")
            if name:
                preview.append(f"{name}({conf}%)" if conf is not None else name)
        if preview:
            parts.append(f"候选结果：{'，'.join(preview)}")

    if not parts:
        return "当前页面还没有新的识别结果。"
    return "\n".join(parts)


def local_assistant_reply(message, context):
    lower_message = message.lower()
    prediction = context.get("prediction") if isinstance(context, dict) else ""
    knowledge = context.get("knowledge") if isinstance(context, dict) else {}
    if not isinstance(knowledge, dict):
        knowledge = {}

    if prediction or knowledge:
        display_name = knowledge.get("display_name") or prediction or "这个字"
        meaning = knowledge.get("meaning") or "可以结合识别结果、相似字形和数据集样本继续观察。"
        origin = knowledge.get("origin") or knowledge.get("tip") or "建议先看主要轮廓，再比较同字异形里的共同部件。"
        return (
            f"我现在还没有接入在线大模型，先用本地知识给你一个简要说明：{display_name}。"
            f"{meaning} {origin} 如果你想启用真正的大模型问答，请在运行网站前配置 "
            "ORACLE_AI_API_KEY、ORACLE_AI_BASE_URL 和 ORACLE_AI_MODEL。"
        )

    if any(keyword in lower_message for keyword in ("配置", "api", "key", "模型", "model")):
        return (
            "这个助手支持 OpenAI-compatible Chat Completions 接口。运行前设置 "
            "ORACLE_AI_API_KEY、ORACLE_AI_BASE_URL、ORACLE_AI_MODEL 即可切换模型，"
            "例如 DeepSeek、通义千问、OpenAI 或本地兼容服务。"
        )

    return (
        "我可以帮你解释识别结果、比较甲骨文字形、整理字源观察，也能回答项目使用问题。"
        "当前未配置在线大模型，所以回答会比较基础；配置 API 后就会调用你指定的模型。"
    )


def build_chat_messages(message, history, context):
    safe_history = []
    if isinstance(history, list):
        for item in history[-8:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                safe_history.append({"role": role, "content": content[:1600]})

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个甲骨文识别网站里的智能助手。请用简洁、准确、友好的中文回答。"
                "你可以解释识别结果、字形演变、同字异形观察、项目使用方式和模型配置。"
                "如果资料不足，请明确说明不确定，不要编造考古结论。"
            ),
        },
        {
            "role": "system",
            "content": f"页面上下文：\n{build_assistant_context(context)}",
        },
        *safe_history,
        {"role": "user", "content": message},
    ]
    return messages


def parse_responses_text(result):
    if not isinstance(result, dict):
        return ""

    text = result.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

    chunks = []
    output = result.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text.strip())
                elif part.get("type") in {"output_text", "text"}:
                    nested_text = part.get("content")
                    if isinstance(nested_text, str) and nested_text.strip():
                        chunks.append(nested_text.strip())
    return "\n".join(chunks).strip()


def post_ai_json(endpoint, payload):
    api_key = app.config.get("AI_API_KEY", "")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 Oracle-Assistant/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=app.config.get("AI_TIMEOUT", 35)) as response:
        raw_response = response.read().decode("utf-8")
        return json.loads(raw_response or "{}")


def call_chat_completions_assistant(message, history, context, model):
    payload = {
        "model": model,
        "messages": build_chat_messages(message, history, context),
        "temperature": app.config.get("AI_TEMPERATURE", 0.4),
        "max_tokens": app.config.get("AI_MAX_TOKENS", 800),
    }
    endpoint = normalize_ai_base_url(app.config.get("AI_BASE_URL", ""))
    result = post_ai_json(endpoint, payload)
    reply = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not reply:
        reply = "模型没有返回有效内容，请稍后再试。"
    return {
        "reply": reply,
        "provider": "chat_completions",
        "model": model,
        "configured": True,
    }


def call_responses_assistant(message, history, context, model):
    chat_messages = build_chat_messages(message, history, context)
    system_parts = [item["content"] for item in chat_messages if item["role"] == "system"]
    input_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in chat_messages
        if item["role"] in {"user", "assistant"}
    ]
    payload = {
        "model": model,
        "instructions": "\n\n".join(system_parts),
        "input": input_messages,
        "temperature": app.config.get("AI_TEMPERATURE", 0.4),
        "max_output_tokens": app.config.get("AI_MAX_TOKENS", 800),
    }
    endpoint = normalize_responses_base_url(app.config.get("AI_BASE_URL", ""))
    result = post_ai_json(endpoint, payload)
    reply = parse_responses_text(result)
    if not reply:
        reply = "模型没有返回有效内容，请稍后再试。"
    return {
        "reply": reply,
        "provider": "responses",
        "model": model,
        "configured": True,
    }


def call_ai_assistant(message, history, context, model_override=None):
    base_url = app.config.get("AI_BASE_URL", "")
    model = (model_override or app.config.get("AI_MODEL") or "").strip()
    if not base_url or not model:
        return {
            "reply": local_assistant_reply(message, context),
            "provider": "local",
            "model": "local-fallback",
            "configured": False,
        }

    wire_api = (app.config.get("AI_WIRE_API") or "chat_completions").replace("-", "_").lower()
    if wire_api == "responses":
        return call_responses_assistant(message, history, context, model)
    return call_chat_completions_assistant(message, history, context, model)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload_model", methods=["POST"])
def upload_model():
    if not is_admin_request():
        return reject_non_admin()
    if "model" not in request.files:
        return jsonify({"success": False, "error": "没有提供模型文件"}), 400
    file = request.files["model"]
    if file.filename == "":
        return jsonify({"success": False, "error": "文件名为空"}), 400
    try:
        validate_model_file(file)
        model_id = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_path = MODEL_DIR / f"{model_id}.pth"
        file.save(model_path)
        temp_classifier = OracleClassifier()
        if temp_classifier.load_model(model_path):
            user_models[model_id] = {"path": str(model_path), "name": file.filename, "classifier": temp_classifier}
            return jsonify({"success": True, "model_id": model_id})
        model_path.unlink(missing_ok=True)
        return jsonify({"success": False, "error": "模型加载失败，请确认类别数与结构匹配"}), 400
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/classify", methods=["POST"])
def classify():
    if "image" not in request.files and "image_data" not in request.form:
        return jsonify({"error": "请先上传图片或在画板上书写"}), 400
    image_path = None
    try:
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "文件名为空"}), 400
            validate_image_file(file)
            safe_name = safe_upload_name(file.filename, "oracle_image")
            image_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_name}"
            file.save(image_path)

        model_id = request.form.get("model_id")
        active_classifier = user_models.get(model_id, {}).get("classifier", classifier)
        result = active_classifier.predict(image_path=image_path, image_data=request.form.get("image_data"))
        if result is None:
            return jsonify({"error": "识别失败，请检查模型文件是否可用"}), 500

        display_name = result.get("display_prediction") or normalize_class_name(result["prediction"])
        knowledge = find_character(display_name)
        if knowledge:
            result["knowledge"] = knowledge
            result["display_prediction"] = knowledge["display_name"]

        conn = get_db()
        cursor = conn.execute(
            "INSERT INTO history (image_path, prediction, confidence, top_predictions) VALUES (?, ?, ?, ?)",
            (
                str(image_path.relative_to(BASE_DIR)).replace("\\", "/") if image_path else "",
                result.get("display_prediction", result["prediction"]),
                result["confidence"],
                json.dumps(result["top_predictions"], ensure_ascii=False),
            ),
        )
        conn.commit()
        result["record_id"] = cursor.lastrowid
        conn.close()
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    conn = get_db()
    records = [dict(row) for row in conn.execute("SELECT * FROM history ORDER BY created_at DESC LIMIT 50").fetchall()]
    conn.close()
    return jsonify(records)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    conn = get_db()
    conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    if not is_admin_request():
        return reject_non_admin()
    conn = get_db()
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/demo/<char_name>", methods=["GET"])
def get_demo_image(char_name):
    if DEMO_DIR.exists():
        for fname in os.listdir(DEMO_DIR):
            if Path(fname).stem == char_name:
                return send_from_directory(DEMO_DIR, fname)
    return jsonify({"error": "没有找到演示图片"}), 404


@app.route("/api/knowledge/<char_name>", methods=["GET"])
def get_knowledge(char_name):
    knowledge = find_character(char_name)
    return jsonify({"success": bool(knowledge), "knowledge": knowledge})


@app.route("/api/variants/<char_name>", methods=["GET"])
def get_variants(char_name):
    clean_name = normalize_class_name(char_name)
    limit = min(int(request.args.get("limit", 60)), 200)
    conn = get_db()
    rows = conn.execute(
        """
        SELECT * FROM oracle_variants
        WHERE char_name = ?
        ORDER BY source_split, filename
        LIMIT ?
        """,
        (clean_name, limit),
    ).fetchall()
    conn.close()
    variants = [
        {
            "filename": row["filename"],
            "folder": row["folder_name"],
            "source": row["source_split"],
            "path": f"/api/variant_image/{row['rel_path']}",
        }
        for row in rows
    ]
    return jsonify({"success": True, "variants": variants})


@app.route("/api/variant_image/<path:rel_path>", methods=["GET"])
def get_variant_image(rel_path):
    safe_path = (DATA_DIR / rel_path).resolve()
    if DATA_DIR.resolve() not in safe_path.parents:
        return jsonify({"error": "非法路径"}), 400
    if safe_path.exists() and safe_path.suffix.lower() in IMAGE_EXTENSIONS:
        return send_file(safe_path)
    return jsonify({"error": "图片不存在"}), 404


@app.route("/api/categories", methods=["GET"])
def get_categories():
    conn = get_db()
    rows = conn.execute(
        "SELECT category, COUNT(*) AS count, SUM(sample_count) AS samples FROM oracle_characters GROUP BY category ORDER BY count DESC"
    ).fetchall()
    conn.close()
    categories = [
        {
            "name": row["category"],
            "description": CATEGORY_DESCRIPTIONS.get(row["category"], CATEGORY_DESCRIPTIONS["未分类"]),
            "count": row["count"],
            "samples": row["samples"] or 0,
        }
        for row in rows
    ]
    return jsonify({"success": True, "categories": categories})


@app.route("/api/characters", methods=["GET"])
def get_characters():
    query = normalize_class_name(request.args.get("q", ""))
    category = request.args.get("category", "")
    limit = min(int(request.args.get("limit", 120)), 500)
    sql = "SELECT * FROM oracle_characters WHERE 1=1"
    params = []
    if query:
        sql += " AND (char_name LIKE ? OR display_name LIKE ? OR category LIKE ?)"
        params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])
    if category:
        sql += " AND category = ?"
        params.append(category)
    sql += " ORDER BY category, char_name LIMIT ?"
    params.append(limit)
    conn = get_db()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return jsonify({"success": True, "characters": [row_to_knowledge(row) for row in rows]})


@app.route("/api/assistant/status", methods=["GET"])
def assistant_status():
    return jsonify({"success": True, **get_assistant_status()})


@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"success": False, "error": "请输入要咨询的问题"}), 400
    if len(message) > 2000:
        return jsonify({"success": False, "error": "问题过长，请精简后再发送"}), 400

    history = data.get("history", [])
    context = data.get("context", {})
    model_override = str(data.get("model", "")).strip()
    try:
        result = call_ai_assistant(message, history, context, model_override)
        return jsonify({"success": True, **result})
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:500]
        return jsonify({"success": False, "error": f"大模型接口返回错误：HTTP {exc.code} {detail}"}), 502
    except urllib.error.URLError as exc:
        return jsonify({"success": False, "error": f"无法连接大模型接口：{exc.reason}"}), 502
    except Exception as exc:
        return jsonify({"success": False, "error": f"AI 助手调用失败：{exc}"}), 500


@app.route("/api/assistant/config", methods=["GET"])
def assistant_config():
    status = get_assistant_status()
    status["success"] = True
    return jsonify(status)


@app.route("/api/database/rebuild", methods=["POST"])
def rebuild_database():
    if not is_admin_request():
        return reject_non_admin()
    count = rebuild_oracle_database()
    return jsonify({"success": True, "count": count})


@app.route("/api/database/stats", methods=["GET"])
def database_stats():
    conn = get_db()
    chars = conn.execute("SELECT COUNT(*) AS count FROM oracle_characters").fetchone()["count"]
    variants = conn.execute("SELECT COUNT(*) AS count FROM oracle_variants").fetchone()["count"]
    conn.close()
    return jsonify({"success": True, "characters": chars, "variants": variants, "data_folder": str(DATA_DIR)})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    app.run(debug=debug, host=host, port=port)
