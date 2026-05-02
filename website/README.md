# Web 网站部署说明

`website/` 是甲骨文识别与字形库的 Flask Web 应用。它复用根目录的模型、演示样本和数据集，提供图片上传识别、手写画板、字形数据库、历史记录和 AI 助手。

## 功能

- 图片上传识别：支持 PNG、JPG、JPEG、BMP、WEBP。
- 手写画板：可直接在页面书写字形并识别。
- 识别结果：显示 Top-1、置信度、候选结果和同字形体。
- 字形知识库：自动扫描 `甲骨文手写体+甲骨文拓片` 并补充内置释义。
- 数据库查询：支持按汉字、类别、编码检索。
- 历史记录：本地 SQLite 保存识别记录。
- AI 助手：可拖动悬浮球，自动解释识别结果，也可继续对话。

## 目录

```text
website/
├── app.py
├── oracle_knowledge.json
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    ├── js/app.js
    ├── uploads/       # 运行时上传目录，不提交
    └── models/        # 运行时上传模型目录，不提交
```

## 运行

在项目根目录执行：

```bash
pip install -r requirements.txt
python website/app.py
```

访问：

```text
http://127.0.0.1:5000
```

## 模型加载

默认模型路径：

```text
saved_models/OracleResNet18_FullImproved.pth
saved_models/classnames.json
```

也可以在网页的“模型管理”区域上传 `.pth` 权重进行当前会话测试。

## AI 助手配置

在项目根目录创建 `.env`，不要提交真实密钥。

```bash
ORACLE_AI_BASE_URL=https://api.example.com/v1
ORACLE_AI_MODEL=your-model-name
ORACLE_AI_API_KEY=your-api-key
ORACLE_AI_PROVIDER=OpenAI-compatible
ORACLE_AI_WIRE_API=chat_completions
ORACLE_AI_TIMEOUT=35
ORACLE_AI_MAX_TOKENS=800
ORACLE_AI_TEMPERATURE=0.4
```

LongAPI / Responses 示例：

```bash
ORACLE_AI_BASE_URL=https://haoshuai.cc.cd
ORACLE_AI_MODEL=gpt-5.4
ORACLE_AI_API_KEY=你的新密钥
ORACLE_AI_PROVIDER=LongAPI
ORACLE_AI_WIRE_API=responses
```

说明：

- `ORACLE_AI_WIRE_API=chat_completions` 时会请求 `<base_url>/chat/completions`。
- `ORACLE_AI_WIRE_API=responses` 时会请求 `<base_url>/responses`。
- 未配置在线模型时，AI 助手会使用本地知识兜底回答。
- 当前识别结果、置信度、候选结果和字形知识会作为上下文传给助手。

## API

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/classify` | POST | 上传图片或画板数据识别 |
| `/api/history` | GET | 获取历史记录 |
| `/api/history/<id>` | DELETE | 删除单条历史 |
| `/api/history/clear` | POST | 清空历史 |
| `/api/demo/<char_name>` | GET | 获取演示样本 |
| `/api/knowledge/<char_name>` | GET | 获取字形知识 |
| `/api/variants/<char_name>` | GET | 获取同字异形 |
| `/api/categories` | GET | 获取类别统计 |
| `/api/characters` | GET | 搜索字形数据库 |
| `/api/assistant/status` | GET | 获取 AI 配置状态 |
| `/api/assistant/chat` | POST | AI 助手问答 |

## 运行时文件

以下文件由程序运行生成，已在 `.gitignore` 中忽略：

```text
website/oracle_history.db
website/static/uploads/
website/static/models/
```
