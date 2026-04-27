import os
import json
import base64
import sqlite3
from datetime import datetime
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'static/models'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 增加到100MB
app.config['DATABASE'] = 'oracle_history.db'

# 允许跨域请求
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            top_predictions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

class BasicBlock(nn.Module):
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

class OracleResNet18(nn.Module):
    def __init__(self, num_classes, use_cbam=True, use_edge_enhance=True):
        super(OracleResNet18, self).__init__()
        self.in_channels = 64
        self.use_edge_enhance = use_edge_enhance
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
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, use_cbam=False)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, use_cbam=False)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, use_cbam=use_cbam)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, use_cbam=False)
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

class OracleClassifier:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.idx_to_class = None
        self.img_size = 224
        self.classnames = self._load_classnames()
        self.has_model = False
    
    def _load_classnames(self):
        json_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'classnames.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('idx_to_class', {})
        return {}
    
    def load_model(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Model file not found: {checkpoint_path}")
            print("Running in demo mode...")
            self.has_model = False
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                if 'idx_to_class' in checkpoint:
                    self.idx_to_class = checkpoint['idx_to_class']
                if 'config' in checkpoint and 'img_size' in checkpoint['config']:
                    self.img_size = checkpoint['config']['img_size']
                    if isinstance(self.img_size, tuple):
                        self.img_size = self.img_size[0]
                use_cbam = False
                use_edge_enhance = False
                if 'ablation' in checkpoint:
                    use_cbam = bool(checkpoint['ablation'].get('use_cbam', False))
                    use_edge_enhance = bool(checkpoint['ablation'].get('use_edge_enhance', False))
            else:
                model_state = checkpoint
                if self.classnames:
                    self.idx_to_class = {int(k): v for k, v in self.classnames.items()}
                use_cbam = False
                use_edge_enhance = True
            
            if self.idx_to_class is None:
                raise RuntimeError("Cannot determine idx_to_class mapping")
            
            num_classes = len(self.idx_to_class)
            self.model = OracleResNet18(num_classes=num_classes, use_cbam=use_cbam, use_edge_enhance=use_edge_enhance)
            self.model.load_state_dict(model_state, strict=False)
            self.model.to(self.device).eval()
            self.has_model = True
            print(f"Model loaded successfully! Device: {self.device}, Classes: {num_classes}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.has_model = False
            return False
    
    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        return transform(image.convert('RGB')).unsqueeze(0)
    
    def predict(self, image_path=None, image_data=None):
        if not self.has_model:
            return {
                'prediction': '暂无模型',
                'confidence': 0.0,
                'top_predictions': [
                    {'rank': 1, 'class': '演示模式', 'confidence': 0.0},
                    {'rank': 2, 'class': '请上传模型', 'confidence': 0.0},
                    {'rank': 3, 'class': 'best_model.pth', 'confidence': 0.0}
                ]
            }
        
        try:
            if image_path:
                img = Image.open(image_path).convert('RGB')
            elif image_data:
                img = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            else:
                return None
            
            x = self.preprocess_image(img).to(self.device)
            
            with torch.no_grad():
                out = self.model(x)
                probs = F.softmax(out, dim=1)
                top_k, indices = torch.topk(probs, min(5, probs.size(1)))
                top_predictions = []
                for i, (prob, idx) in enumerate(zip(top_k[0], indices[0])):
                    class_name = self.idx_to_class.get(int(idx.item()), 'UNKNOWN')
                    top_predictions.append({
                        'rank': i + 1,
                        'class': class_name,
                        'confidence': round(float(prob.item()) * 100, 2)
                    })
                
                pred_idx = indices[0, 0].item()
                pred_class = self.idx_to_class.get(pred_idx, 'UNKNOWN')
                confidence = float(probs[0, pred_idx].item())
            
            return {
                'prediction': pred_class,
                'confidence': round(confidence * 100, 2),
                'top_predictions': top_predictions
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'prediction': '识别失败',
                'confidence': 0.0,
                'top_predictions': [{'rank': 1, 'class': '错误', 'confidence': 0.0}]
            }

classifier = OracleClassifier()

# 尝试加载模型，但即使失败也继续运行
classifier.load_model()

# 存储用户上传的模型
user_models = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    print('接收到模型上传请求')
    print('请求文件:', list(request.files.keys()))
    print('请求表单:', list(request.form.keys()))
    
    if 'model' not in request.files:
        print('错误: 没有提供模型文件')
        return jsonify({'success': False, 'error': 'No model file provided'}), 400
    
    file = request.files['model']
    print('文件名:', file.filename)
    print('文件类型:', file.content_type)
    print('文件大小:', file.content_length)
    
    if file.filename == '':
        print('错误: 空文件名')
        return jsonify({'success': False, 'error': 'Empty filename'}), 400
    
    if not file.filename.endswith('.pth'):
        print('错误: 不是 .pth 文件')
        return jsonify({'success': False, 'error': 'Only .pth files are allowed'}), 400
    
    try:
        model_id = f"model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_path = os.path.join(app.config['MODEL_FOLDER'], f"{model_id}.pth")
        print('保存路径:', model_path)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        file.save(model_path)
        print('文件保存成功')
        
        # 测试模型是否可以加载
        print('开始加载模型...')
        temp_classifier = OracleClassifier()
        if temp_classifier.load_model(model_path):
            user_models[model_id] = {
                'path': model_path,
                'name': file.filename,
                'classifier': temp_classifier
            }
            print('模型加载成功，添加到用户模型列表')
            return jsonify({'success': True, 'model_id': model_id})
        else:
            os.remove(model_path)
            print('错误: 模型加载失败')
            return jsonify({'success': False, 'error': 'Failed to load model'}), 400
    except Exception as e:
        print('错误:', str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    if 'image' not in request.files and 'image_data' not in request.form:
        return jsonify({'error': 'No image provided'}), 400
    
    image_path = None
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400
            
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
        
        # 检查是否使用用户上传的模型
        model_id = request.form.get('model_id')
        if model_id and model_id in user_models:
            result = user_models[model_id]['classifier'].predict(image_path=image_path, image_data=request.form.get('image_data'))
        else:
            result = classifier.predict(image_path=image_path, image_data=request.form.get('image_data'))
        
        if result is None:
            return jsonify({'error': 'Prediction failed. Please ensure model file exists.'}), 500
        
        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO history (image_path, prediction, confidence, top_predictions) VALUES (?, ?, ?, ?)',
            (image_path or '', result['prediction'], result['confidence'], json.dumps(result['top_predictions']))
        )
        conn.commit()
        result['record_id'] = cursor.lastrowid
        conn.close()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = get_db()
        cursor = conn.execute(
            'SELECT * FROM history ORDER BY created_at DESC LIMIT 50'
        )
        records = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(records)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:record_id>', methods=['DELETE'])
def delete_history(record_id):
    try:
        conn = get_db()
        conn.execute('DELETE FROM history WHERE id = ?', (record_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        conn = get_db()
        conn.execute('DELETE FROM history')
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo/<char_name>', methods=['GET'])
def get_demo_image(char_name):
    demo_dir = os.path.join(os.path.dirname(__file__), '..', '演示数据')
    if os.path.exists(demo_dir):
        for fname in os.listdir(demo_dir):
            if fname.replace('.png', '') == char_name:
                return send_from_directory(demo_dir, fname)
    return jsonify({'error': 'Demo image not found'}), 404

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)