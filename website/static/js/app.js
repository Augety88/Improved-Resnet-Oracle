const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewImage = document.getElementById('previewImage');
const fileInput = document.getElementById('fileInput');
const selectBtn = document.getElementById('selectBtn');
const recognizeBtn = document.getElementById('recognizeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultArea = document.getElementById('resultArea');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const loadingModal = document.getElementById('loadingModal');
const toast = document.getElementById('toast');
const demoButtons = document.querySelectorAll('.demo-btn');
const modelInput = document.getElementById('modelInput');
const uploadModelBtn = document.getElementById('uploadModelBtn');
const modelStatus = document.getElementById('modelStatus');

let selectedFile = null;
let currentImageData = null;
let currentModel = null;

function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = 'toast show ' + type;
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

function showLoading() {
    loadingModal.classList.add('show');
}

function hideLoading() {
    loadingModal.classList.remove('show');
}

function setPreviewImage(src) {
    previewImage.src = src;
    previewImage.style.display = 'block';
    uploadPlaceholder.style.display = 'none';
    recognizeBtn.disabled = false;
}

function clearPreview() {
    previewImage.src = '';
    previewImage.style.display = 'none';
    uploadPlaceholder.style.display = 'block';
    fileInput.value = '';
    selectedFile = null;
    currentImageData = null;
    recognizeBtn.disabled = true;
}

function renderResult(data) {
    if (data.error) {
        resultArea.innerHTML = `
            <div class="error-result">
                <p>❌ ${data.error}</p>
            </div>
        `;
        return;
    }

    const topPredictionsHTML = data.top_predictions.map((pred, index) => `
        <div class="top-prediction">
            <span class="top-rank rank-${index + 1}">${index + 1}</span>
            <span class="top-class">${pred.class}</span>
            <span class="top-confidence">${pred.confidence}%</span>
        </div>
    `).join('');

    resultArea.innerHTML = `
        <div class="result-main">
            <div class="result-char">🔮</div>
            <div class="result-label">${data.prediction}</div>
            <div class="result-confidence">置信度: ${data.confidence}%</div>
            <div class="result-bar">
                <div class="result-bar-fill" style="width: ${data.confidence}%"></div>
            </div>
            <div class="result-top">
                <h4>Top 5 预测结果</h4>
                ${topPredictionsHTML}
            </div>
        </div>
    `;
}

function renderHistory(records) {
    if (!records || records.length === 0) {
        historyList.innerHTML = '<p class="history-placeholder">暂无历史记录</p>';
        return;
    }

    historyList.innerHTML = records.map(record => {
        const time = new Date(record.created_at).toLocaleString('zh-CN');
        const imageHTML = record.image_path 
            ? `<img src="/${record.image_path}" class="history-image" alt="历史图像">`
            : '';
        
        let topPredictions = [];
        try {
            topPredictions = JSON.parse(record.top_predictions || '[]');
        } catch (e) {}

        return `
            <div class="history-item" data-id="${record.id}">
                ${imageHTML}
                <div class="history-prediction">${record.prediction}</div>
                <div class="history-confidence">置信度: ${record.confidence}%</div>
                ${topPredictions.length > 1 ? `
                    <div style="margin-top: 8px; font-size: 0.8rem; color: #888;">
                        其他: ${topPredictions.slice(1, 4).map(p => `${p.class}(${p.confidence}%)`).join(', ')}
                    </div>
                ` : ''}
                <div class="history-time">${time}</div>
                <button class="history-delete" onclick="deleteHistoryItem(${record.id}, event)">删除</button>
            </div>
        `;
    }).join('');
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        renderHistory(data);
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

async function deleteHistoryItem(id, event) {
    event.stopPropagation();
    try {
        const response = await fetch(`/api/history/${id}`, {
            method: 'DELETE'
        });
        const data = await response.json();
        if (data.success) {
            showToast('已删除历史记录', 'success');
            loadHistory();
        }
    } catch (error) {
        showToast('删除失败', 'error');
    }
}

async function clearAllHistory() {
    try {
        const response = await fetch('/api/history/clear', {
            method: 'POST'
        });
        const data = await response.json();
        if (data.success) {
            showToast('已清空历史记录', 'success');
            loadHistory();
        }
    } catch (error) {
        showToast('清空失败', 'error');
    }
}

async function recognizeImage(imageData = null) {
    showLoading();
    
    try {
        const formData = new FormData();
        
        if (imageData) {
            formData.append('image_data', imageData);
        } else if (selectedFile) {
            formData.append('image', selectedFile);
        } else {
            hideLoading();
            showToast('请先选择图像', 'error');
            return;
        }

        // 如果有自定义模型，添加模型信息
        if (currentModel) {
            formData.append('model_id', currentModel.id);
        }

        const response = await fetch('/api/classify', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        hideLoading();

        if (data.error) {
            showToast(data.error, 'error');
            renderResult({ error: data.error });
        } else {
            renderResult(data);
            showToast('识别成功！', 'success');
            loadHistory();
        }
    } catch (error) {
        hideLoading();
        showToast('识别请求失败: ' + error.message, 'error');
        console.error('Recognition error:', error);
    }
}

async function uploadModel(file) {
    if (!file) return;
    
    if (!file.name.endsWith('.pth')) {
        showToast('请选择 .pth 模型文件', 'error');
        return;
    }

    modelStatus.className = 'model-status loading';
    modelStatus.textContent = '当前状态：上传中...';

    try {
        const formData = new FormData();
        formData.append('model', file);

        console.log('开始上传模型:', file.name);
        console.log('文件大小:', file.size, 'bytes');
        
        const response = await fetch('/api/upload_model', {
            method: 'POST',
            body: formData
        });

        console.log('响应状态:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('响应数据:', data);
        
        if (data.success) {
            currentModel = {
                id: data.model_id,
                name: file.name
            };
            modelStatus.className = 'model-status success';
            modelStatus.textContent = `当前状态：已加载模型 ${file.name}`;
            showToast('模型上传成功！', 'success');
        } else {
            modelStatus.className = 'model-status error';
            modelStatus.textContent = `当前状态：上传失败 - ${data.error}`;
            showToast('模型上传失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('上传错误详情:', error);
        modelStatus.className = 'model-status error';
        modelStatus.textContent = '当前状态：上传失败 - 网络错误';
        showToast('模型上传失败: ' + error.message, 'error');
    }
}

function handleFileSelect(file) {
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showToast('请选择图像文件', 'error');
        return;
    }

    selectedFile = file;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        currentImageData = e.target.result.split(',')[1];
        setPreviewImage(e.target.result);
    };
    reader.readAsDataURL(file);
}

function loadDemoImage(charName) {
    showLoading();
    
    const demoDir = '演示数据';
    const filename = charName + '.png';
    
    fetch(`/api/demo/${charName}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Demo image not found');
            }
            return response.blob();
        })
        .then(blob => {
            const file = new File([blob], filename, { type: 'image/png' });
            selectedFile = file;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                currentImageData = e.target.result.split(',')[1];
                setPreviewImage(e.target.result);
                hideLoading();
            };
            reader.readAsDataURL(file);
        })
        .catch(error => {
            hideLoading();
            showToast('加载演示样本失败', 'error');
            console.error('Demo loading error:', error);
        });
}

uploadArea.addEventListener('click', () => {
    if (!previewImage.style.display || previewImage.style.display === 'none') {
        fileInput.click();
    }
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
});

selectBtn.addEventListener('click', () => {
    fileInput.click();
});

recognizeBtn.addEventListener('click', () => {
    recognizeImage();
});

clearBtn.addEventListener('click', () => {
    clearPreview();
    resultArea.innerHTML = '<p class="result-placeholder">请上传图像或选择演示样本开始识别</p>';
});

clearHistoryBtn.addEventListener('click', () => {
    if (confirm('确定要清空所有历史记录吗？')) {
        clearAllHistory();
    }
});

demoButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const charName = btn.dataset.char;
        loadDemoImage(charName);
    });
});

uploadModelBtn.addEventListener('click', () => {
    modelInput.click();
});

modelInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        uploadModel(file);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
});