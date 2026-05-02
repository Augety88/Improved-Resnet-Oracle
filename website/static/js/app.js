const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

const uploadArea = $('#uploadArea');
const uploadPlaceholder = $('#uploadPlaceholder');
const previewImage = $('#previewImage');
const fileInput = $('#fileInput');
const selectBtn = $('#selectBtn');
const recognizeBtn = $('#recognizeBtn');
const clearBtn = $('#clearBtn');
const resultArea = $('#resultArea');
const historyList = $('#historyList');
const clearHistoryBtn = $('#clearHistoryBtn');
const loadingModal = $('#loadingModal');
const toast = $('#toast');
const modelInput = $('#modelInput');
const uploadModelBtn = $('#uploadModelBtn');
const modelStatus = $('#modelStatus');
const uploadSection = $('#uploadSection');
const drawSection = $('#drawSection');
const drawCanvas = $('#drawCanvas');
const brushSize = $('#brushSize');
const brushSizeValue = $('#brushSizeValue');
const clearCanvasBtn = $('#clearCanvasBtn');
const undoBtn = $('#undoBtn');
const recognizeCanvasBtn = $('#recognizeCanvasBtn');
const knowledgeSection = $('#knowledgeSection');
const knowledgeContent = $('#knowledgeContent');
const variantsSection = $('#variantsSection');
const variantsGrid = $('#variantsGrid');
const loadMoreVariantsBtn = $('#loadMoreVariantsBtn');
const searchInput = $('#searchInput');
const searchBtn = $('#searchBtn');
const databaseResults = $('#databaseResults');
const databaseStats = $('#databaseStats');
const categoryStrip = $('#categoryStrip');
const historyDetailModal = $('#historyDetailModal');
const historyDetailContent = $('#historyDetailContent');
const aiAssistant = $('#aiAssistant');
const aiToggle = $('#aiToggle');
const aiPanel = $('#aiPanel');
const aiResetBtn = $('#aiResetBtn');
const aiCloseBtn = $('#aiCloseBtn');
const aiStatus = $('#aiStatus');
const aiMessages = $('#aiMessages');
const aiForm = $('#aiForm');
const aiInput = $('#aiInput');
const aiSendBtn = $('#aiSendBtn');
let selectedFile = null;
let currentModel = null;
let currentPrediction = null;
let currentAssistantContext = {};
let assistantHistory = [];
let assistantBusy = false;
let assistantResultRequestId = 0;
let assistantDrag = {
    active: false,
    moved: false,
    pointerId: null,
    offsetX: 0,
    offsetY: 0
};
let currentVariantLimit = 24;
let lastVariants = [];
let ctx = null;
let isDrawing = false;
let brushHistory = [];
let canvasHasInk = false;

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    }[ch]));
}

function showToast(message, type = 'info') {
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    setTimeout(() => {
        toast.className = 'toast';
    }, 2600);
}

function showLoading() {
    loadingModal.classList.add('show');
}

function hideLoading() {
    loadingModal.classList.remove('show');
}

function setAssistantContext(context = {}) {
    currentAssistantContext = {
        prediction: context.prediction || '',
        confidence: context.confidence ?? null,
        knowledge: context.knowledge || {},
        top_predictions: Array.isArray(context.top_predictions) ? context.top_predictions : []
    };
}

function renderAssistantMessage(role, content, meta = '') {
    if (!aiMessages) return;
    const item = document.createElement('article');
    item.className = `ai-message ${role}`;
    item.innerHTML = `
        <div>${escapeHtml(content)}</div>
        ${meta ? `<small>${escapeHtml(meta)}</small>` : ''}
    `;
    aiMessages.appendChild(item);
    aiMessages.scrollTop = aiMessages.scrollHeight;
}

function seedAssistant() {
    if (!aiMessages || aiMessages.dataset.seeded) return;
    renderAssistantMessage(
        'assistant',
        '你好，我可以帮你解释识别结果、比较同字异形，也能回答模型配置和项目使用问题。'
    );
    aiMessages.dataset.seeded = '1';
}

function clampAssistantPosition(left, top) {
    const size = aiToggle?.offsetWidth || 62;
    const margin = 10;
    return {
        left: Math.min(Math.max(margin, left), window.innerWidth - size - margin),
        top: Math.min(Math.max(margin, top), window.innerHeight - size - margin)
    };
}

function readAssistantBubblePosition() {
    try {
        const saved = JSON.parse(localStorage.getItem('oracleAssistantBubblePosition') || 'null');
        if (Number.isFinite(saved?.left) && Number.isFinite(saved?.top)) return saved;
    } catch (error) {
        localStorage.removeItem('oracleAssistantBubblePosition');
    }
    return null;
}

function saveAssistantBubblePosition(position) {
    try {
        localStorage.setItem('oracleAssistantBubblePosition', JSON.stringify(position));
    } catch (error) {
        // Position persistence is a convenience; dragging should keep working without it.
    }
}

function placeAssistant(left, top, persist = false) {
    if (!aiAssistant) return;
    const position = clampAssistantPosition(left, top);
    aiAssistant.style.left = `${position.left}px`;
    aiAssistant.style.top = `${position.top}px`;
    aiAssistant.style.right = 'auto';
    aiAssistant.style.bottom = 'auto';
    if (persist) {
        saveAssistantBubblePosition(position);
    }
    placeAssistantPanel();
}

function placeAssistantPanel() {
    if (!aiAssistant || !aiPanel) return;
    const bubble = aiAssistant.getBoundingClientRect();
    const panelWidth = Math.min(390, window.innerWidth - 32);
    const panelHeight = Math.min(680, window.innerHeight - 116);
    const gap = 14;
    let left = bubble.right - panelWidth;
    let top = bubble.top - panelHeight - gap;

    if (top < 12) {
        top = bubble.bottom + gap;
    }
    left = Math.min(Math.max(12, left), window.innerWidth - panelWidth - 12);
    top = Math.min(Math.max(12, top), window.innerHeight - panelHeight - 12);

    aiPanel.style.left = `${left}px`;
    aiPanel.style.top = `${top}px`;
    aiPanel.style.right = 'auto';
    aiPanel.style.bottom = 'auto';
    aiPanel.style.maxHeight = `${panelHeight}px`;
}

function initAssistantBubble() {
    if (!aiAssistant || !aiToggle) return;
    const saved = readAssistantBubblePosition();
    const defaultLeft = window.innerWidth - (aiToggle.offsetWidth || 62) - 24;
    const defaultTop = window.innerHeight - (aiToggle.offsetHeight || 62) - 24;
    placeAssistant(Number.isFinite(saved?.left) ? saved.left : defaultLeft, Number.isFinite(saved?.top) ? saved.top : defaultTop);
}

function setAssistantOpen(open) {
    if (!aiAssistant) return;
    aiAssistant.classList.toggle('open', open);
    if (open) {
        placeAssistantPanel();
        seedAssistant();
        setTimeout(() => aiInput?.focus(), 80);
    }
}

async function loadAssistantStatus() {
    if (!aiStatus) return;
    try {
        const response = await fetch('/api/assistant/status');
        const data = await response.json();
        if (data.success && data.enabled) {
            const provider = data.provider_name || 'OpenAI-compatible';
            const authText = data.has_api_key ? '已配置密钥' : '未配置密钥';
            const wire = data.wire_api || 'chat_completions';
            aiStatus.textContent = `已连接：${provider} / ${data.model} / ${wire} / ${authText}`;
            aiStatus.className = 'ai-status ready';
        } else if (data.success && data.has_endpoint) {
            const provider = data.provider_name || 'OpenAI-compatible';
            const authText = data.has_api_key ? '已配置密钥' : '未配置密钥';
            const wire = data.wire_api || 'chat_completions';
            aiStatus.textContent = `已配置接口（${provider}，${wire}，${authText}），但还没有配置模型名`;
            aiStatus.className = 'ai-status fallback';
        } else {
            aiStatus.textContent = '未配置在线模型，将使用本地兜底回答';
            aiStatus.className = 'ai-status fallback';
        }
    } catch (error) {
        aiStatus.textContent = 'AI 助手状态读取失败';
        aiStatus.className = 'ai-status error';
    }
}

async function sendAssistantMessage(message) {
    if (!message || assistantBusy) return;
    assistantBusy = true;
    aiSendBtn.disabled = true;
    renderAssistantMessage('user', message);
    aiInput.value = '';
    renderAssistantMessage('assistant pending', '正在思考...');
    try {
        const response = await fetch('/api/assistant/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: assistantHistory,
                context: currentAssistantContext
            })
        });
        const data = await response.json();
        const pending = aiMessages.querySelector('.ai-message.pending');
        if (pending) pending.remove();
        if (!response.ok || !data.success) {
            const errorText = data.error || 'AI 助手请求失败';
            renderAssistantMessage('assistant error', errorText);
            showToast(errorText, 'error');
        } else {
            const meta = data.configured ? data.model : '本地兜底';
            renderAssistantMessage('assistant', data.reply, meta);
            assistantHistory.push({ role: 'user', content: message }, { role: 'assistant', content: data.reply });
            assistantHistory = assistantHistory.slice(-10);
        }
    } catch (error) {
        const pending = aiMessages.querySelector('.ai-message.pending');
        if (pending) pending.remove();
        renderAssistantMessage('assistant error', `AI 助手连接失败：${error.message}`);
    } finally {
        assistantBusy = false;
        aiSendBtn.disabled = false;
        aiInput.focus();
    }
}

function resetAssistantConversation() {
    assistantHistory = [];
    if (aiMessages) {
        aiMessages.innerHTML = '';
        aiMessages.dataset.seeded = '';
        seedAssistant();
    }
}

function updateResultAssistantText(text, state = '') {
    const target = $('#aiResultSummary');
    if (!target) return;
    target.textContent = text;
    target.classList.toggle('loading', state === 'loading');
    target.classList.toggle('error', state === 'error');
}

async function requestResultAssistantSummary(fallbackText = '') {
    const target = $('#aiResultSummary');
    if (!target || !currentAssistantContext.prediction) return;
    const requestId = ++assistantResultRequestId;
    updateResultAssistantText('AI 正在结合识别结果生成说明...', 'loading');
    try {
        const name = currentAssistantContext.knowledge?.display_name || currentAssistantContext.prediction;
        const response = await fetch('/api/assistant/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: `请基于当前识别结果，用一两句话解释为什么它可能是“${name}”，重点说明可观察的字形特征。不要提接口、模型或项目配置。`,
                history: [],
                context: currentAssistantContext
            })
        });
        const data = await response.json();
        if (requestId !== assistantResultRequestId) return;
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'AI 说明生成失败');
        }
        updateResultAssistantText(data.reply || fallbackText || '暂无详细释义。');
    } catch (error) {
        if (requestId !== assistantResultRequestId) return;
        const text = fallbackText || `AI 说明暂不可用：${error.message}`;
        updateResultAssistantText(text, fallbackText ? '' : 'error');
    }
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
    uploadPlaceholder.style.display = 'grid';
    fileInput.value = '';
    selectedFile = null;
    recognizeBtn.disabled = true;
}

function clearKnowledge() {
    knowledgeSection.style.display = 'none';
    knowledgeContent.innerHTML = '';
}

function clearVariants() {
    variantsSection.style.display = 'none';
    variantsGrid.innerHTML = '';
}

function renderKnowledge(knowledge) {
    if (!knowledge || Object.keys(knowledge).length === 0) {
        clearKnowledge();
        return;
    }

    knowledgeContent.innerHTML = `
        <article class="knowledge-card primary">
            <div class="card-kicker">字形含义</div>
            <h3>${escapeHtml(knowledge.display_name || knowledge.char_name)}</h3>
            <p>${escapeHtml(knowledge.meaning || '暂无详细释义。')}</p>
        </article>
        <article class="knowledge-card">
            <div class="card-kicker">类别说明</div>
            <h3>${escapeHtml(knowledge.category || '未分类')}</h3>
            <p>${escapeHtml(knowledge.category_desc || '该字暂未归入明确主题。')}</p>
        </article>
        <article class="knowledge-card">
            <div class="card-kicker">字源观察</div>
            <p>${escapeHtml(knowledge.origin || '可结合不同拓片和摹写样本观察字形差异。')}</p>
        </article>
        <article class="knowledge-card">
            <div class="card-kicker">数据集线索</div>
            <p>${escapeHtml(knowledge.tip || '')}</p>
            <div class="sample-meta">
                <span>总样本 ${knowledge.sample_count || 0}</span>
                <span>训练集 ${knowledge.train_count || 0}</span>
                <span>测试集 ${knowledge.test_count || 0}</span>
            </div>
        </article>
    `;
    knowledgeSection.style.display = 'block';
}

async function fetchKnowledge(charName) {
    try {
        const response = await fetch(`/api/knowledge/${encodeURIComponent(charName)}`);
        const data = await response.json();
        if (data.success) {
            renderKnowledge(data.knowledge);
            if (charName === currentPrediction || data.knowledge?.display_name === currentPrediction) {
                setAssistantContext({
                    prediction: data.knowledge?.display_name || charName,
                    confidence: currentAssistantContext.confidence,
                    knowledge: data.knowledge
                });
            }
            return data.knowledge;
        }
    } catch (error) {
        console.error('读取知识失败', error);
    }
    clearKnowledge();
    return null;
}

async function loadVariants(charName, limit = currentVariantLimit) {
    try {
        const response = await fetch(`/api/variants/${encodeURIComponent(charName)}?limit=${limit}`);
        const data = await response.json();
        if (data.success && data.variants.length > 0) {
            lastVariants = data.variants;
            variantsGrid.innerHTML = data.variants.map((variant, index) => `
                <button class="variant-item" type="button" title="${escapeHtml(variant.folder)} / ${escapeHtml(variant.filename)}">
                    <img src="${variant.path}" alt="${escapeHtml(variant.filename)}" loading="lazy">
                    <span>${index + 1}</span>
                </button>
            `).join('');
            variantsSection.style.display = 'block';
            loadMoreVariantsBtn.style.display = data.variants.length >= limit ? 'inline-flex' : 'none';
            renderInlineSimilar(data.variants);
        } else {
            lastVariants = [];
            clearVariants();
            renderInlineSimilar([]);
        }
    } catch (error) {
        console.error('读取异体字形失败', error);
        clearVariants();
    }
}

function renderInlineSimilar(variants) {
    const inlineGrid = $('#inlineSimilarGrid');
    if (!inlineGrid) return;
    const picks = (variants || []).slice(1, 7);
    if (picks.length === 0) {
        inlineGrid.innerHTML = '<div class="mini-empty">暂无可对比字形</div>';
        return;
    }
    inlineGrid.innerHTML = picks.map((variant, index) => `
        <button class="mini-variant" type="button" onclick="jumpToVariants()" title="${escapeHtml(variant.filename)}">
            <img src="${variant.path}" alt="相似字形 ${index + 1}" loading="lazy">
            <span>${index + 1}</span>
        </button>
    `).join('');
}

function renderResult(data) {
    if (data.error) {
        resultArea.innerHTML = `<div class="error-result">${escapeHtml(data.error)}</div>`;
        clearKnowledge();
        clearVariants();
        return;
    }

    const displayName = data.display_prediction || data.prediction;
    currentPrediction = displayName;
    currentVariantLimit = 24;
    const knowledge = data.knowledge || {};
    const firstImage = knowledge.first_image || '';
    const topList = Array.isArray(data.top_predictions) ? data.top_predictions : [];
    setAssistantContext({
        prediction: displayName,
        confidence: Number(data.confidence || 0).toFixed(2),
        knowledge,
        top_predictions: topList
    });

    resultArea.innerHTML = `
        <div class="result-layout">
            <div class="result-main">
                <div class="result-glyph">
                    ${firstImage ? `<img src="${firstImage}" alt="${escapeHtml(displayName)}">` : `<span>${escapeHtml(displayName)}</span>`}
                </div>
                <div class="result-info">
                    <p class="eyebrow">识别结果 Top-1</p>
                    <div class="result-char">${escapeHtml(displayName)}</div>
                    <div class="result-confidence">置信度 ${Number(data.confidence || 0).toFixed(2)}%</div>
                    <div class="result-bar"><span style="width: ${Math.max(0, Math.min(100, data.confidence || 0))}%"></span></div>
                    <div class="result-knowledge loading" id="aiResultSummary">AI 正在结合识别结果生成说明...</div>
                    ${topList.length > 1 ? `
                        <div class="top-list" aria-label="候选识别结果">
                            ${topList.map((item) => `
                                <div>
                                    <span>${item.rank}. ${escapeHtml(item.display_class || item.class)}</span>
                                    <strong>${Number(item.confidence || 0).toFixed(2)}%</strong>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            </div>
            <aside class="inline-similar">
                <p class="eyebrow">相似字形</p>
                <div id="inlineSimilarGrid" class="inline-similar-grid">
                    <div class="mini-empty">正在读取同字形体...</div>
                </div>
            </aside>
        </div>
        <div class="result-actions">
            <button class="btn btn-primary" type="button" onclick="jumpToVariants()">查看该字其他形体</button>
            <button class="btn btn-secondary" type="button" onclick="jumpToKnowledge()">查看字形解读</button>
            <button class="btn btn-secondary" type="button" onclick="askAssistantAboutResult()">询问 AI 助手</button>
        </div>
    `;

    if (data.knowledge) {
        renderKnowledge(data.knowledge);
        requestResultAssistantSummary(data.knowledge.meaning || '');
    } else {
        fetchKnowledge(displayName).then((freshKnowledge) => {
            if (freshKnowledge) {
                setAssistantContext({
                    prediction: freshKnowledge.display_name || displayName,
                    confidence: currentAssistantContext.confidence,
                    knowledge: freshKnowledge,
                    top_predictions: topList
                });
            }
            requestResultAssistantSummary(freshKnowledge?.meaning || '');
        });
    }
    loadVariants(displayName);
}

function renderDatabaseDetail(knowledge) {
    setAssistantContext({
        prediction: knowledge?.display_name || knowledge?.char_name || currentPrediction || '',
        confidence: null,
        knowledge: knowledge || {},
        top_predictions: []
    });
    resultArea.innerHTML = `
        <div class="result-layout">
            <div class="result-main compact">
                <div class="result-glyph">
                    ${knowledge?.first_image ? `<img src="${knowledge.first_image}" alt="${escapeHtml(knowledge.display_name)}">` : `<span>${escapeHtml(knowledge?.display_name || '')}</span>`}
                </div>
                <div class="result-info">
                    <p class="eyebrow">数据库条目 #${String(knowledge?.id || '').padStart(3, '0')}</p>
                    <div class="result-char">${escapeHtml(knowledge?.display_name || '')}</div>
                    <div class="result-knowledge">${escapeHtml(knowledge?.meaning || '暂无详细释义。')}</div>
                </div>
            </div>
            <aside class="inline-similar">
                <p class="eyebrow">相似字形</p>
                <div id="inlineSimilarGrid" class="inline-similar-grid"></div>
            </aside>
        </div>
    `;
}

function jumpToVariants() {
    variantsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function jumpToKnowledge() {
    knowledgeSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function askAssistantAboutResult() {
    setAssistantOpen(true);
    const name = currentAssistantContext.knowledge?.display_name || currentAssistantContext.prediction || '这个识别结果';
    if (aiInput) {
        aiInput.value = `请解释一下${name}的字形含义，并说说应该怎样观察它。`;
        aiInput.focus();
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
            showToast('请先选择图片或书写字形', 'error');
            return;
        }
        if (currentModel) {
            formData.append('model_id', currentModel.id);
        }
        const response = await fetch('/api/classify', { method: 'POST', body: formData });
        const data = await response.json();
        hideLoading();
        renderResult(data);
        if (data.error) {
            showToast(data.error, 'error');
        } else {
            showToast('识别完成', 'success');
            loadHistory();
        }
    } catch (error) {
        hideLoading();
        showToast(`识别请求失败：${error.message}`, 'error');
    }
}

function handleFileSelect(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showToast('请选择图片文件', 'error');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (event) => setPreviewImage(event.target.result);
    reader.readAsDataURL(file);
}

async function loadDemoImage(charName) {
    showLoading();
    try {
        const response = await fetch(`/api/demo/${charName}`);
        if (!response.ok) throw new Error('没有找到演示样本');
        const blob = await response.blob();
        const file = new File([blob], `${charName}.png`, { type: 'image/png' });
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            setPreviewImage(event.target.result);
            hideLoading();
        };
        reader.readAsDataURL(file);
    } catch (error) {
        hideLoading();
        showToast('演示样本加载失败', 'error');
    }
}

async function uploadModel(file) {
    if (!file) return;
    if (!file.name.endsWith('.pth')) {
        showToast('请选择 .pth 模型文件', 'error');
        return;
    }
    modelStatus.textContent = '当前状态：正在上传并加载模型...';
    modelStatus.className = 'model-status loading';
    try {
        const formData = new FormData();
        formData.append('model', file);
        const response = await fetch('/api/upload_model', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.success) {
            currentModel = { id: data.model_id, name: file.name };
            modelStatus.textContent = `当前状态：已加载 ${file.name}`;
            modelStatus.className = 'model-status success';
            showToast('模型上传成功', 'success');
        } else {
            modelStatus.textContent = `当前状态：模型加载失败，${data.error}`;
            modelStatus.className = 'model-status error';
            showToast(data.error, 'error');
        }
    } catch (error) {
        modelStatus.textContent = `当前状态：上传失败，${error.message}`;
        modelStatus.className = 'model-status error';
    }
}

function initCanvas() {
    if (!drawCanvas || ctx) return;
    ctx = drawCanvas.getContext('2d');
    resizeCanvas();
    ctx.strokeStyle = '#2f2118';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Number(brushSize.value);
    clearCanvas();

    drawCanvas.addEventListener('mousedown', startDrawing);
    drawCanvas.addEventListener('mousemove', draw);
    drawCanvas.addEventListener('mouseup', stopDrawing);
    drawCanvas.addEventListener('mouseleave', stopDrawing);
    drawCanvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    drawCanvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    drawCanvas.addEventListener('touchend', stopDrawing);
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    if (!drawCanvas) return;
    const rect = drawCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const snapshot = ctx ? drawCanvas.toDataURL() : null;
    drawCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
    drawCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = '#2f2118';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Number(brushSize.value);
    if (snapshot) {
        const img = new Image();
        img.onload = () => ctx.drawImage(img, 0, 0, rect.width, rect.height);
        img.src = snapshot;
    }
}

function getCanvasPoint(event) {
    const rect = drawCanvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function saveState() {
    brushHistory.push(drawCanvas.toDataURL());
    if (brushHistory.length > 24) brushHistory.shift();
}

function startDrawing(event) {
    isDrawing = true;
    saveState();
    const point = getCanvasPoint(event);
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
}

function draw(event) {
    if (!isDrawing) return;
    const point = getCanvasPoint(event);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    canvasHasInk = true;
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.closePath();
}

function handleTouchStart(event) {
    event.preventDefault();
    startDrawing(event.touches[0]);
}

function handleTouchMove(event) {
    event.preventDefault();
    draw(event.touches[0]);
}

function clearCanvas() {
    if (!ctx) return;
    const rect = drawCanvas.getBoundingClientRect();
    ctx.fillStyle = '#f7ead0';
    ctx.fillRect(0, 0, rect.width, rect.height);
    brushHistory = [];
    canvasHasInk = false;
}

function undoCanvas() {
    if (!ctx || brushHistory.length === 0) return;
    const snapshot = brushHistory.pop();
    const img = new Image();
    const rect = drawCanvas.getBoundingClientRect();
    img.onload = () => {
        ctx.clearRect(0, 0, rect.width, rect.height);
        ctx.drawImage(img, 0, 0, rect.width, rect.height);
        canvasHasInk = brushHistory.length > 0;
    };
    img.src = snapshot;
}

function recognizeCanvas() {
    if (!canvasHasInk) {
        showToast('请先在画板上书写字形', 'error');
        return;
    }
    const rect = drawCanvas.getBoundingClientRect();
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 224;
    tempCanvas.height = 224;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.fillStyle = '#ffffff';
    tempCtx.fillRect(0, 0, 224, 224);
    tempCtx.drawImage(drawCanvas, 0, 0, rect.width, rect.height, 0, 0, 224, 224);
    recognizeImage(tempCanvas.toDataURL('image/png').split(',')[1]);
}

function switchTab(tabName) {
    $$('.tab-btn').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === tabName));
    uploadSection.style.display = tabName === 'upload' ? 'block' : 'none';
    drawSection.style.display = tabName === 'draw' ? 'block' : 'none';
    if (tabName === 'draw') {
        initCanvas();
        setTimeout(resizeCanvas, 50);
    }
}

async function loadDatabaseStats() {
    try {
        const [statsResponse, categoriesResponse] = await Promise.all([
            fetch('/api/database/stats'),
            fetch('/api/categories')
        ]);
        const stats = await statsResponse.json();
        const categories = await categoriesResponse.json();
        if (stats.success) {
            databaseStats.textContent = `已建立数据库：${stats.characters} 个字类，${stats.variants} 张手写体/拓片样本。`;
        }
        if (categories.success) {
            categoryStrip.innerHTML = categories.categories.map((category) => `
                <button class="category-pill" type="button" data-category="${escapeHtml(category.name)}" title="${escapeHtml(category.description)}">
                    ${escapeHtml(category.name)} <span>${category.count}</span>
                </button>
            `).join('');
            $$('.category-pill').forEach((button) => {
                button.addEventListener('click', () => {
                    searchInput.value = button.dataset.category;
                    searchDatabase(button.dataset.category);
                });
            });
        }
    } catch (error) {
        databaseStats.textContent = '数据库读取失败，请确认数据文件夹存在。';
    }
}

async function searchDatabase(queryOverride = null) {
    const query = (queryOverride ?? searchInput.value).trim();
    if (!query) {
        databaseResults.innerHTML = '<div class="empty-state">请输入字名、类别或编码进行查询。</div>';
        return;
    }
    try {
        const response = await fetch(`/api/characters?q=${encodeURIComponent(query)}&limit=80`);
        const data = await response.json();
        if (!data.success || data.characters.length === 0) {
            databaseResults.innerHTML = '<div class="empty-state">没有找到匹配记录，可以换一个字或类别试试。</div>';
            return;
        }
        databaseResults.innerHTML = `
            <div class="database-char-grid">
                ${data.characters.map((item, index) => `
                    <button class="database-char-item" type="button" data-char="${escapeHtml(item.char_name)}">
                        <em>序号 ${String(item.id || index + 1).padStart(3, '0')}</em>
                        ${item.first_image ? `<img src="${item.first_image}" alt="数据库字形 ${index + 1}" loading="lazy">` : ''}
                        <strong>${escapeHtml(item.display_name)}</strong>
                        <span>${item.sample_count || 0} 张</span>
                    </button>
                `).join('')}
            </div>
        `;
        $$('.database-char-item').forEach((item) => {
            item.addEventListener('click', () => selectDatabaseChar(item.dataset.char));
        });
    } catch (error) {
        databaseResults.innerHTML = '<div class="empty-state">查询失败，请稍后重试。</div>';
    }
}

async function selectDatabaseChar(charName) {
    currentPrediction = charName;
    currentVariantLimit = 24;
    const knowledge = await fetchKnowledge(charName);
    await loadVariants(charName);
    renderDatabaseDetail(knowledge);
    renderInlineSimilar(lastVariants);
    knowledgeSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderHistory(records) {
    if (!records || records.length === 0) {
        historyList.innerHTML = '<div class="empty-state">暂无历史记录</div>';
        return;
    }
    historyList.innerHTML = records.map((record) => `
        <article class="history-item" data-id="${record.id}" data-prediction="${escapeHtml(record.prediction)}">
            ${record.image_path ? `<img src="/${record.image_path}" class="history-image" alt="历史图片">` : ''}
            <strong>${escapeHtml(record.prediction)}</strong>
            <span>置信度 ${Number(record.confidence || 0).toFixed(2)}%</span>
            <small>${new Date(record.created_at).toLocaleString('zh-CN')}</small>
            <button class="history-delete" type="button" data-id="${record.id}">删除</button>
        </article>
    `).join('');
    $$('.history-item').forEach((item) => {
        item.addEventListener('click', (event) => {
            if (event.target.classList.contains('history-delete')) return;
            showHistoryDetail(Number(item.dataset.id), records);
        });
    });
    $$('.history-delete').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            deleteHistoryItem(Number(button.dataset.id));
        });
    });
}

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        renderHistory(await response.json());
    } catch (error) {
        console.error('历史记录加载失败', error);
    }
}

async function showHistoryDetail(recordId, records) {
    const record = records.find((item) => item.id === recordId);
    if (!record) return;
    const knowledge = await fetchKnowledge(record.prediction);
    historyDetailContent.innerHTML = `
        ${record.image_path ? `<img src="/${record.image_path}" class="history-detail-image" alt="识别图片">` : ''}
        <div class="history-detail-prediction">${escapeHtml(record.prediction)}</div>
        <div class="history-detail-confidence">置信度 ${Number(record.confidence || 0).toFixed(2)}%</div>
        <div class="history-detail-time">识别时间：${new Date(record.created_at).toLocaleString('zh-CN')}</div>
        ${knowledge ? `<p>${escapeHtml(knowledge.meaning)}</p><p><strong>类别：</strong>${escapeHtml(knowledge.category)}，${escapeHtml(knowledge.category_desc)}</p>` : ''}
    `;
    historyDetailModal.classList.add('show');
}

function closeHistoryModal() {
    historyDetailModal.classList.remove('show');
}

async function deleteHistoryItem(id) {
    try {
        const response = await fetch(`/api/history/${id}`, { method: 'DELETE' });
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
        const response = await fetch('/api/history/clear', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            showToast('已清空历史记录', 'success');
            loadHistory();
        } else {
            showToast(data.error || '清空失败', 'error');
        }
    } catch (error) {
        showToast('清空失败', 'error');
    }
}

uploadArea.addEventListener('click', () => {
    if (previewImage.style.display === 'none' || !previewImage.style.display) fileInput.click();
});
uploadArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadArea.classList.add('dragover');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFileSelect(event.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (event) => handleFileSelect(event.target.files[0]));
selectBtn.addEventListener('click', () => fileInput.click());
recognizeBtn.addEventListener('click', () => recognizeImage());
clearBtn.addEventListener('click', () => {
    clearPreview();
    clearKnowledge();
    clearVariants();
    currentPrediction = null;
    setAssistantContext({});
    resultArea.innerHTML = '<div class="empty-state">上传图片或在画板上写一个甲骨文字形，识别结果会显示在这里。</div>';
});
$$('.demo-btn').forEach((button) => button.addEventListener('click', () => loadDemoImage(button.dataset.char)));
uploadModelBtn.addEventListener('click', () => modelInput.click());
modelInput.addEventListener('change', (event) => uploadModel(event.target.files[0]));
$$('.tab-btn').forEach((tab) => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));
brushSize.addEventListener('input', () => {
    brushSizeValue.textContent = brushSize.value;
    if (ctx) ctx.lineWidth = Number(brushSize.value);
});
clearCanvasBtn.addEventListener('click', clearCanvas);
undoBtn.addEventListener('click', undoCanvas);
recognizeCanvasBtn.addEventListener('click', recognizeCanvas);
searchBtn.addEventListener('click', () => searchDatabase());
searchInput.addEventListener('keyup', (event) => {
    if (event.key === 'Enter') searchDatabase();
});
loadMoreVariantsBtn.addEventListener('click', () => {
    if (!currentPrediction) return;
    currentVariantLimit += 48;
    loadVariants(currentPrediction, currentVariantLimit);
});
clearHistoryBtn.addEventListener('click', () => {
    if (confirm('确定清空所有历史记录吗？')) clearAllHistory();
});
$('.modal-close').addEventListener('click', closeHistoryModal);
historyDetailModal.addEventListener('click', (event) => {
    if (event.target === historyDetailModal) closeHistoryModal();
});
aiToggle?.addEventListener('pointerdown', (event) => {
    if (!aiAssistant) return;
    const rect = aiAssistant.getBoundingClientRect();
    assistantDrag = {
        active: true,
        moved: false,
        pointerId: event.pointerId,
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top
    };
    aiAssistant.classList.add('dragging');
    aiToggle.setPointerCapture?.(event.pointerId);
});
aiToggle?.addEventListener('pointermove', (event) => {
    if (!assistantDrag.active || assistantDrag.pointerId !== event.pointerId) return;
    const left = event.clientX - assistantDrag.offsetX;
    const top = event.clientY - assistantDrag.offsetY;
    const current = aiAssistant.getBoundingClientRect();
    if (Math.abs(current.left - left) > 2 || Math.abs(current.top - top) > 2) {
        assistantDrag.moved = true;
    }
    placeAssistant(left, top);
});
aiToggle?.addEventListener('pointerup', (event) => {
    if (!assistantDrag.active || assistantDrag.pointerId !== event.pointerId) return;
    const wasMoved = assistantDrag.moved;
    assistantDrag.active = false;
    aiAssistant?.classList.remove('dragging');
    aiToggle.releasePointerCapture?.(event.pointerId);
    const rect = aiAssistant.getBoundingClientRect();
    placeAssistant(rect.left, rect.top, true);
    if (!wasMoved) setAssistantOpen(!aiAssistant.classList.contains('open'));
});
aiToggle?.addEventListener('pointercancel', () => {
    assistantDrag.active = false;
    aiAssistant?.classList.remove('dragging');
});
aiResetBtn?.addEventListener('click', resetAssistantConversation);
aiCloseBtn?.addEventListener('click', () => setAssistantOpen(false));
aiForm?.addEventListener('submit', (event) => {
    event.preventDefault();
    sendAssistantMessage(aiInput.value.trim());
});
aiInput?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendAssistantMessage(aiInput.value.trim());
    }
});
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closeHistoryModal();
        setAssistantOpen(false);
    }
});

document.addEventListener('DOMContentLoaded', () => {
    initAssistantBubble();
    initCanvas();
    loadHistory();
    loadDatabaseStats();
    loadAssistantStatus();
});
window.addEventListener('resize', () => {
    if (!aiAssistant) return;
    const rect = aiAssistant.getBoundingClientRect();
    placeAssistant(rect.left, rect.top, true);
});
