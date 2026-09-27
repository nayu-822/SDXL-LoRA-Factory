const appState = {
    datasetPath: '',
    gdriveDatasetPath: '',
    jobId: '',
    recommendation: null,
    socket: null,
    reconnectTimer: null,
    pollingTimer: null,
    captionSyncTimer: null,
    isRunPod: false,
};

const storagePrefix = 'sdxl_factory_';

function byId(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = byId(id);
    if (element) element.textContent = value == null || value === '' ? '—' : String(value);
}

function getValue(id, fallback = '') {
    const element = byId(id);
    return element ? element.value : fallback;
}

function getNumber(id, fallback = 0) {
    const value = Number(getValue(id));
    return Number.isFinite(value) ? value : fallback;
}

function getOptionalNumber(id) {
    const raw = getValue(id).trim();
    if (!raw) return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
}

function isChecked(id) {
    return Boolean(byId(id)?.checked);
}

function saveField(id) {
    const element = byId(id);
    if (!element) return;
    localStorage.setItem(storagePrefix + id, element.type === 'checkbox' ? String(element.checked) : element.value);
}

function loadSavedFields() {
    const ids = [
        'gdrive-dataset-path', 'dataset-path', 'model-path', 'vae-path', 'output-dir', 'output-name',
        'training-type',
        'training-trigger-word',
        'epochs', 'repeats', 'batch-size', 'resolution-width', 'resolution-height', 'lora-rank',
        'lora-alpha', 'optimizer-type', 'learning-rate-input', 'unet-lr', 'text-encoder-lr',
        'vram-mode', 'mixed-precision', 'workers', 'gradient-accumulation', 'keep-tokens', 'min-snr-gamma-value',
        'optimizer-args', 'seed', 'sample-prompts', 'sample-negative', 'sample-seed', 'sample-steps',
        'sample-width', 'sample-height', 'sample-sampler', 'sample-every', 'gdrive-output-path',
    ];
    const checkboxes = [
        'tagger-auto-sync', 'gradient-checkpointing', 'flip-aug', 'shuffle-caption', 'min-snr-gamma',
        'optimizer-low-vram', 'sample-at-first', 'auto-sync-output', 'auto-shutdown',
    ];
    ids.forEach(id => {
        const element = byId(id);
        const saved = localStorage.getItem(storagePrefix + id);
        if (element && saved !== null) element.value = saved;
        if (element) element.addEventListener('change', () => saveField(id));
    });
    checkboxes.forEach(id => {
        const element = byId(id);
        const saved = localStorage.getItem(storagePrefix + id);
        if (element && saved !== null) element.checked = saved === 'true';
        if (element) element.addEventListener('change', () => saveField(id));
    });
}

async function api(url, options = {}) {
    const response = await fetch(url, options);
    let data = {};
    try { data = await response.json(); } catch (_) { /* handled below */ }
    if (!response.ok) {
        const detail = typeof data.detail === 'object' ? data.detail.message : data.detail;
        throw new Error(detail || data.message || `HTTP ${response.status}`);
    }
    return data;
}

function jsonOptions(body) {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    };
}

function showError(error) {
    console.error(error);
    alert(`エラー / Error: ${error.message || error}`);
}

function setStatus(id, message, kind = '') {
    const element = byId(id);
    if (!element) return;
    element.textContent = message;
    element.className = `status-box ${kind}`.trim();
}

// Navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const tab = item.dataset.tab;
        document.querySelectorAll('.nav-item').forEach(nav => nav.classList.toggle('active', nav === item));
        document.querySelectorAll('.tab-content').forEach(content => { content.hidden = content.id !== `tab-${tab}`; });
    });
});

async function checkSystemStatus() {
    try {
        const data = await api('/api/system/status');
        appState.isRunPod = data.settings?.platform === 'linux';
        const rclone = data.rclone || {};
        const gdrive = data.gdrive || {};
        const driveDisabled = data.settings?.gdrive_enabled === false || rclone.enabled === false;
        const ready = driveDisabled || (rclone.executable?.available && rclone.config?.exists && !gdrive.error);
        const dot = byId('system-status-dot');
        if (dot) dot.classList.toggle('ready', Boolean(ready));
        setText('system-status-text', driveDisabled ? 'GDriveバックアップ無効' : ready ? 'GDrive接続準備完了' : 'GDrive設定を確認してください');
        setText('system-summary', `${data.settings?.data_root_base || data.settings?.data_root || 'data'} · Model: ${data.models?.models?.length || 0} · rclone: ${driveDisabled ? 'disabled' : ready ? 'ready' : '未設定'}`);
        if (data.gpu) setText('gpu-info-display', `${data.gpu.name} (VRAM: ${data.gpu.memory})`);
        if (data.models) updateModelStatus(data.models);
        if (!ready && gdrive.error) {
            setStatus('dataset-sync-status', `${gdrive.error.code}: ${gdrive.error.message}`, 'error');
        }
    } catch (error) {
        setText('system-status-text', 'Backend接続エラー');
        showError(error);
    }
}

function updateModelStatus(data) {
    const status = byId('model-directory-status');
    if (status) {
        status.textContent = data.error
            ? `${data.model_dir}: ${data.error}`
            : `${data.model_dir} · ${data.models.length} model(s) · 読み取り専用`;
    }
    const select = byId('model-select');
    if (!select) return;
    const manualInput = byId('model-path');
    const browseButton = byId('model-browse-btn');
    if (manualInput) manualInput.disabled = appState.isRunPod;
    if (browseButton) browseButton.disabled = appState.isRunPod;
    if (manualInput && appState.isRunPod) {
        manualInput.placeholder = 'RunPod/Linuxでは上のモデル一覧から選択してください';
    }
    const current = getValue('model-path').trim();
    select.replaceChildren();
    if (!data.models?.length) {
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = data.error || 'モデルがありません（手動パスも入力できます）';
        select.appendChild(empty);
        return;
    }
    const manual = document.createElement('option');
    manual.value = '';
    manual.textContent = 'モデルを選択してください';
    select.appendChild(manual);
    data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.path;
        option.textContent = `${model.name} (${model.relative_path})`;
        select.appendChild(option);
    });
    if (current) select.value = current;
}

async function loadBaseModels() {
    try {
        updateModelStatus(await api('/api/base-models'));
    } catch (error) {
        setText('model-directory-status', error.message);
        showError(error);
    }
}

async function browseFolder() {
    try {
        const data = await api('/api/browse-folder');
        if (data.path) {
            byId('dataset-path').value = data.path;
            saveField('dataset-path');
        }
    } catch (error) {
        showError(error);
    }
}

async function browseFile(inputId) {
    try {
        const data = await api('/api/browse-file');
        if (data.path) {
            byId(inputId).value = data.path;
            saveField(inputId);
        }
    } catch (error) {
        showError(error);
    }
}

async function syncDataset() {
    const gdrivePath = getValue('gdrive-dataset-path').trim();
    if (!gdrivePath) return alert('Google Drive Dataset Pathを入力してください。');
    const button = byId('sync-dataset-btn');
    if (button) { button.disabled = true; button.textContent = '同期中... / Syncing...'; }
    setStatus('dataset-sync-status', 'rclone copyを実行中...', 'running');
    try {
        const data = await api('/api/dataset/sync', jsonOptions({ gdrive_path: gdrivePath }));
        appState.gdriveDatasetPath = data.gdrive_path || gdrivePath;
        appState.datasetPath = data.local_path || '';
        if (appState.datasetPath) {
            byId('dataset-path').value = appState.datasetPath;
            saveField('dataset-path');
        }
        setStatus('dataset-sync-status', `同期開始: ${data.local_path || '待機中'}`, 'running');
        watchStatus();
    } catch (error) {
        setStatus('dataset-sync-status', error.message, 'error');
        showError(error);
    } finally {
        if (button) { button.disabled = false; button.textContent = '☁️ DatasetをPodへ同期'; }
    }
}

async function validatePath() {
    const path = getValue('dataset-path').trim();
    if (!path) return alert('Dataset pathを入力してください。');
    try {
        const data = await api(`/api/validate-dataset?path=${encodeURIComponent(path)}`);
        appState.datasetPath = data.path;
        updateDatasetStats(data);
        await loadDataset(appState.datasetPath);
        document.querySelector('.nav-item[data-tab="tagger"]').click();
    } catch (error) {
        showError(error);
    }
}

function updateDatasetStats(data) {
    setText('image-count', data.image_count);
    setText('caption-count', data.caption_count);
    setText('missing-caption-count', data.missing_caption_count);
    setText('recommendation-image-count', data.image_count);
    setText('active-dataset-summary', `Dataset: ${data.path || appState.datasetPath || '未選択'}`);
}

async function refreshDataset() {
    const path = appState.datasetPath || getValue('dataset-path').trim();
    if (!path) return alert('Dataset pathを入力してください。');
    try {
        await loadDataset(path);
    } catch (error) {
        showError(error);
    }
}

async function loadDataset(path) {
    const container = byId('tag-editor-content');
    if (!container) return;
    container.textContent = 'Datasetを読み込み中...';
    const data = await api(`/api/dataset/images?path=${encodeURIComponent(path)}`);
    appState.datasetPath = data.path || path;
    updateDatasetStats(data);
    container.replaceChildren();
    if (!data.files?.length) {
        const empty = document.createElement('div');
        empty.className = 'card placeholder-card muted';
        empty.textContent = '画像がありません。Dataset同期とパスを確認してください。';
        container.appendChild(empty);
        return;
    }
    data.files.forEach(file => container.appendChild(createTagCard(file)));
    updateTagSuggestions();
}

function createTagChip(tag, category = 'tag-general') {
    const chip = document.createElement('span');
    chip.className = `tag-chip ${category}`;
    chip.dataset.tag = tag;
    chip.append(document.createTextNode(tag));
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'remove-btn';
    remove.textContent = '×';
    remove.addEventListener('click', event => { event.stopPropagation(); removeTag(remove); });
    chip.appendChild(remove);
    return chip;
}

function createTagCard(file) {
    const card = document.createElement('article');
    card.className = 'card image-tag-card';
    const image = document.createElement('img');
    image.className = 'tag-preview-img';
    image.alt = file.name;
    image.loading = 'lazy';
    image.src = `/api/image?path=${encodeURIComponent(file.path)}`;
    const body = document.createElement('div');
    body.className = 'tag-card-body';
    const title = document.createElement('div');
    title.className = 'image-name';
    title.textContent = file.relative_path || file.name;
    const tags = document.createElement('div');
    tags.className = 'tags-container';
    tags.dataset.path = file.path;
    (file.tags || []).forEach(tag => tags.appendChild(createTagChip(tag.name, tag.category)));
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'add-tag-input';
    input.placeholder = '+ Add';
    input.addEventListener('keydown', event => handleTagInput(event, input));
    tags.appendChild(input);
    body.append(title, tags);
    card.append(image, body);
    return card;
}

function removeTag(element) {
    const chip = element.closest('.tag-chip');
    const container = chip?.closest('.tags-container');
    chip?.remove();
    if (container) saveTags(container);
    updateTagSuggestions();
}

function handleTagInput(event, input) {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const values = input.value.split(',').map(value => value.trim()).filter(Boolean);
    values.forEach(value => input.parentElement.insertBefore(createTagChip(value), input));
    input.value = '';
    if (values.length) saveTags(input.parentElement);
    updateTagSuggestions();
}

async function saveTags(container) {
    const path = container.dataset.path;
    const tags = Array.from(container.querySelectorAll('.tag-chip')).map(chip => chip.dataset.tag);
    try {
        await api('/api/dataset/update-tags', jsonOptions({ path, tags }));
        const status = byId('caption-sync-status');
        if (status) status.textContent = 'ローカルcaptionを保存済み / Local caption saved';
        scheduleCaptionBackup();
    } catch (error) {
        showError(error);
    }
}

function updateTagSuggestions() {
    const datalist = byId('tag-suggestions');
    if (!datalist) return;
    const allTags = new Set(Array.from(document.querySelectorAll('.tag-chip')).map(chip => chip.dataset.tag));
    datalist.replaceChildren();
    Array.from(allTags).sort().forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        datalist.appendChild(option);
    });
}

async function batchAddTags() {
    await runBatchTags('append', getValue('batch-tags-input'));
}

async function batchRemoveTags() {
    await runBatchTags('remove', getValue('batch-tags-input'));
}

async function runBatchTags(position, rawTags) {
    const tags = rawTags.split(',').map(tag => tag.trim()).filter(Boolean);
    const path = appState.datasetPath || getValue('dataset-path').trim();
    if (!path || !tags.length) return alert('Dataset pathとタグを入力してください。');
    try {
        const data = await api('/api/dataset/batch-tags', jsonOptions({ path, tags, position }));
        alert(`${data.images_changed} images changed / ${position}`);
        await loadDataset(path);
        scheduleCaptionBackup();
    } catch (error) {
        showError(error);
    }
}

async function batchAddTriggerWords() {
    const path = appState.datasetPath || getValue('dataset-path').trim();
    const trigger = getValue('trigger-word-input').trim() || getValue('batch-tags-input').split(',')[0].trim();
    if (!path || !trigger) return alert('Trigger WordとDataset pathを入力してください。');
    try {
        const data = await api('/api/dataset/trigger-word', jsonOptions({ path, trigger_word: trigger }));
        alert(`Trigger Wordを${data.images_changed} imagesの先頭へ追加しました。`);
        await loadDataset(path);
        scheduleCaptionBackup();
    } catch (error) {
        showError(error);
    }
}

async function runAutoTagging() {
    const path = appState.datasetPath || getValue('dataset-path').trim();
    if (!path) return alert('先にDatasetを開いてください。');
    const overwrite = document.querySelector('input[name="caption-policy"]:checked')?.value === 'overwrite';
    if (overwrite && !window.confirm('既存captionを上書きします。手動編集したタグが失われる可能性があります。続行しますか？')) return;
    const button = byId('run-tagger-btn');
    if (button) { button.disabled = true; button.textContent = 'タグ付け中...'; }
    const progress = byId('tagger-progress-container');
    if (progress) progress.hidden = false;
    try {
        const data = await api('/api/run-tagger', jsonOptions({
            path,
            auto_sync_captions: isChecked('tagger-auto-sync'),
            gdrive_path: appState.gdriveDatasetPath || getValue('gdrive-dataset-path').trim(),
            overwrite_existing_captions: overwrite,
        }));
        setStatus('caption-sync-status', `WD14開始: ${data.image_count} images / ${overwrite ? 'overwrite' : 'skip existing'}`, 'running');
        connectWebSocket();
        watchStatus();
    } catch (error) {
        if (button) { button.disabled = false; button.textContent = '自動タグ付け / Run WD14'; }
        showError(error);
    }
}

async function syncCaptions() {
    const path = appState.datasetPath || getValue('dataset-path').trim();
    const gdrivePath = appState.gdriveDatasetPath || getValue('gdrive-dataset-path').trim();
    if (!path || !gdrivePath) return alert('Dataset pathとGoogle Drive pathを入力してください。');
    try {
        const data = await api('/api/dataset/sync-captions', jsonOptions({ local_dataset_path: path, gdrive_path: gdrivePath }));
        setStatus('caption-sync-status', `caption同期開始: ${data.caption_count || 0} files`, 'running');
        watchStatus();
    } catch (error) {
        setStatus('caption-sync-status', error.message, 'error');
        showError(error);
    }
}

function scheduleCaptionBackup() {
    if (!isChecked('tagger-auto-sync')) return;
    const path = appState.datasetPath || getValue('dataset-path').trim();
    const gdrivePath = appState.gdriveDatasetPath || getValue('gdrive-dataset-path').trim();
    if (!path || !gdrivePath) return;
    if (appState.captionSyncTimer) clearTimeout(appState.captionSyncTimer);
    setStatus('caption-sync-status', 'captionバックアップを準備中...', 'running');
    appState.captionSyncTimer = setTimeout(() => {
        appState.captionSyncTimer = null;
        syncCaptions().catch(showError);
    }, 900);
}

function updateTrainingProgress(progress = {}) {
    const epoch = progress.total_epochs ? `${progress.epoch ?? '—'} / ${progress.total_epochs}` : progress.epoch;
    const step = progress.total_steps ? `${progress.step ?? '—'} / ${progress.total_steps}` : progress.step;
    setText('train-epoch', epoch);
    setText('train-step', step);
    setText('train-loss', progress.loss == null ? null : Number(progress.loss).toFixed(6));
    setText('train-average-loss', progress.average_loss == null ? null : Number(progress.average_loss).toFixed(6));
    setText('train-learning-rate', progress.learning_rate == null ? null : progress.learning_rate);
}

function trainingPayload() {
    const prompts = getValue('sample-prompts').split('\n').map(line => line.trim()).filter(Boolean);
    return {
        path: appState.datasetPath || getValue('dataset-path').trim(),
        model: getValue('model-path').trim(),
        vae: getValue('vae-path').trim(),
        output_dir: getValue('output-dir').trim(),
        name: getValue('output-name').trim(),
        training_type: getValue('training-type', 'character'),
        vram: getValue('vram-mode'),
        epochs: getNumber('epochs', 10),
        lr: getNumber('learning-rate-input', 0.0001),
        rank: getNumber('lora-rank', 16),
        alpha: getNumber('lora-alpha', 16),
        repeats: getNumber('repeats', 10),
        batch_size: getNumber('batch-size', 1),
        resolution_width: getNumber('resolution-width', 1024),
        resolution_height: getNumber('resolution-height', 1024),
        gradient_checkpointing: isChecked('gradient-checkpointing'),
        mixed_precision: getValue('mixed-precision', 'bf16'),
        workers: getNumber('workers', 2),
        gradient_accumulation_steps: getNumber('gradient-accumulation', 1),
        seed: getOptionalNumber('seed'),
        unet_lr: getOptionalNumber('unet-lr'),
        text_encoder_lr: getOptionalNumber('text-encoder-lr'),
        flip_aug: isChecked('flip-aug'),
        shuffle_caption: isChecked('shuffle-caption'),
        trigger_word: getValue('training-trigger-word').trim(),
        keep_tokens: getNumber('keep-tokens', 1),
        min_snr_gamma: isChecked('min-snr-gamma'),
        min_snr_gamma_value: getNumber('min-snr-gamma-value', 5),
        optimizer_type: getValue('optimizer-type', 'AdamW'),
        optimizer_args: getValue('optimizer-args'),
        optimizer_low_vram: isChecked('optimizer-low-vram'),
        sample_prompt: prompts[0] || '',
        sample_prompts: prompts,
        sample_negative_prompt: getValue('sample-negative'),
        sample_seed: getNumber('sample-seed', 12345),
        sample_sampler: getValue('sample-sampler', 'euler_a'),
        sample_steps: getNumber('sample-steps', 20),
        sample_width: getNumber('sample-width', 1024),
        sample_height: getNumber('sample-height', 1024),
        sample_every_n_epochs: getNumber('sample-every', 1),
        sample_at_first: isChecked('sample-at-first'),
        gdrive_dataset_path: appState.gdriveDatasetPath || getValue('gdrive-dataset-path').trim(),
        gdrive_output_path: getValue('gdrive-output-path').trim(),
        gdrive_backup_enabled: isChecked('auto-sync-output'),
        auto_sync_output: isChecked('auto-sync-output'),
        shutdown: isChecked('auto-shutdown'),
    };
}

async function startTraining() {
    const payload = trainingPayload();
    if (!payload.path || !payload.model) return alert('Dataset pathとBase Modelを入力してください。');
    const button = byId('start-train-btn');
    if (button) { button.disabled = true; button.textContent = '学習準備中...'; }
    setText('job-info', '学習ジョブを作成中...');
    try {
        const data = await api('/api/start-training', jsonOptions(payload));
        appState.jobId = data.job_id;
        setStatus('job-info', `Job started: ${data.job_id}`, 'running');
        byId('cancel-train-btn').disabled = false;
        connectWebSocket();
        watchStatus();
    } catch (error) {
        if (button) { button.disabled = false; button.textContent = '🚀 LoRA学習開始 / Start Training'; }
        setStatus('job-info', error.message, 'error');
        showError(error);
    }
}

async function cancelTraining() {
    try {
        await api('/api/cancel/training', { method: 'POST' });
        setStatus('job-info', '停止要求を送信しました。', 'running');
    } catch (error) {
        showError(error);
    }
}

async function syncOutput() {
    const state = appState.lastTraining || {};
    const localPath = state.output_dir;
    const gdrivePath = getValue('gdrive-output-path').trim() || state.gdrive_output_path;
    if (!localPath || !gdrivePath) return alert('学習完了後に出力先が確定します。Google Drive pathも入力してください。');
    try {
        const data = await api('/api/sync-output', jsonOptions({ local_output_path: localPath, gdrive_path: gdrivePath }));
        setStatus('output-sync-status', `output同期開始: ${data.file_count || 0} files`, 'running');
        watchStatus();
    } catch (error) {
        setStatus('output-sync-status', error.message, 'error');
        showError(error);
    }
}

function appendConsole(line, channel = 'train') {
    const output = byId('console-output');
    if (!output) return;
    const item = document.createElement('div');
    item.className = `log-line log-${channel}`;
    item.textContent = line;
    output.appendChild(item);
    while (output.childElementCount > 2000) output.firstElementChild.remove();
    output.scrollTop = output.scrollHeight;
}

function showOutputFiles(files = []) {
    const container = byId('output-files');
    if (!container) return;
    container.replaceChildren();
    if (!files.length) { container.textContent = '学習完了後に表示されます。'; return; }
    files.forEach(file => {
        const row = document.createElement('div');
        row.className = 'file-row';
        row.textContent = file;
        container.appendChild(row);
    });
}

function handleStatus(channel, status) {
    if (channel === 'train') {
        appState.lastTraining = status;
        appState.jobId = status.job_id || appState.jobId;
        updateTrainingProgress(status.progress);
        if (status.status) setStatus('job-info', `${status.status}${status.job_id ? ` · ${status.job_id}` : ''}`, status.status === 'failed' ? 'error' : status.status === 'completed' ? 'success' : 'running');
        showOutputFiles(status.output_files || []);
        const outputSync = status.output_sync || {};
        if (outputSync.enabled) {
            const syncLabel = outputSync.status === 'completed'
                ? 'Output同期完了'
                : outputSync.status === 'failed'
                    ? `Output同期失敗: ${outputSync.error || 'unknown error'}`
                    : 'Output同期中...';
            setStatus('output-sync-status', syncLabel, outputSync.status === 'failed' ? 'error' : outputSync.status === 'completed' ? 'success' : 'running');
        }
        const backup = status.gdrive_backup || {};
        if (backup.enabled) {
            const label = backup.status === 'completed'
                ? `Google Driveバックアップ完了 (${backup.phase || 'finish'})`
                : backup.status === 'failed'
                    ? `Google Driveバックアップ失敗: ${backup.error || 'unknown error'}`
                    : `Google Driveバックアップ中 (${backup.phase || 'start'})`;
            setStatus('gdrive-backup-status', label, backup.status === 'failed' ? 'error' : backup.status === 'completed' ? 'success' : 'running');
        }
        if (['completed', 'failed', 'cancelled'].includes(status.status)) {
            const button = byId('start-train-btn');
            if (button) { button.disabled = false; button.textContent = '🚀 LoRA学習開始 / Start Training'; }
            byId('cancel-train-btn').disabled = true;
            document.title = 'SDXL LoRA Factory';
        }
    } else if (channel === 'tagger') {
        const progress = status.progress || {};
        updateTaggerProgress(progress.current || 0, progress.total || status.image_count || 0);
        const countLabel = `tagged ${status.tagged_count || 0} / skipped ${status.skipped_count || 0} / failed ${status.failed_count || 0}`;
        if (status.status === 'completed') {
            byId('run-tagger-btn').disabled = false;
            byId('run-tagger-btn').textContent = '自動タグ付け / Run WD14';
            setStatus('caption-sync-status', `WD14完了: ${countLabel} (${status.caption_count || 0} captions)`, 'success');
            setTimeout(() => refreshDataset(), 250);
        } else if (status.status === 'failed') {
            byId('run-tagger-btn').disabled = false;
            byId('run-tagger-btn').textContent = '自動タグ付け / Run WD14';
            setStatus('caption-sync-status', `${status.error || 'WD14 failed'} · ${countLabel}`, 'error');
        } else if (status.status === 'cancelled') {
            byId('run-tagger-btn').disabled = false;
            byId('run-tagger-btn').textContent = '自動タグ付け / Run WD14';
            setStatus('caption-sync-status', `WD14停止 · ${countLabel}`, 'error');
        }
    } else if (channel === 'sync') {
        const operation = status.operation;
        const label = status.status === 'completed' ? '同期完了' : status.status === 'failed' ? `同期失敗: ${status.error}` : '同期中...';
        if (operation === 'dataset') {
            setStatus('dataset-sync-status', `${label} ${status.local_path || ''}`, status.status === 'failed' ? 'error' : status.status === 'completed' ? 'success' : 'running');
            if (status.status === 'completed' && status.local_path) {
                appState.datasetPath = status.local_path;
                byId('dataset-path').value = status.local_path;
                refreshDataset().catch(showError);
            }
        } else if (operation === 'captions') {
            setStatus('caption-sync-status', label, status.status === 'failed' ? 'error' : status.status === 'completed' ? 'success' : 'running');
        } else if (operation === 'output') {
            setStatus('output-sync-status', label, status.status === 'failed' ? 'error' : status.status === 'completed' ? 'success' : 'running');
        } else if (operation === 'backup') {
            setStatus('gdrive-backup-status', label, status.status === 'failed' ? 'error' : status.status === 'completed' ? 'success' : 'running');
        }
    }
}

function updateTaggerProgress(current, total) {
    const percent = total ? Math.round(current / total * 100) : 0;
    setText('tagger-progress-text', `タグ付け進捗 / Tagging: ${current} / ${total}`);
    setText('tagger-progress-percent', `${percent}%`);
    const bar = byId('tagger-progress-bar');
    if (bar) bar.style.width = `${percent}%`;
    const wrapper = byId('tagger-progress-container');
    if (wrapper) wrapper.hidden = current >= total && total > 0;
}

function handleEvent(event) {
    if (event.type === 'log') appendConsole(`[${event.channel}] ${event.line}`, event.channel);
    if (event.type === 'progress' && event.channel === 'train') updateTrainingProgress(event.progress);
    if (event.type === 'progress' && event.channel === 'tagger') updateTaggerProgress(event.progress.current, event.progress.total);
    if (event.type === 'status') handleStatus(event.channel, event.status || {});
    if (event.type === 'state' && event.state) Object.entries(event.state).forEach(([channel, status]) => handleStatus(channel, status));
}

function connectWebSocket() {
    if (appState.socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(appState.socket.readyState)) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${window.location.host}/ws/logs`);
    appState.socket = socket;
    socket.onmessage = event => {
        try { handleEvent(JSON.parse(event.data)); } catch (_) { appendConsole(event.data); }
    };
    socket.onclose = () => {
        if (appState.reconnectTimer) return;
        appState.reconnectTimer = setTimeout(() => { appState.reconnectTimer = null; connectWebSocket(); }, 3000);
    };
}

function watchStatus() {
    if (appState.pollingTimer) return;
    appState.pollingTimer = setInterval(async () => {
        try {
            const suffix = appState.jobId ? `?job_id=${encodeURIComponent(appState.jobId)}` : '';
            const data = await api(`/api/status${suffix}`);
            if (data.training) handleStatus('train', data.training);
            if (data.tagger) handleStatus('tagger', data.tagger);
            if (data.sync) handleStatus('sync', data.sync);
            const active = data.processes && Object.values(data.processes).some(item => ['starting', 'running'].includes(item.status));
            if (!active) { clearInterval(appState.pollingTimer); appState.pollingTimer = null; }
        } catch (error) { console.warn('status polling failed', error); }
    }, 2500);
}

async function setupScripts() {
    const button = byId('setup-scripts-btn');
    if (button) { button.disabled = true; button.textContent = 'セットアップ中...'; }
    try {
        const data = await api('/api/setup-scripts', { method: 'POST' });
        setText('setup-progress-text', `セットアップ開始: ${data.run_id}`);
        connectWebSocket();
    } catch (error) {
        if (button) button.disabled = false;
        showError(error);
    }
}

async function calculateRecommendation() {
    const path = appState.datasetPath || getValue('dataset-path').trim();
    let imageCount = getNumber('image-count', 0);
    try {
        if (path) imageCount = (await api(`/api/validate-dataset?path=${encodeURIComponent(path)}`)).image_count;
        const data = await api('/api/recommendations', jsonOptions({
            image_count: imageCount,
            training_type: getValue('training-type', 'character'),
            vram: getValue('vram-mode'),
            gradient_accumulation_steps: getNumber('gradient-accumulation', 1),
        }));
        appState.recommendation = data;
        const result = byId('recommendation-result');
        if (result) result.textContent = `Epochs ${data.epochs} / Repeats ${data.repeats} / Dim ${data.network_dim} / Alpha ${data.network_alpha} / ${data.optimizer} / LR ${data.learning_rate} / Min SNR ${data.min_snr_gamma} / ${data.estimated_total_steps_label || 'estimated'} steps ${data.estimated_total_steps ?? '—'}`;
        byId('apply-recommendation-btn').hidden = false;
    } catch (error) { showError(error); }
}

function applyRecommendation() {
    const data = appState.recommendation;
    if (!data) return;
    const mappings = { epochs: data.epochs, repeats: data.repeats, 'batch-size': data.batch_size, 'lora-rank': data.network_dim, 'lora-alpha': data.network_alpha, 'optimizer-type': data.optimizer, 'learning-rate-input': data.learning_rate, 'min-snr-gamma-value': data.min_snr_gamma, 'gradient-accumulation': data.gradient_accumulation_steps || 1 };
    Object.entries(mappings).forEach(([id, value]) => { if (byId(id)) { byId(id).value = value; saveField(id); } });
    if (byId('min-snr-gamma')) { byId('min-snr-gamma').checked = true; saveField('min-snr-gamma'); }
    alert('提案をフォームへ適用しました。内容を確認してから学習を開始してください。');
}

document.addEventListener('DOMContentLoaded', () => {
    loadSavedFields();
    connectWebSocket();
    checkSystemStatus();
    loadBaseModels();
    byId('model-select')?.addEventListener('change', event => {
        const value = event.target.value;
        if (value) {
            byId('model-path').value = value;
            saveField('model-path');
        }
    });
    api('/api/check-scripts').then(data => {
        if (!data.exists) setText('setup-progress-text', `sd-scriptsが見つかりません: ${data.entrypoint}`);
    }).catch(console.warn);
    const savedDataset = getValue('dataset-path').trim();
    if (savedDataset) {
        appState.datasetPath = savedDataset;
        refreshDataset().catch(() => { /* stale local path is harmless */ });
    }
});
