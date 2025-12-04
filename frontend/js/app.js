// VisLang Frontend - Complete & Working
const API_URL = 'http://localhost:8000';

let currentImageFile = null;
let currentChatFile = null;
let videoOutputFile = null;

// ============ INIT ============
window.addEventListener('load', () => {
    checkStatus();
    setupTabs();
    setupDetection();
    setupChat();
    setupVideo();
    console.log('✅ VisLang loaded');
});

// ============ STATUS ============
async function checkStatus() {
    try {
        const r = await fetch(`${API_URL}/health`);
        if (r.ok) {
            document.getElementById('statusBar').textContent = '✅ Connected';
            document.getElementById('statusBar').style.color = 'green';
        } else {
            document.getElementById('statusBar').textContent = '⚠️ Error';
        }
    } catch (e) {
        document.getElementById('statusBar').textContent = '❌ Offline';
        console.error('Backend check failed:', e);
    }
}

// ============ TABS ============
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tab).classList.add('active');
            btn.classList.add('active');
        });
    });
}

// ============ DETECTION ============
function setupDetection() {
    const inp = document.getElementById('imageInput');
    inp.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        currentImageFile = file;
        
        const reader = new FileReader();
        reader.onload = (evt) => {
            document.getElementById('previewImg').src = evt.target.result;
            document.getElementById('imagePreview').classList.remove('hidden');
            document.getElementById('detectBtn').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');
            document.getElementById('errorMsg').classList.add('hidden');
        };
        reader.readAsDataURL(file);
    });
}

async function detectImage() {
    if (!currentImageFile) {
        showError('Select image first');
        return;
    }

    try {
        document.getElementById('detectBtn').disabled = true;
        document.getElementById('detectBtn').textContent = '⏳ Detecting...';
        document.getElementById('loadingMsg').classList.remove('hidden');
        document.getElementById('errorMsg').classList.add('hidden');

        const fd = new FormData();
        fd.append('file', currentImageFile);

        const r = await fetch(`${API_URL}/api/v1/detect`, {
            method: 'POST',
            body: fd
        });

        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Detection failed');

        document.getElementById('fileName').textContent = data.filename || 'image';
        document.getElementById('objCount').textContent = data.detections.length;

        const tbody = document.getElementById('resultsTable');
        tbody.innerHTML = '';
        data.detections.forEach(det => {
            tbody.innerHTML += `<tr>
                <td>${det.class_name}</td>
                <td>${(det.confidence * 100).toFixed(1)}%</td>
                <td>(${det.bbox.x1.toFixed(0)}, ${det.bbox.y1.toFixed(0)}) → (${det.bbox.x2.toFixed(0)}, ${det.bbox.y2.toFixed(0)})</td>
            </tr>`;
        });

        document.getElementById('results').classList.remove('hidden');

    } catch (e) {
        console.error(e);
        showError(e.message);
    } finally {
        document.getElementById('detectBtn').disabled = false;
        document.getElementById('detectBtn').textContent = '🚀 Detect Objects';
        document.getElementById('loadingMsg').classList.add('hidden');
    }
}

function showError(msg) {
    const err = document.getElementById('errorMsg');
    err.textContent = '❌ ' + msg;
    err.classList.remove('hidden');
}

// ============ CHAT ============
function setupChat() {
    const inp = document.getElementById('chatInput');
    inp.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        currentChatFile = file;
        
        const reader = new FileReader();
        reader.onload = (evt) => {
            document.getElementById('chatImg').src = evt.target.result;
            document.getElementById('chatPreview').classList.remove('hidden');
            document.getElementById('chatUI').classList.remove('hidden');
            document.getElementById('chatBox').innerHTML = '';
            addMsg('assistant', 'Ready! Ask about the image.');
        };
        reader.readAsDataURL(file);
    });
}

function addMsg(sender, text) {
    const box = document.getElementById('chatBox');
    const div = document.createElement('div');
    div.className = `chat-message ${sender}`;
    div.innerHTML = `<strong>${sender === 'user' ? 'You' : 'AI'}:</strong> ${text}`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}

async function sendChat() {
    const q = document.getElementById('question').value.trim();
    if (!q || !currentChatFile) {
        alert('Select image & ask question');
        return;
    }

    addMsg('user', q);
    document.getElementById('question').value = '';
    document.getElementById('chatLoading').classList.remove('hidden');

    try {
        const fd = new FormData();
        fd.append('file', currentChatFile);
        fd.append('question', q);

        const r = await fetch(`${API_URL}/api/v1/chat`, {
            method: 'POST',
            body: fd
        });

        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Chat failed');

        addMsg('assistant', data.answer);

    } catch (e) {
        console.error(e);
        addMsg('assistant', '❌ ' + (e.message || 'Failed'));
    } finally {
        document.getElementById('chatLoading').classList.add('hidden');
    }
}

async function describeImg() {
    if (!currentChatFile) {
        alert('Select image first');
        return;
    }

    addMsg('user', 'Describe this image');
    const btn = document.querySelectorAll('.btn-primary')[1];
    btn.disabled = true;
    btn.textContent = 'Generating...';
    document.getElementById('chatLoading').classList.remove('hidden');

    try {
        const fd = new FormData();
        fd.append('file', currentChatFile);

        const r = await fetch(`${API_URL}/api/v1/describe`, {
            method: 'POST',
            body: fd
        });

        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Failed');

        addMsg('assistant', data.description);

    } catch (e) {
        console.error(e);
        addMsg('assistant', '❌ ' + (e.message || 'Failed'));
    } finally {
        btn.disabled = false;
        btn.textContent = '📝 Describe';
        document.getElementById('chatLoading').classList.add('hidden');
    }
}

function resetChat() {
    document.getElementById('chatBox').innerHTML = '';
    document.getElementById('question').value = '';
    addMsg('assistant', 'Chat cleared!');
}

// ============ VIDEO ============
function setupVideo() {
    document.getElementById('videoInput').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        document.getElementById('videoName').textContent = file.name;
        document.getElementById('videoSize').textContent = (file.size / (1024 * 1024)).toFixed(2);
        document.getElementById('videoInfo').classList.remove('hidden');
        document.getElementById('processBtn').classList.remove('hidden');
        document.getElementById('stats').classList.add('hidden');
    });
}

function updateThresh() {
    const val = document.getElementById('threshold').value;
    document.getElementById('threshDisplay').textContent = parseFloat(val).toFixed(2);
}

async function processVideo() {
    const file = document.getElementById('videoInput').files[0];
    const thresh = parseFloat(document.getElementById('threshold').value);

    if (!file) {
        alert('Select video');
        return;
    }

    try {
        document.getElementById('processBtn').disabled = true;
        document.getElementById('processBtn').textContent = '⏳ Starting...';
        document.getElementById('progress').classList.remove('hidden');
        document.getElementById('videoError').classList.add('hidden');

        // Upload
        setProgress(10);
        const uploadFd = new FormData();
        uploadFd.append('file', file);

        const uploadR = await fetch(`${API_URL}/api/v1/video/upload`, {
            method: 'POST',
            body: uploadFd
        });

        if (!uploadR.ok) throw new Error('Upload failed');
        const uploadData = await uploadR.json();

        // Process
        setProgress(40);
        const procFd = new FormData();
        procFd.append('video_path', uploadData.path);
        procFd.append('conf_threshold', thresh);

        const procR = await fetch(`${API_URL}/api/v1/video/process`, {
            method: 'POST',
            body: procFd
        });

        if (!procR.ok) {
            const err = await procR.json();
            throw new Error(err.error || 'Processing failed');
        }

        const procData = await procR.json();

        // Show results
        document.getElementById('statFrames').textContent = procData.stats.total_frames;
        document.getElementById('statFps').textContent = procData.stats.avg_fps.toFixed(2);
        document.getElementById('statDets').textContent = procData.stats.total_detections;

        const dur = procData.stats.video_info.duration;
        const m = Math.floor(dur / 60);
        const s = Math.floor(dur % 60);
        document.getElementById('statDur').textContent = `${m}m ${s}s`;

        videoOutputFile = procData.output_file;
        setProgress(100);
        document.getElementById('stats').classList.remove('hidden');

    } catch (e) {
        console.error(e);
        const err = document.getElementById('videoError');
        err.textContent = '❌ ' + (e.message || 'Error');
        err.classList.remove('hidden');
        document.getElementById('progress').classList.add('hidden');
    } finally {
        document.getElementById('processBtn').disabled = false;
        document.getElementById('processBtn').textContent = '🚀 Process';
    }
}

function setProgress(pct) {
    document.getElementById('progressBar').style.width = pct + '%';
    document.getElementById('progressPct').textContent = pct;
}

function downloadVideo() {
    if (videoOutputFile) {
        window.location.href = `${API_URL}/api/v1/video/download?file=${videoOutputFile}`;
    } else {
        alert('No video');
    }
}

console.log('✅ VisLang Frontend Ready!');