// VisLang Frontend - Full Feature Set
const API_URL = 'http://localhost:8000';

// Global State
let currentImageFile = null;
let currentChatFile = null;
let videoOutputFile = null;
let currentVideoPath = null;

// Segmentation variables
let segmentationImage = null;
let segmentationCanvas = null;
let segmentationCtx = null;
let currentSegPrompts = [];
let lastSegmentationResult = null;

// Detection variables
let detImage = new Image();
let detDetections = [];
let detScale = 1;

// ============ INIT ============
window.addEventListener('load', () => {
    checkStatus();
    setupTabs();
    setupDetection();
    setupChat();
    setupVideo();
    setupSegmentation();
    console.log('✅ VisLang loaded');
});

// ============ STATUS ============
async function checkStatus() {
    try {
        const r = await fetch(`${API_URL}/health`);
        if (r.ok) {
            const statusBar = document.getElementById('statusBar');
            statusBar.innerHTML = '<span class="status-dot" style="background:green"></span> Connected';
            statusBar.style.color = 'green';
        } else {
            document.getElementById('statusBar').innerHTML = '<span class="status-dot" style="background:red"></span> Error';
        }
    } catch (e) {
        document.getElementById('statusBar').innerHTML = '<span class="status-dot" style="background:gray"></span> Offline';
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

// ============ DETECTION & SAM WORKFLOW ============
function setupDetection() {
    const inp = document.getElementById('imageInput');
    const canvas = document.getElementById('detCanvas');

    inp.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        currentImageFile = file;
        
        const reader = new FileReader();
        reader.onload = (evt) => {
            detImage = new Image();
            detImage.onload = () => {
                // Prepare Canvas
                canvas.width = detImage.width;
                canvas.height = detImage.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(detImage, 0, 0);

                // Show/Hide Elements
                document.getElementById('detCanvasContainer').classList.remove('hidden');
                document.getElementById('detectBtn').classList.remove('hidden');
                document.getElementById('results').classList.add('hidden');
                document.getElementById('errorMsg').classList.add('hidden');
                document.getElementById('samPanel').classList.add('hidden');
                
                detDetections = []; // Reset detections
            };
            detImage.src = evt.target.result;
        };
        reader.readAsDataURL(file);
    });

    // Add Click Handler for Detection Canvas
    canvas.addEventListener('click', handleDetectionCanvasClick);
}

async function detectImage() {
    if (!currentImageFile) {
        showError('Select image first');
        return;
    }

    try {
        const btn = document.getElementById('detectBtn');
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Detecting...';
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

        detDetections = data.detections;
        drawDetections(); // Draw bounding boxes on canvas

        document.getElementById('fileName').textContent = data.filename || 'image';
        document.getElementById('objCount').textContent = data.detections.length;
        document.getElementById('results').classList.remove('hidden');
        document.getElementById('detTip').classList.remove('hidden');

    } catch (e) {
        console.error(e);
        showError(e.message);
    } finally {
        const btn = document.getElementById('detectBtn');
        btn.disabled = false;
        btn.innerHTML = 'Run Detection';
        document.getElementById('loadingMsg').classList.add('hidden');
    }
}

function drawDetections(selectedIndex = -1) {
    const canvas = document.getElementById('detCanvas');
    const ctx = canvas.getContext('2d');
    
    // Clear and redraw image
    ctx.drawImage(detImage, 0, 0);

    detDetections.forEach((det, i) => {
        const { x1, y1, x2, y2 } = det.bbox;
        const w = x2 - x1;
        const h = y2 - y1;

        ctx.lineWidth = selectedIndex === i ? 6 : 4;
        ctx.strokeStyle = selectedIndex === i ? '#00FF00' : '#00CCFF';
        
        if (selectedIndex === i) {
            ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
            ctx.fillRect(x1, y1, w, h);
        }
        
        ctx.strokeRect(x1, y1, w, h);

        // Label
        if (i !== selectedIndex) {
            ctx.font = 'bold 20px Arial';
            ctx.fillStyle = '#00CCFF';
            ctx.fillRect(x1, y1 - 25, ctx.measureText(det.class_name).width + 10, 25);
            ctx.fillStyle = 'black';
            ctx.fillText(det.class_name, x1 + 5, y1 - 5);
        }
    });
}

function handleDetectionCanvasClick(e) {
    if (detDetections.length === 0) return;

    const canvas = document.getElementById('detCanvas');
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    // Find clicked box (iterate reverse to find top-most box)
    let clickedIdx = -1;
    for (let i = detDetections.length - 1; i >= 0; i--) {
        const { x1, y1, x2, y2 } = detDetections[i].bbox;
        if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
            clickedIdx = i;
            break;
        }
    }

    if (clickedIdx !== -1) {
        drawDetections(clickedIdx); // Highlight box
        runSAMOnBox(detDetections[clickedIdx]); // Trigger SAM
    }
}

async function runSAMOnBox(detection) {
    const panel = document.getElementById('samPanel');
    const content = document.getElementById('samContent');
    const loading = document.getElementById('samLoading');
    
    panel.classList.remove('hidden');
    content.classList.add('hidden');
    loading.classList.remove('hidden');

    try {
        const { x1, y1, x2, y2 } = detection.bbox;
        const boxArr = [x1, y1, x2, y2];

        const fd = new FormData();
        fd.append('file', currentImageFile);
        fd.append('box', JSON.stringify(boxArr));
        fd.append('class_name', detection.class_name);

        const r = await fetch(`${API_URL}/api/v1/segment/box`, { method: 'POST', body: fd });
        const data = await r.json();

        if (!r.ok) throw new Error(data.error);

        // Update Panel UI Stats
        document.getElementById('samClass').textContent = data.class_name;
        document.getElementById('samArea').textContent = data.pixels + ' px';
        document.getElementById('samConf').textContent = (data.confidence * 100).toFixed(1) + '%';

        // Update Images with Error Checking
        const baseUrl = API_URL;

        // 1. Transparent Object (No BG)
        const noBgImg = document.getElementById('samNoBg');
        if(noBgImg) noBgImg.src = baseUrl + data.object;
        const dlNoBg = document.getElementById('dlNoBg');
        if(dlNoBg) dlNoBg.href = baseUrl + data.object;
        
        // 2. Crop
        const cropImg = document.getElementById('samCrop');
        if (cropImg) cropImg.src = baseUrl + data.crop;
        
        // 3. Mask
        const maskImg = document.getElementById('samMask');
        if (maskImg) maskImg.src = baseUrl + data.mask;

        content.classList.remove('hidden');

    } catch (e) {
        console.error(e);
        alert("SAM Segmentation Failed: " + e.message);
    } finally {
        loading.classList.add('hidden');
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
        const r = await fetch(`${API_URL}/api/v1/chat`, { method: 'POST', body: fd });
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
    document.getElementById('chatLoading').classList.remove('hidden');

    try {
        const fd = new FormData();
        fd.append('file', currentChatFile);
        const r = await fetch(`${API_URL}/api/v1/describe`, { method: 'POST', body: fd });
        const data = await r.json();
        if (!r.ok) throw new Error(data.error || 'Failed');
        addMsg('assistant', data.description);
    } catch (e) {
        console.error(e);
        addMsg('assistant', '❌ ' + (e.message || 'Failed'));
    } finally {
        document.getElementById('chatLoading').classList.add('hidden');
    }
}

function resetChat() {
    document.getElementById('chatBox').innerHTML = '';
    document.getElementById('question').value = '';
    addMsg('assistant', 'Chat cleared!');
}

// ============ VIDEO & SUMMARIZATION ============
function setupVideo() {
    document.getElementById('videoInput').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        document.getElementById('videoName').textContent = file.name;
        document.getElementById('videoInfo').classList.remove('hidden');
        document.getElementById('processBtn').classList.remove('hidden');
        document.getElementById('stats').classList.add('hidden');
        
        // Reset summary UI when new video selected
        document.getElementById('summaryResult').classList.add('hidden');
        currentVideoPath = null;
    });
}

async function processVideo() {
    const file = document.getElementById('videoInput').files[0];
    const thresh = 0.5; // Default

    if (!file) {
        alert('Select video');
        return;
    }

    try {
        const btn = document.getElementById('processBtn');
        btn.disabled = true;
        btn.textContent = '⏳ Starting...';
        document.getElementById('progress').classList.remove('hidden');
        document.getElementById('videoError').classList.add('hidden');

        // 1. Upload
        setProgress(10);
        const uploadFd = new FormData();
        uploadFd.append('file', file);
        const uploadR = await fetch(`${API_URL}/api/v1/video/upload`, { method: 'POST', body: uploadFd });
        if (!uploadR.ok) throw new Error('Upload failed');
        const uploadData = await uploadR.json();
        
        // NEW: Save path for summarizer
        currentVideoPath = uploadData.path; 

        // 2. Process (Detection)
        setProgress(40);
        const procFd = new FormData();
        procFd.append('video_path', uploadData.path);
        procFd.append('conf_threshold', thresh);
        const procR = await fetch(`${API_URL}/api/v1/video/process`, { method: 'POST', body: procFd });
        if (!procR.ok) {
            const err = await procR.json();
            throw new Error(err.error || 'Processing failed');
        }
        const procData = await procR.json();

        // Show results
        document.getElementById('statFrames').textContent = procData.stats.total_frames;
        document.getElementById('statFps').textContent = procData.stats.avg_fps.toFixed(2);
        document.getElementById('statDets').textContent = procData.stats.total_detections;
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
        const btn = document.getElementById('processBtn');
        btn.disabled = false;
        btn.textContent = 'Start Processing';
    }
}

// NEW: Video Summarization Function
async function summarizeVideo() {
    if (!currentVideoPath) {
        alert("Please upload and process a video first.");
        return;
    }

    const btn = document.getElementById('summarizeBtn');
    const loading = document.getElementById('summaryLoading');
    const resultDiv = document.getElementById('summaryResult');
    const textDiv = document.getElementById('summaryText');

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
    loading.classList.remove('hidden');
    resultDiv.classList.add('hidden');

    try {
        const fd = new FormData();
        fd.append('video_path', currentVideoPath);

        const r = await fetch(`${API_URL}/api/v1/video/summarize`, {
            method: 'POST',
            body: fd
        });

        const data = await r.json();

        if (!r.ok) throw new Error(data.error || 'Summarization failed');

        textDiv.innerText = data.summary;
        resultDiv.classList.remove('hidden');

    } catch (e) {
        console.error(e);
        alert("Failed to summarize: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'Generate Summary';
        loading.classList.add('hidden');
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

// ============ MANUAL SEGMENTATION ============
function setupSegmentation() {
    const segInput = document.getElementById('segInput');
    segInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            const img = new Image();
            img.onload = () => {
                segmentationCanvas = document.getElementById('segCanvas');
                segmentationCtx = segmentationCanvas.getContext('2d');
                segmentationCanvas.width = img.width;
                segmentationCanvas.height = img.height;
                segmentationCtx.drawImage(img, 0, 0);
                segmentationImage = img;
                currentSegPrompts = [];
                lastSegmentationResult = null;
                document.getElementById('segPreview').classList.remove('hidden');
                document.getElementById('segControls').classList.remove('hidden');
                document.getElementById('segResults').classList.add('hidden');
                document.getElementById('segStatus').textContent = '✅ Ready for clicks';
                addSegmentationClickHandlers();
            };
            img.src = evt.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function addSegmentationClickHandlers() {
    segmentationCanvas.addEventListener('click', (e) => {
        const rect = segmentationCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (segmentationCanvas.width / rect.width);
        const y = (e.clientY - rect.top) * (segmentationCanvas.height / rect.height);
        addSegmentationPrompt(x, y, 1);
    });
    segmentationCanvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const rect = segmentationCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (segmentationCanvas.width / rect.width);
        const y = (e.clientY - rect.top) * (segmentationCanvas.height / rect.height);
        addSegmentationPrompt(x, y, 0);
    });
}

function addSegmentationPrompt(x, y, label) {
    currentSegPrompts.push([x, y, label]);
    const ctx = segmentationCtx;
    const radius = 8;
    ctx.fillStyle = label === 1 ? 'rgba(0, 255, 0, 0.5)' : 'rgba(255, 0, 0, 0.5)';
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = label === 1 ? 'green' : 'red';
    ctx.lineWidth = 2;
    ctx.stroke();
    document.getElementById('segStatus').textContent = `Added ${currentSegPrompts.length} prompt(s)`;
}

async function runSegmentation() {
    if (!segmentationImage || currentSegPrompts.length === 0) {
        alert('Select image and add at least one click');
        return;
    }
    try {
        document.getElementById('segControls').classList.add('hidden'); // Hide controls during load
        document.getElementById('segStatus').textContent = '⏳ Processing...';
        document.getElementById('segLoading').classList.remove('hidden');
        
        segmentationCanvas.toBlob(async (blob) => {
            try {
                const formData = new FormData();
                formData.append('file', blob, 'image.png');
                const points = currentSegPrompts.map(p => [p[0], p[1]]);
                const labels = currentSegPrompts.map(p => p[2]);
                formData.append('points', JSON.stringify(points));
                formData.append('labels', JSON.stringify(labels));
                
                const response = await fetch(`${API_URL}/api/v1/segment/point`, { method: 'POST', body: formData });
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Segmentation failed');
                }
                const result = await response.json();
                lastSegmentationResult = result;
                
                document.getElementById('segResults').classList.remove('hidden');
                document.getElementById('segStatus').textContent = '✅ Segmentation complete!';
                
                const vizImg = new Image();
                vizImg.onload = () => { segmentationCtx.drawImage(vizImg, 0, 0); };
                vizImg.src = `${API_URL}${result.visualization}`;
                
            } catch (e) {
                alert('Segmentation Error: ' + e.message);
            } finally {
                document.getElementById('segControls').classList.remove('hidden');
                document.getElementById('segLoading').classList.add('hidden');
            }
        }, 'image/png');
    } catch (e) {
        alert('Error: ' + e.message);
    }
}

function clearSegPoints() {
    if (!segmentationImage) return;
    segmentationCtx.drawImage(segmentationImage, 0, 0);
    currentSegPrompts = [];
    document.getElementById('segStatus').textContent = '✅ Clicks cleared';
}

function resetSegmentation() {
    document.getElementById('segInput').value = '';
    segmentationImage = null;
    currentSegPrompts = [];
    lastSegmentationResult = null;
    document.getElementById('segPreview').classList.add('hidden');
    document.getElementById('segControls').classList.add('hidden');
    document.getElementById('segResults').classList.add('hidden');
    document.getElementById('segStatus').textContent = 'Ready for clicks';
}

function downloadSegViz() {
    if (lastSegmentationResult && lastSegmentationResult.visualization) {
        const link = document.createElement('a');
        link.href = `${API_URL}${lastSegmentationResult.visualization}`;
        link.download = 'segmentation_viz.png';
        link.click();
    }
}

function downloadSegMask() {
    if (lastSegmentationResult && lastSegmentationResult.mask) {
        const link = document.createElement('a');
        link.href = `${API_URL}${lastSegmentationResult.mask}`;
        link.download = 'segmentation_mask.png';
        link.click();
    }
}