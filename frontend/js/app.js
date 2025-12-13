// VisLang Frontend - FIXED Segmentation
const API_URL = 'http://localhost:8000';

let currentImageFile = null;
let currentChatFile = null;
let videoOutputFile = null;

// Segmentation variables
let segmentationImage = null;
let segmentationCanvas = null;
let segmentationCtx = null;
let currentSegPrompts = [];
let lastSegmentationResult = null;

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

// ============================================================
// SEGMENTATION (SAM) - MANUAL POINTS + AUTOMATIC
// ============================================================

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
                document.getElementById('promptCount').textContent = '0';
                document.getElementById('segStatus').textContent = '✅ Ready for clicks';
                
                addSegmentationClickHandlers();
                
                console.log(`✅ Image loaded: ${img.width}x${img.height}`);
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
        addSegmentationPrompt(x, y, 1); // Include
    });
    
    segmentationCanvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const rect = segmentationCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (segmentationCanvas.width / rect.width);
        const y = (e.clientY - rect.top) * (segmentationCanvas.height / rect.height);
        addSegmentationPrompt(x, y, 0); // Exclude
    });
}


function addSegmentationPrompt(x, y, label) {
    currentSegPrompts.push([x, y, label]);
    
    const ctx = segmentationCtx;
    const radius = 8;
    
    if (label === 1) {
        ctx.fillStyle = 'rgba(0, 255, 0, 0.5)'; // Green for include
    } else {
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)'; // Red for exclude
    }
    
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.strokeStyle = label === 1 ? 'green' : 'red';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    document.getElementById('promptCount').textContent = currentSegPrompts.length;
    document.getElementById('segStatus').textContent = `Added ${currentSegPrompts.length} prompt(s)`;
}


async function runSegmentation() {
    if (!segmentationImage || currentSegPrompts.length === 0) {
        alert('Select image and add at least one click');
        return;
    }
    
    try {
        document.getElementById('segLoading').classList.remove('hidden');
        document.getElementById('segError').classList.add('hidden');
        document.getElementById('segStatus').textContent = '⏳ Processing...';
        
        // Convert canvas to blob
        segmentationCanvas.toBlob(async (blob) => {
            try {
                const formData = new FormData();
                formData.append('file', blob, 'image.png');
                
                // Create arrays from points
                const points = currentSegPrompts.map(p => [p[0], p[1]]);
                const labels = currentSegPrompts.map(p => p[2]);
                
                // Convert to JSON strings
                const pointsJson = JSON.stringify(points);
                const labelsJson = JSON.stringify(labels);
                
                // Log for debugging
                console.log(`🎯 Sending segmentation request:`);
                console.log(`   Points: ${pointsJson}`);
                console.log(`   Labels: ${labelsJson}`);
                console.log(`   Points count: ${points.length}`);
                console.log(`   Labels count: ${labels.length}`);
                
                // Append as form fields
                formData.append('points', pointsJson);
                formData.append('labels', labelsJson);
                
                // Send request
                const response = await fetch(`${API_URL}/api/v1/segment/point`, {
                    method: 'POST',
                    body: formData
                });
                
                console.log(`Response status: ${response.status}`);
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Segmentation failed');
                }
                
                const result = await response.json();
                lastSegmentationResult = result;
                
                console.log('✅ Segmentation complete', result);
                
                document.getElementById('segConfidence').textContent = (result.confidence * 100).toFixed(2) + '%';
                document.getElementById('segPixels').textContent = result.pixels;
                document.getElementById('segCoverage').textContent = result.coverage_percent.toFixed(2) + '%';
                document.getElementById('segResults').classList.remove('hidden');
                document.getElementById('segStatus').textContent = '✅ Segmentation complete!';
                
                // Show visualization
                const vizImg = new Image();
                vizImg.onload = () => {
                    segmentationCtx.drawImage(vizImg, 0, 0);
                };
                vizImg.onerror = () => {
                    console.error('Failed to load visualization image');
                };
                vizImg.src = `${API_URL}${result.visualization}`;
                
            } catch (e) {
                console.error('Error in blob callback:', e);
                document.getElementById('segError').textContent = '❌ ' + e.message;
                document.getElementById('segError').classList.remove('hidden');
                document.getElementById('segStatus').textContent = '❌ Failed';
            } finally {
                document.getElementById('segLoading').classList.add('hidden');
            }
        }, 'image/png');
        
    } catch (e) {
        console.error('Error:', e);
        document.getElementById('segError').textContent = '❌ ' + e.message;
        document.getElementById('segError').classList.remove('hidden');
        document.getElementById('segLoading').classList.add('hidden');
        document.getElementById('segStatus').textContent = '❌ Failed';
    }
}


function clearSegPoints() {
    if (!segmentationImage) return;
    
    segmentationCtx.drawImage(segmentationImage, 0, 0);
    currentSegPrompts = [];
    document.getElementById('promptCount').textContent = '0';
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
    document.getElementById('segError').classList.add('hidden');
    document.getElementById('segStatus').textContent = 'Ready for clicks';
    document.getElementById('promptCount').textContent = '0';
    
    console.log('🔄 Segmentation reset');
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


// ============================================================
// AUTOMATIC SEGMENTATION
// ============================================================

let autoSegmentationResult = null;
let autoSegmentFile = null;

// Handle file selection for auto segmentation
document.getElementById('autoSegInput').addEventListener('change', function(e) {
    autoSegmentFile = e.target.files[0];
    if (autoSegmentFile) {
        const reader = new FileReader();
        reader.onload = function(event) {
            // Show preview
            document.getElementById('autoSegImage').src = event.target.result;
            document.getElementById('autoSegPreview').classList.remove('hidden');
            document.getElementById('autoSegControls').classList.remove('hidden');
            document.getElementById('autoSegResults').classList.add('hidden');
            document.getElementById('autoSegError').classList.add('hidden');
            
            console.log('Auto segmentation image loaded:', autoSegmentFile.name);
        };
        reader.readAsDataURL(autoSegmentFile);
    }
});

async function runAutoSegmentation() {
    if (!autoSegmentFile) {
        alert('Please select an image first');
        return;
    }
    
    // Show loading
    document.getElementById('autoSegLoading').classList.remove('hidden');
    document.getElementById('autoSegError').classList.add('hidden');
    document.getElementById('autoSegResults').classList.add('hidden');
    
    try {
        const formData = new FormData();
        formData.append('file', autoSegmentFile);
        
        console.log('🤖 Sending automatic segmentation request...');
        
        const response = await fetch(`${API_URL}/api/v1/segment/automatic`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Segmentation result:', data);
        
        // Store result
        autoSegmentationResult = data;
        
        // Update statistics
        document.getElementById('autoObjectCount').textContent = data.objects_found;
        document.getElementById('autoSegPixels').textContent = data.pixels.toLocaleString();
        document.getElementById('autoSegCoverage').textContent = data.coverage_percent.toFixed(2);
        document.getElementById('autoSegConfidence').textContent = (data.average_confidence * 100).toFixed(2);
        
        // Display visualization
        const vizImg = document.getElementById('autoSegVizImage');
        vizImg.src = `${API_URL}${data.visualization}`;
        
        // Show results
        document.getElementById('autoSegResults').classList.remove('hidden');
        document.getElementById('autoSegLoading').classList.add('hidden');
        
    } catch (error) {
        console.error('❌ Error:', error);
        document.getElementById('autoSegError').textContent = 'Error: ' + error.message;
        document.getElementById('autoSegError').classList.remove('hidden');
        document.getElementById('autoSegLoading').classList.add('hidden');
    }
}

function downloadAutoSegViz() {
    if (!autoSegmentationResult) {
        alert('No segmentation result to download');
        return;
    }
    console.log('📥 Downloading visualization...');
    window.location.href = `${API_URL}${autoSegmentationResult.visualization}`;
}

function downloadAutoSegMask() {
    if (!autoSegmentationResult) {
        alert('No segmentation result to download');
        return;
    }
    console.log('📥 Downloading mask...');
    window.location.href = `${API_URL}${autoSegmentationResult.mask}`;
}

function resetAutoSegmentation() {
    // Reset all
    document.getElementById('autoSegInput').value = '';
    document.getElementById('autoSegPreview').classList.add('hidden');
    document.getElementById('autoSegControls').classList.add('hidden');
    document.getElementById('autoSegResults').classList.add('hidden');
    document.getElementById('autoSegError').classList.add('hidden');
    document.getElementById('autoSegLoading').classList.add('hidden');
    
    autoSegmentFile = null;
    autoSegmentationResult = null;
    
    console.log('🔄 Auto segmentation reset');
}

// ============================================================
// END SEGMENTATION SECTION
// ============================================================


console.log('✅ VisLang Frontend Ready!');  