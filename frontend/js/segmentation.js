class SegmentationApp {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.image = null;
        this.originalImage = null;
        this.mask = null;
        this.visualizationUrl = null;
        
        this.mode = 'point';
        this.points = [];
        this.labels = [];
        
        this.isDrawing = false;
        this.boxStartX = null;
        this.boxStartY = null;
        this.boxCurrentX = null;
        this.boxCurrentY = null;
        
        this.setupEventListeners();
        this.showStatus('Ready for image upload', 'info');
    }
    
    setupEventListeners() {
        // File upload
        document.getElementById('imageInput').addEventListener('change', (e) => this.handleImageUpload(e));
        document.getElementById('uploadArea').addEventListener('click', () => document.getElementById('imageInput').click());
        
        // Drag and drop
        document.getElementById('uploadArea').addEventListener('dragover', (e) => {
            e.preventDefault();
            document.getElementById('uploadArea').style.borderColor = 'var(--primary-dark)';
        });
        
        document.getElementById('uploadArea').addEventListener('dragleave', () => {
            document.getElementById('uploadArea').style.borderColor = 'var(--primary)';
        });
        
        document.getElementById('uploadArea').addEventListener('drop', (e) => {
            e.preventDefault();
            document.getElementById('uploadArea').style.borderColor = 'var(--primary)';
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('imageInput').files = files;
                this.handleImageUpload({ target: { files } });
            }
        });
        
        // Mode buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.mode = e.target.dataset.mode;
                this.updateInstructions();
            });
        });
        
        // Canvas interactions
        this.canvas.addEventListener('mousedown', (e) => this.handleCanvasMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleCanvasMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleCanvasMouseUp(e));
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Action buttons
        document.getElementById('segmentBtn').addEventListener('click', () => this.segment());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearPrompts());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
        
        // Download buttons
        document.getElementById('downloadViz').addEventListener('click', () => this.downloadVisualization());
        document.getElementById('downloadMask').addEventListener('click', () => this.downloadMask());
        document.getElementById('extractRegion').addEventListener('click', () => this.extractRegion());
    }
    
    handleImageUpload(e) {
        const file = e.target.files;
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.image = img;
                this.drawCanvas();
                this.enableControls();
                this.showStatus(`Image loaded: ${file.name}`, 'success');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    drawCanvas() {
        if (!this.image) return;
        
        this.canvas.width = this.image.width;
        this.canvas.height = this.image.height;
        
        // Draw image
        this.ctx.drawImage(this.image, 0, 0);
        
        // Draw points
        this.points.forEach((point, idx) => {
            const label = this.labels[idx];
            
            if (label === 1) {
                this.ctx.fillStyle = '#00FF00';
                this.ctx.strokeStyle = '#00AA00';
            } else {
                this.ctx.fillStyle = '#FF0000';
                this.ctx.strokeStyle = '#AA0000';
            }
            
            this.ctx.beginPath();
            this.ctx.arc(point, point, 6, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            
            // Draw label text
            this.ctx.fillStyle = label === 1 ? '#00FF00' : '#FF0000';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(label === 1 ? '+' : '-', point - 4, point + 4);
        });
        
        // Draw bounding box preview
        if (this.isDrawing && this.mode === 'box' && this.boxCurrentX && this.boxCurrentY) {
            this.ctx.strokeStyle = '#0099FF';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(
                this.boxStartX,
                this.boxStartY,
                this.boxCurrentX - this.boxStartX,
                this.boxCurrentY - this.boxStartY
            );
        }
    }
    
    handleCanvasMouseDown(e) {
        if (!this.image) return;
        
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        
        if (this.mode === 'point') {
            const label = e.button === 0 ? 1 : (e.button === 2 ? 0 : -1);
            
            if (label !== -1) {
                this.points.push([x, y]);
                this.labels.push(label);
                this.drawCanvas();
                
                const labelText = label === 1 ? 'Include' : 'Exclude';
                this.showStatus(`Point added: ${labelText} (${this.points.length} total)`, 'info');
            }
        } else if (this.mode === 'box') {
            this.isDrawing = true;
            this.boxStartX = x;
            this.boxStartY = y;
        }
    }
    
    handleCanvasMouseMove(e) {
        if (!this.isDrawing || this.mode !== 'box') return;
        
        const rect = this.canvas.getBoundingClientRect();
        this.boxCurrentX = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        this.boxCurrentY = (e.clientY - rect.top) * (this.canvas.height / rect.height);
        
        this.drawCanvas();
    }
    
    handleCanvasMouseUp(e) {
        if (this.mode === 'box' && this.isDrawing) {
            this.isDrawing = false;
            
            const box = [
                Math.min(this.boxStartX, this.boxCurrentX),
                Math.min(this.boxStartY, this.boxCurrentY),
                Math.max(this.boxStartX, this.boxCurrentX),
                Math.max(this.boxStartY, this.boxCurrentY)
            ];
            
            this.showStatus(`Box drawn: [${box.map(v => Math.round(v)).join(', ')}]`, 'info');
            this.drawCanvas();
        }
    }
    
    async segment() {
        if (!this.originalImage) {
            this.showStatus('Please upload an image first', 'error');
            return;
        }
        
        if (this.points.length === 0) {
            this.showStatus('Please click on image to add prompts', 'error');
            return;
        }
        
        this.showLoading(true);
        this.showStatus('Processing...', 'loading');
        
        try {
            const formData = new FormData();
            
            // Convert canvas to blob
            this.canvas.toBlob(async (blob) => {
                formData.append('file', blob, 'image.png');
                formData.append('points', JSON.stringify(this.points));
                formData.append('labels', JSON.stringify(this.labels));
                
                const response = await fetch('/api/segment/point', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Load visualization
                const img = new Image();
                img.onload = () => {
                    this.visualizationUrl = data.visualization;
                    this.mask = data.mask;
                    this.image = img;
                    this.drawCanvas();
                    
                    // Update statistics
                    document.getElementById('statConfidence').textContent = (data.confidence * 100).toFixed(2) + '%';
                    document.getElementById('statPixels').textContent = data.pixels.toLocaleString();
                    document.getElementById('statCoverage').textContent = data.coverage_percent.toFixed(2) + '%';
                    
                    this.enableDownloadButtons();
                    this.showLoading(false);
                    this.showStatus('Segmentation complete! ✅', 'success');
                };
                img.src = data.visualization;
            });
        } catch (error) {
            this.showLoading(false);
            this.showStatus(`Error: ${error.message}`, 'error');
            console.error(error);
        }
    }
    
    downloadVisualization() {
        if (!this.visualizationUrl) return;
        
        const link = document.createElement('a');
        link.href = this.visualizationUrl;
        link.download = `segmentation_${Date.now()}.png`;
        link.click();
        
        this.showStatus('Visualization downloaded', 'success');
    }
    
    downloadMask() {
        if (!this.mask) return;
        
        const link = document.createElement('a');
        link.href = this.mask;
        link.download = `mask_${Date.now()}.png`;
        link.click();
        
        this.showStatus('Mask downloaded', 'success');
    }
    
    extractRegion() {
        if (!this.mask) {
            this.showStatus('Please segment first', 'error');
            return;
        }
        
        const link = document.createElement('a');
        link.href = this.mask;
        link.download = `extracted_${Date.now()}.png`;
        link.click();
        
        this.showStatus('Region extracted and downloaded', 'success');
    }
    
    clearPrompts() {
        this.points = [];
        this.labels = [];
        this.drawCanvas();
        this.showStatus('Prompts cleared', 'info');
    }
    
    reset() {
        this.image = this.originalImage;
        this.mask = null;
        this.visualizationUrl = null;
        this.points = [];
        this.labels = [];
        this.drawCanvas();
        
        document.getElementById('statConfidence').textContent = '-';
        document.getElementById('statPixels').textContent = '-';
        document.getElementById('statCoverage').textContent = '-';
        
        this.showStatus('Reset complete', 'info');
    }
    
    updateInstructions() {
        const instructionsMap = {
            'point': '<p><strong>Point Mode:</strong></p><ul><li>🟢 Left-click: Include area</li><li>🔴 Right-click: Exclude area</li><li>Multiple clicks to refine</li></ul>',
            'box': '<p><strong>Box Mode:</strong></p><ul><li>Drag to draw bounding box</li><li>Release to confirm</li></ul>'
        };
        
        document.getElementById('instructions').innerHTML = instructionsMap[this.mode];
    }
    
    enableControls() {
        document.getElementById('segmentBtn').disabled = false;
        document.getElementById('clearBtn').disabled = false;
        document.getElementById('resetBtn').disabled = false;
    }
    
    enableDownloadButtons() {
        document.getElementById('downloadViz').disabled = false;
        document.getElementById('downloadMask').disabled = false;
        document.getElementById('extractRegion').disabled = false;
    }
    
    showLoading(show) {
        document.getElementById('loading').classList.toggle('hidden', !show);
    }
    
    showStatus(message, type) {
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = message;
        statusDiv.className = `status ${type}`;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new SegmentationApp();
});
