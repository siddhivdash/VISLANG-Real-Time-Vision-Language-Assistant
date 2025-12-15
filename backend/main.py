from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import torch
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from utils.gpu_monitor import GPUMonitor
from config import settings
from models.ollama_vision import get_ollama_engine as get_llava_engine
from models.video_processor import get_video_processor
import threading
import json
from models.sam_engine import SAMSegmentationEngine
import tempfile
import shutil
import time as time_module
import cv2
import time
import traceback
from models.video_summarizer import get_video_summarizer
import asyncio
import sys
import psutil
import uvicorn

load_dotenv()

# ============================================================
# 1. ROBUST PATH SETUP (CRITICAL FIX)
# ============================================================
# Get the absolute path to the backend folder (e.g. /app/backend)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths for data directories
# Docker will map these to the volume mounts defined in docker-compose
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOADS_DIR = settings.UPLOAD_FOLDER 

# Ensure directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

print(f"📂 Server Directories Configured:")
print(f"   - Base: {BASE_DIR}")
print(f"   - Outputs: {OUTPUTS_DIR}")
print(f"   - Uploads: {UPLOADS_DIR}")

# ============================================================
# HELPER: CONVERT NUMPY TO PYTHON NATIVE TYPES
# ============================================================
def to_native(obj):
    """
    Recursively convert NumPy types to Python native types.
    Using .item() handles all numpy scalars (int32, float32, etc.) safely.
    """
    if hasattr(obj, 'tolist'):  # Handles numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Handles numpy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(i) for i in obj]
    return obj

# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="VisLang Backend",
    description="Real-time Multimodal Vision-Language Assistant API",
    version="0.2.3"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MOUNT STATIC FILES (Correctly mapped to OUTPUTS_DIR)
# This tells FastAPI: "When a user asks for /outputs/file.png, look in the ABSOLUTE OUTPUTS_DIR"
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Video processing storage
active_jobs = {}

# Initialize SAM Engine (lazy loading - loads on first request)
sam_engine = None
predictor = None

def get_sam_engine():
    """Lazy load SAM engine"""
    global sam_engine, predictor
    if sam_engine is None:
        print("🚀 Initializing SAM Engine for first time...")
        try:
            sam_engine = SAMSegmentationEngine(model_type="vit_b", device="cpu")
            predictor = sam_engine.predictor
        except Exception as e:
            print(f"❌ Failed to load SAM Engine: {e}")
            raise e
    return sam_engine

# ============================================================
# MODEL LOADING AT STARTUP
# ============================================================

print("="*60)
print("🚀 Loading VisLang Models at Startup")
print("="*60)

# Load YOLOv8
try:
    from ultralytics import YOLO
    # Check for local model file first
    local_yolo = os.path.join(BASE_DIR, "yolov8n.pt")
    if os.path.exists(local_yolo):
        yolo_model = YOLO(local_yolo)
        print(f"✅ YOLOv8 loaded from local file: {local_yolo}")
    else:
        yolo_model = YOLO(settings.YOLO_MODEL)
        print(f"✅ YOLOv8 loaded from settings: {settings.YOLO_MODEL}")
except Exception as e:
    print(f"❌ Failed to load YOLOv8: {e}")
    yolo_model = None

# Load CLIP
try:
    import clip
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, clip_preprocess = clip.load(settings.CLIP_MODEL, device=device)
    print(f"✅ CLIP loaded: {settings.CLIP_MODEL} on {device}")
except Exception as e:
    print(f"❌ Failed to load CLIP: {e}")
    clip_model = None
    clip_preprocess = None

print("✅ Backend models initialized")
print("📌 Ollama (LLaVA) will load on first chat/describe request")
print("📌 Video Processor ready for video uploads")
print("📌 SAM will load on first segmentation request")
print("="*60 + "\n")

# ============================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================

@app.get('/health')
async def health_check():
    """Health check endpoint with system info"""
    gpu_info = GPUMonitor.check_gpu_availability()
    sys_memory = GPUMonitor.get_system_memory_usage()
    
    return JSONResponse(to_native({
        'status': 'healthy',
        'service': 'VisLang Backend',
        'version': '0.2.3',
        'device': settings.DEVICE,
        'gpu': gpu_info,
        'system_memory': sys_memory,
        'models': {
            'yolo': 'loaded' if yolo_model else 'not_loaded',
            'clip': 'loaded' if clip_model else 'not_loaded',
            'ollama_llava': 'lazy-loading',
            'sam': 'lazy-loading',
            'video_processor': 'ready'
        }
    }))

@app.get('/api/v1/status')
async def get_status():
    """Detailed system status endpoint"""
    pytorch_info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': settings.DEVICE
    }
    
    return JSONResponse(to_native({
        'service': 'VisLang Backend',
        'version': '0.2.3',
        'pytorch': pytorch_info,
        'models_loaded': {
            'yolo': yolo_model is not None,
            'clip': clip_model is not None,
            'ollama_llava': 'lazy-loading (loads on first request)',
            'sam': 'lazy-loading (loads on first request)',
            'video_processor': 'ready'
        },
        'features': {
            'image_detection': yolo_model is not None,
            'image_chat': True,
            'video_processing': yolo_model is not None,
            'image_segmentation': True
        }
    }))

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get('/')
async def root():
    """Root endpoint with API documentation"""
    return JSONResponse({
        'service': 'VisLang Backend API','status': 'Online',
        'version': '0.2.3',
        'features': {
            'image_detection': '✅ Object Detection with YOLOv8',
            'image_chat': '✅ Vision-Language Chat with Ollama/LLaVA',
            'video_processing': '✅ Real-Time Video Detection',
            'image_segmentation': '✅ AI Image Segmentation with SAM'
        },
        'endpoints': {
            'health': '/health',
            'status': '/api/v1/status',
            'models': '/api/v1/models',
            'image_detect': 'POST /api/v1/detect',
            'image_chat': 'POST /api/v1/chat',
            'image_describe': 'POST /api/v1/describe',
            'segment_point': 'POST /api/v1/segment/point',
            'video_upload': 'POST /api/v1/video/upload',
            'video_process': 'POST /api/v1/video/process',
            'video_download': 'GET /api/v1/video/download',
            'docs': '/docs (Swagger UI)',
            'redoc': '/redoc (ReDoc)'
        }
    })

# ============================================================
# IMAGE DETECTION ENDPOINTS
# ============================================================

@app.post('/api/v1/detect')
async def detect_objects(file: UploadFile = File(...)):
    """YOLOv8 object detection endpoint"""
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLOv8 model not loaded")
    
    try:
        print(f"\n📸 Detection request: {file.filename}")
        
        # Read uploaded file
        contents = await file.read()
        
        # Save temporary file using absolute path
        temp_path = os.path.join(UPLOADS_DIR, file.filename)
        with open(temp_path, 'wb') as f:
            f.write(contents)
        
        # Run detection
        results = yolo_model.predict(
            source=temp_path,
            conf=settings.CONF_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            device=settings.DEVICE,
            verbose=False
        )
        
        # Extract results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append({
                        'class': int(box.cls),
                        'class_name': yolo_model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': {
                            'x1': float(box.xyxy[0][0]),
                            'y1': float(box.xyxy[0][1]),
                            'x2': float(box.xyxy[0][2]),
                            'y2': float(box.xyxy[0][3])
                        }
                    })
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"✅ Detection complete: {len(detections)} objects found")
        
        return JSONResponse(to_native({
            'filename': file.filename,
            'detections': detections,
            'detection_count': len(detections)
        }))
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

# ============================================================
# IMAGE CHAT/VQA ENDPOINTS
# ============================================================

@app.post('/api/v1/chat')
async def chat_with_image(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        print(f"\n💬 Chat request: {question}")
        llava = get_llava_engine()
        image_bytes = await file.read()
        
        # Pass bytes directly to the engine (it handles opening/RGB conversion)
        response = llava.chat(image_bytes, question)
        
        print(f"✅ Chat response generated")
        return JSONResponse({
            'question': question,
            'answer': response,
            'filename': file.filename,
            'model': llava.model_name, # Dynamic model name check
            'device': 'CPU'
        })
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Chat error: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={'error': error_msg}
        )

@app.post('/api/v1/describe')
async def describe_image(file: UploadFile = File(...)):
    try:
        print(f"\n📝 Description request")
        llava = get_llava_engine()
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        prompt = "Describe this image in detail. What objects, people, or scenes do you see?"
        description = llava.chat(image, prompt)
        print(f"✅ Description generated")
        return JSONResponse({
            'filename': file.filename,
            'description': description,
            'model': 'LLaVA (Ollama CPU)',
            'device': 'CPU'
        })
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Description error: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={'error': error_msg}
        )

# ============================================================
# SEGMENTATION ENDPOINTS
# ============================================================

@app.post("/api/v1/segment/point")
async def segment_point(
    file: UploadFile = File(...),
    points: str = Form(...),
    labels: str = Form(...)
):
    """Segment image using point prompts (SAM)"""
    try:
        points_list = json.loads(points)
        labels_list = json.loads(labels)
        
        if len(points_list) == 0:
            raise ValueError("At least one point required")
        if len(points_list) != len(labels_list):
            raise ValueError(f"Points/labels mismatch: {len(points_list)} vs {len(labels_list)}")
        
        print(f"🎯 Segmentation request: {len(points_list)} points")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        points_np = np.array(points_list, dtype=np.float32)
        labels_np = np.array(labels_list, dtype=np.int32)
        
        engine = get_sam_engine()
        pred = engine.predictor
        pred.set_image(image_np)
        
        print("   Running SAM prediction...")
        masks, scores, logits = pred.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=False
        )
        
        mask = masks[0]
        
        # --- PATH FIX: USE OUTPUTS_DIR ---
        timestamp = int(time_module.time() * 1000)
        
        # 1. Define Filenames
        mask_filename = f'mask_{timestamp}.png'
        viz_filename = f'viz_{timestamp}.png'
        
        # 2. Define Absolute Paths (For Saving)
        mask_path = os.path.join(OUTPUTS_DIR, mask_filename)
        viz_path = os.path.join(OUTPUTS_DIR, viz_filename)
        
        # 3. Save Files
        mask_binary = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_img.save(mask_path)
        
        viz_img = image_np.copy().astype(np.float32)
        viz_img[mask] = [0, 255, 0]
        viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
        viz_img_pil = Image.fromarray(viz_img)
        viz_img_pil.save(viz_path)
        
        pixels = int(np.sum(mask))
        total_pixels = image_np.shape[0] * image_np.shape[1]
        coverage = (pixels / total_pixels) * 100
        confidence = float(scores[0])
        
        print(f"   ✅ Segmentation success! Saved to {mask_path}")
        
        # 4. Return URLs (For Frontend - Relative to Server Root)
        return JSONResponse(to_native({
            "success": True,
            "mask": f"/outputs/{mask_filename}",
            "visualization": f"/outputs/{viz_filename}",
            "pixels": pixels,
            "coverage_percent": coverage,
            "confidence": confidence,
            "image_size": {
                "width": int(image_np.shape[1]),
                "height": int(image_np.shape[0])
            }
        }))
        
    except ValueError as e:
        print(f"❌ ValueError: {str(e)}")
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        print(f"❌ Segmentation error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.post("/api/v1/segment/automatic")
async def segment_automatic(file: UploadFile = File(...)):
    """Automatic segmentation"""
    try:
        print(f"🎯 Automatic segmentation request")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        engine = get_sam_engine()
        pred = engine.predictor
        pred.set_image(image_np)
        
        print("   Running automatic segmentation...")
        masks, scores, logits = pred.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True
        )
        
        # --- PATH FIX: USE OUTPUTS_DIR ---
        timestamp = int(time_module.time() * 1000)
        
        # 1. Define Filenames
        viz_filename = f'auto_viz_{timestamp}.png'
        mask_filename = f'auto_mask_{timestamp}.png'
        
        # 2. Define Absolute Paths
        viz_path = os.path.join(OUTPUTS_DIR, viz_filename)
        mask_path = os.path.join(OUTPUTS_DIR, mask_filename)
        
        # 3. Save Visualization
        viz_img = image_np.copy().astype(np.float32)
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 165, 0], [128, 0, 128],
        ]
        
        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            viz_img[mask] = color
        
        viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
        viz_img_pil = Image.fromarray(viz_img)
        viz_img_pil.save(viz_path)
        
        # 4. Save Mask
        combined_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined_mask = combined_mask | mask
        
        mask_binary = (combined_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_img.save(mask_path)
        
        pixels = int(np.sum(combined_mask))
        coverage = (pixels / (image_np.shape[0] * image_np.shape[1])) * 100
        avg_confidence = float(np.mean(scores))
        
        # 5. Return URLs
        return JSONResponse(to_native({
            "success": True,
            "mask": f"/outputs/{mask_filename}",
            "visualization": f"/outputs/{viz_filename}",
            "objects_found": len(masks),
            "pixels": pixels,
            "coverage_percent": coverage,
            "average_confidence": avg_confidence,
            "image_size": {
                "width": int(image_np.shape[1]),
                "height": int(image_np.shape[0])
            }
        }))
        
    except Exception as e:
        print(f"❌ Segmentation error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.post("/api/v1/segment/box")
async def segment_box(
    file: UploadFile = File(...),
    box: str = Form(...),
    class_name: str = Form(None)
):
    """
    Segment image using a bounding box prompt (SAM).
    """
    try:
        # Parse box coordinates
        box_list = json.loads(box)
        if len(box_list) != 4:
            raise ValueError("Box must be a list of four numbers [x1, y1, x2, y2]")
        
        print(f"🎯 Box segmentation request: box={box_list}, class={class_name}")

        # Read and prepare image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Coordinate clipping/validation
        x1, y1, x2, y2 = map(int, box_list)
        x1 = np.clip(x1, 0, w-1); y1 = np.clip(y1, 0, h-1)
        x2 = np.clip(x2, 0, w-1); y2 = np.clip(y2, 0, h-1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid box coordinates after clipping")

        # Initialize/Load SAM Engine
        engine = get_sam_engine()
        pred = engine.predictor
        pred.set_image(image_np)

        print("   Running SAM prediction with bounding box...")
        box_np = np.array(box_list, dtype=np.float32)
        
        # Run Prediction
        masks, scores, logits = pred.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=False
        )
        mask = masks[0]
        score = scores[0] if len(scores) > 0 else 0.0

        # --- PATH FIX: USE OUTPUTS_DIR ---
        timestamp = int(time.time() * 1000)

        # 1. Define Filenames
        mask_filename = f'mask_{timestamp}.png'
        crop_filename = f'crop_{timestamp}.png'
        obj_filename = f'object_{timestamp}.png'

        # 2. Define Absolute Paths (For Saving)
        mask_path = os.path.join(OUTPUTS_DIR, mask_filename)
        crop_path = os.path.join(OUTPUTS_DIR, crop_filename)
        obj_path = os.path.join(OUTPUTS_DIR, obj_filename)

        # 3. Save Mask
        mask_binary = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_img.save(mask_path)

        # 4. Save Crop
        crop_img = image.crop((x1, y1, x2, y2))
        crop_img.save(crop_path)

        # 5. Save Transparent Object
        image_rgba = image.convert("RGBA")
        alpha_mask = Image.fromarray(mask_binary).convert("L")
        image_rgba.putalpha(alpha_mask)
        image_rgba.save(obj_path)

        # Calculate metrics
        pixels = int(np.sum(mask))
        coverage = (pixels / (h * w)) * 100
        
        # 6. Construct Response (Return URLs)
        response_data = {
            "success": True,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "class_name": class_name,
            "mask": f"/outputs/{mask_filename}",
            "crop": f"/outputs/{crop_filename}",
            "object": f"/outputs/{obj_filename}",
            "pixels": pixels,
            "coverage_percent": coverage,
            "confidence": score,
            "image_size": {"width": w, "height": h}
        }

        print(f"   ✅ Box segmentation success! Saved to {mask_path}")
        return JSONResponse(to_native(response_data))

    except ValueError as e:
        print(f"❌ Box Value Error: {e}")
        return JSONResponse(status_code=400, content={'error': str(e)})
    except Exception as e:
        print(f"❌ Segmentation (box) error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})

# ============================================================
# VIDEO PROCESSING ENDPOINTS
# ============================================================

@app.post('/api/v1/video/upload')
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    try:
        print(f"\n📹 Video upload: {file.filename}")
        
        valid_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in valid_video_formats:
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid format. Supported: {valid_video_formats}'}
            )
        
        # Use Absolute Path for uploads
        video_path = os.path.join(UPLOADS_DIR, file.filename)
        contents = await file.read()
        
        with open(video_path, 'wb') as f:
            f.write(contents)
        
        file_size = len(contents) / (1024 * 1024)
        
        print(f"✅ Video uploaded: {file_size:.2f} MB")
        
        return JSONResponse({
            'filename': file.filename,
            'path': video_path,
            'size_mb': round(file_size, 2),
            'status': 'uploaded'
        })
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

@app.post('/api/v1/video/process')
async def process_video(
    video_path: str = Form(...),
    conf_threshold: float = Form(0.5)
):
    """Process uploaded video with YOLOv8 detection"""
    try:
        if not video_path:
            return JSONResponse(status_code=400, content={'error': 'video_path required'})
        
        if not os.path.exists(video_path):
            return JSONResponse(status_code=404, content={'error': 'Video file not found'})
        
        print(f"\n🎬 Processing video: {video_path}")
        
        if yolo_model is None:
            return JSONResponse(status_code=500, content={'error': 'YOLOv8 model not loaded'})
        
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        # Save output to OUTPUTS_DIR (Absolute path)
        output_path = os.path.join(OUTPUTS_DIR, f"{name}_detected.mp4")
        
        processor = get_video_processor(yolo_model)
        
        stats = processor.process_video(
            video_path,
            output_path=output_path,
            conf_threshold=conf_threshold
        )
        
        print(f"✅ Video processing complete")
        
        # Use to_native to clean up numpy integers in stats
        return JSONResponse(to_native({
            'status': 'completed',
            'input_file': filename,
            'output_file': os.path.basename(output_path),
            'stats': stats,
            'download_url': f'/api/v1/video/download?file={os.path.basename(output_path)}'
        }))
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

@app.get('/api/v1/video/download')
async def download_video(file: str):
    """Download processed video file"""
    try:
        # Search in both output and upload folders to be safe
        paths = [
            os.path.join(OUTPUTS_DIR, file),
            os.path.join(UPLOADS_DIR, file)
        ]
        for p in paths:
            if os.path.exists(p):
                return FileResponse(path=p, filename=file, media_type='video/mp4')
        
        return JSONResponse(status_code=404, content={'error': 'File not found'})
        
    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

@app.post('/api/v1/video/stream')
async def stream_video_processing(
    video_path: str = Form(...),
    conf_threshold: float = Form(0.5)
):
    """Stream video processing progress (for real-time updates)"""
    try:
        if yolo_model is None:
            return JSONResponse(status_code=500, content={'error': 'YOLOv8 model not loaded'})
        
        processor = get_video_processor(yolo_model)
        
        progress_data = []
        
        def on_frame(data):
            progress_data.append(data)
        
        processor.process_video_streaming(
            video_path,
            conf_threshold=conf_threshold,
            callback=on_frame
        )
        
        return JSONResponse(to_native({
            'status': 'completed',
            'total_frames_processed': len(progress_data),
            'frames': progress_data[-10:]
        }))
        
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )
    

# ============================================================
# VIDEO summarizer ENDPOINTS
# ============================================================

@app.post('/api/v1/video/summarize')
async def summarize_video(video_path: str = Form(...)):
    """Generate a text summary of the video content"""
    try:
        # 1. Validate file existence
        if not video_path or not os.path.exists(video_path):
            return JSONResponse(
                status_code=404, 
                content={'error': f'Video file not found at: {video_path}'}
            )
        
        print(f"\n📝 Video Summary Request: {video_path}")
        
        # 2. Get the summarizer engine
        summarizer = get_video_summarizer()
        
        # 3. Run CPU-intensive task in a separate thread
        # This is crucial so the API doesn't "freeze" while processing images
        summary = await asyncio.to_thread(summarizer.summarize, video_path)
        
        print(f"✅ Summary generated successfully")
        
        # 4. Return result (using to_native to prevent any int32 errors)
        return JSONResponse(to_native({
            'status': 'success',
            'summary': summary,
            'video_file': os.path.basename(video_path)
        }))
        
    except ValueError as ve:
        # Catch known errors (like "0 frames extracted") as 400 Bad Request
        print(f"⚠️ Video Validation Error: {ve}")
        return JSONResponse(status_code=400, content={'error': str(ve)})
        
    except Exception as e:
        # Catch unexpected crashes as 500 Internal Error
        print(f"❌ Critical Summary Error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': f"Processing failed: {str(e)}"})
        
    except Exception as e:
        print(f"❌ Summary error: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={'error': str(exc)}
    )

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 VISLANG BACKEND - STARTING SERVER")
    print("="*70)
    print(f"Host: {settings.SERVER_HOST}")
    print(f"Port: {settings.SERVER_PORT}")
    print(f"Device: {settings.DEVICE}")
    print("="*70)
    print("\n📚 API DOCUMENTATION:")
    print(f"  • Swagger UI: http://localhost:{settings.SERVER_PORT}/docs")
    print(f"  • ReDoc: http://localhost:{settings.SERVER_PORT}/redoc")
    print("\n✨ FEATURES AVAILABLE:")
    print("  ✅ Image Object Detection (YOLOv8)")
    print("  ✅ Image Chat & Description (Ollama LLaVA)")
    print("  ✅ Image Segmentation (SAM - Segment Anything)")
    print("  ✅ Video Real-Time Processing (YOLOv8 + OpenCV)")
    print("\n⏳ LAZY LOADING:")
    print("  • Ollama LLaVA loads on first chat request (2-3 minutes)")
    print("  • SAM loads on first segmentation request")
    print("  • Video processing runs locally without cloud dependency")
    print("="*70 + "\n")
    
    uvicorn.run(
        app, 
        host=settings.SERVER_HOST, 
        port=settings.SERVER_PORT, 
        reload=True
    )