from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
from fastapi.staticfiles import StaticFiles
import time
import traceback

load_dotenv()

# ============================================================
# HELPER: CONVERT NUMPY TO PYTHON NATIVE TYPES
# ============================================================
def to_native(obj):
    """
    Recursively convert NumPy types to Python native types
    to avoid 'int32 is not JSON serializable' errors.
    """
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
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
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Create uploads folder if it doesn't exist
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
os.makedirs('outputs', exist_ok=True)

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
    yolo_model = YOLO(settings.YOLO_MODEL)
    print(f"✅ YOLOv8 loaded: {settings.YOLO_MODEL}")
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
        'service': 'VisLang Backend API',
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
        
        # Save temporary file
        temp_path = f"{settings.UPLOAD_FOLDER}/{file.filename}"
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
        image = Image.open(io.BytesIO(image_bytes))
        response = llava.chat(image, question)
        print(f"✅ Chat response generated")
        return JSONResponse({
            'question': question,
            'answer': response,
            'filename': file.filename,
            'model': 'LLaVA (Ollama CPU)',
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
        
        os.makedirs('outputs', exist_ok=True)
        timestamp = int(time_module.time() * 1000)
        
        mask_binary = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_path = f'outputs/mask_{timestamp}.png'
        mask_img.save(mask_path)
        
        viz_img = image_np.copy().astype(np.float32)
        viz_img[mask] = [0, 255, 0]
        viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
        viz_img_pil = Image.fromarray(viz_img)
        viz_path = f'outputs/viz_{timestamp}.png'
        viz_img_pil.save(viz_path)
        
        pixels = int(np.sum(mask))
        total_pixels = image_np.shape[0] * image_np.shape[1]
        coverage = (pixels / total_pixels) * 100
        confidence = float(scores[0])
        
        print(f"   ✅ Segmentation success!")
        
        # Use to_native helper to clean response
        return JSONResponse(to_native({
            "success": True,
            "mask": f"/{mask_path}",
            "visualization": f"/{viz_path}",
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
        
        os.makedirs('outputs', exist_ok=True)
        timestamp = int(time_module.time() * 1000)
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
        viz_path = f'outputs/auto_viz_{timestamp}.png'
        viz_img_pil.save(viz_path)
        
        combined_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined_mask = combined_mask | mask
        
        mask_binary = (combined_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_path = f'outputs/auto_mask_{timestamp}.png'
        mask_img.save(mask_path)
        
        pixels = int(np.sum(combined_mask))
        coverage = (pixels / (image_np.shape[0] * image_np.shape[1])) * 100
        avg_confidence = float(np.mean(scores))
        
        # Use to_native helper to clean response
        return JSONResponse(to_native({
            "success": True,
            "mask": f"/{mask_path}",
            "visualization": f"/{viz_path}",
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
        box_list = json.loads(box)
        if len(box_list) != 4:
            raise ValueError("Box must be a list of four numbers [x1, y1, x2, y2]")
        
        print(f"🎯 Box segmentation request: box={box_list}, class={class_name}")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        x1, y1, x2, y2 = map(int, box_list)
        x1 = np.clip(x1, 0, w-1); y1 = np.clip(y1, 0, h-1)
        x2 = np.clip(x2, 0, w-1); y2 = np.clip(y2, 0, h-1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid box coordinates after clipping")

        engine = get_sam_engine()
        pred = engine.predictor
        pred.set_image(image_np)

        print("   Running SAM prediction with bounding box...")
        box_np = np.array(box_list, dtype=np.float32)
        
        # NOTE: Standard SAM Predictor 'predict' method takes 'box' as a separate argument.
        masks, scores, logits = pred.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=False
        )
        mask = masks[0]
        score = scores[0] if len(scores) > 0 else 0.0

        os.makedirs('outputs', exist_ok=True)
        timestamp = int(time.time() * 1000)

        # 1. Mask
        mask_binary = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_path = f'outputs/mask_{timestamp}.png'
        mask_img.save(mask_path)

        # 2. Crop
        crop_img = image.crop((x1, y1, x2, y2))
        crop_path = f'outputs/crop_{timestamp}.png'
        crop_img.save(crop_path)

        # 3. Transparent Object
        image_rgba = image.convert("RGBA")
        alpha_mask = Image.fromarray(mask_binary).convert("L")
        image_rgba.putalpha(alpha_mask)
        object_path = f'outputs/object_{timestamp}.png'
        image_rgba.save(object_path)

        pixels = int(np.sum(mask))
        coverage = (pixels / (h * w)) * 100
        
        # Use to_native helper to ensure JSON compatibility
        response_data = {
            "success": True,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "class_name": class_name,
            "mask": f"/{mask_path}",
            "crop": f"/{crop_path}",
            "object": f"/{object_path}",
            "pixels": pixels,
            "coverage_percent": coverage,
            "confidence": score,
            "image_size": {"width": w, "height": h}
        }

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
        
        video_path = f"{settings.UPLOAD_FOLDER}/{file.filename}"
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
        output_path = f"{settings.UPLOAD_FOLDER}/{name}_detected.mp4"
        
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
        file_path = os.path.join(settings.UPLOAD_FOLDER, file)
        
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={'error': 'File not found'})
        
        print(f"\n📥 Downloading: {file}")
        
        return FileResponse(
            path=file_path,
            filename=file,
            media_type='video/mp4'
        )
        
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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={'error': str(exc)}
    )

if __name__ == '__main__':
    import uvicorn
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