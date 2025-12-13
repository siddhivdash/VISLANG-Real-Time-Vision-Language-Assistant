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



load_dotenv()


# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================


app = FastAPI(
    title="VisLang Backend",
    description="Real-time Multimodal Vision-Language Assistant API",
    version="0.2.0"
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
        sam_engine = SAMSegmentationEngine(model_type="vit_b", device="cpu")
        predictor = sam_engine.predictor
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
    
    return JSONResponse({
        'status': 'healthy',
        'service': 'VisLang Backend',
        'version': '0.2.0',
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
    })


@app.get('/api/v1/status')
async def get_status():
    """Detailed system status endpoint"""
    pytorch_info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': settings.DEVICE
    }
    
    return JSONResponse({
        'service': 'VisLang Backend',
        'version': '0.2.0',
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
    })


# ============================================================
# ROOT ENDPOINT
# ============================================================


@app.get('/')
async def root():
    """Root endpoint with API documentation"""
    return JSONResponse({
        'service': 'VisLang Backend API',
        'version': '0.2.0',
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
        
        return JSONResponse({
            'filename': file.filename,
            'detections': detections,
            'detection_count': len(detections)
        })
        
    except Exception as e:
        print(f"❌ Detection error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.get('/api/v1/models')
async def get_models():
    """Get list of loaded models and their info"""
    return JSONResponse({
        'models': {
            'yolo': {
                'name': 'YOLOv8n',
                'model_file': settings.YOLO_MODEL,
                'loaded': yolo_model is not None,
                'type': 'object_detection',
                'classes': 80 if yolo_model else 0
            },
            'clip': {
                'name': 'CLIP ViT-B/32',
                'model_name': settings.CLIP_MODEL,
                'loaded': clip_model is not None,
                'type': 'vision_embeddings'
            },
            'ollama_llava': {
                'name': 'LLaVA-1.5-7B',
                'model_name': 'liuhaotian/llava-v1.5-7b',
                'status': 'lazy-loading',
                'type': 'vision_language_model',
                'backend': 'Ollama (CPU optimized)'
            },
            'sam': {
                'name': 'Segment Anything Model (ViT-B)',
                'status': 'lazy-loading',
                'type': 'image_segmentation',
                'device': 'CPU'
            },
            'video_processor': {
                'name': 'VideoProcessor',
                'status': 'ready',
                'type': 'video_analysis',
                'uses_model': 'YOLOv8'
            }
        }
    })


# ============================================================
# IMAGE CHAT/VQA ENDPOINTS
# ============================================================


@app.post('/api/v1/chat')
async def chat_with_image(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Vision-Language Chat using Ollama LLaVA
    First request loads model, subsequent requests use cached model
    """
    try:
        print(f"\n💬 Chat request: {question}")
        
        # Get lazy-loaded engine
        llava = get_llava_engine()
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get response
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
    """
    Get detailed image description using Ollama LLaVA
    """
    try:
        print(f"\n📝 Description request")
        
        # Get lazy-loaded engine
        llava = get_llava_engine()
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate description
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
    """Segment image using point prompts (SAM) - 100% FIXED"""
    try:
        # Parse JSON strings
        points_list = json.loads(points)
        labels_list = json.loads(labels)
        
        # Validate inputs
        if len(points_list) == 0:
            raise ValueError("At least one point required")
        
        if len(points_list) != len(labels_list):
            raise ValueError(f"Points/labels mismatch: {len(points_list)} vs {len(labels_list)}")
        
        print(f"🎯 Segmentation request: {len(points_list)} points")
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        print(f"   Image shape: {image_np.shape}")
        
        # Convert to numpy arrays with correct dtypes
        points_np = np.array(points_list, dtype=np.float32)  # Shape: (N, 2)
        labels_np = np.array(labels_list, dtype=np.int32)     # Shape: (N,)
        
        print(f"   Points shape: {points_np.shape}, Labels shape: {labels_np.shape}")
        
        # Get SAM predictor (lazy load)
        engine = get_sam_engine()
        pred = engine.predictor
        
        # Set image FIRST
        print("   Setting image...")
        pred.set_image(image_np)
        
        # Run SAM prediction
        print("   Running SAM prediction...")
        masks, scores, logits = pred.predict(
            point_coords=points_np,      # Shape: (N, 2)
            point_labels=labels_np,      # Shape: (N,)
            multimask_output=False
        )
        
        # ✅ FIX #1: Get FIRST mask from output (not entire array!)
        mask = masks[0]  # THIS IS THE KEY FIX!
        
        print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        # Save mask as binary image
        mask_binary = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_path = f'outputs/mask_{int(time_module.time() * 1000)}.png'
        mask_img.save(mask_path)
        print(f"   ✅ Mask saved: {mask_path}")
        
        # Create visualization: SIMPLE green overlay
        viz_img = image_np.copy().astype(np.float32)
        # Set masked pixels to green [0, 255, 0]
        viz_img[mask] = [0, 255, 0]
        viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
        viz_img_pil = Image.fromarray(viz_img)
        viz_path = f'outputs/viz_{int(time_module.time() * 1000)}.png'
        viz_img_pil.save(viz_path)
        print(f"   ✅ Visualization saved: {viz_path}")
        
        # ✅ FIX #2: Correct statistics calculations
        pixels = int(np.sum(mask))
        total_pixels = image_np.shape[0] * image_np.shape[1]  # H * W
        coverage = (pixels / total_pixels) * 100
        confidence = float(scores[0])
        
        print(f"   ✅ Segmentation success!")
        print(f"      Pixels: {pixels}, Coverage: {coverage:.2f}%, Confidence: {confidence:.2f}")
        
        return JSONResponse({
            "success": True,
            "mask": f"/{mask_path}",
            "visualization": f"/{viz_path}",
            "pixels": pixels,
            "coverage_percent": coverage,
            "confidence": confidence,
            "image_size": {
                "width": int(image_np.shape[1]),   # Width
                "height": int(image_np.shape[0])   # Height
            }
        })
        
    except ValueError as e:
        print(f"❌ ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Segmentation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")
    

@app.post("/api/v1/segment/automatic")
async def segment_automatic(file: UploadFile = File(...)):
    """Automatic segmentation - segments ALL objects without manual input!"""
    try:
        print(f"🎯 Automatic segmentation request")
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        print(f"   Image shape: {image_np.shape}")
        
        # Get SAM predictor
        engine = get_sam_engine()
        pred = engine.predictor
        
        # Set image
        pred.set_image(image_np)
        
        # AUTOMATIC segmentation (no points needed!)
        print("   Running automatic segmentation...")
        masks, scores, logits = pred.predict(
            point_coords=None,      # No manual points!
            point_labels=None,      # No labels!
            multimask_output=True   # Get ALL objects
        )
        
        print(f"   Found {len(masks)} objects")
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        # Create visualization with different colors for each object
        viz_img = image_np.copy().astype(np.float32)
        
        colors = [
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
        ]
        
        # Color each object differently
        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            viz_img[mask] = color
        
        viz_img = np.clip(viz_img, 0, 255).astype(np.uint8)
        viz_img_pil = Image.fromarray(viz_img)
        viz_path = f'outputs/auto_viz_{int(time_module.time() * 1000)}.png'
        viz_img_pil.save(viz_path)
        print(f"   ✅ Visualization saved: {viz_path}")
        
        # Create combined mask (all objects together)
        combined_mask = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined_mask = combined_mask | mask
        
        mask_binary = (combined_mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_path = f'outputs/auto_mask_{int(time_module.time() * 1000)}.png'
        mask_img.save(mask_path)
        print(f"   ✅ Mask saved: {mask_path}")
        
        # Statistics
        pixels = int(np.sum(combined_mask))
        total_pixels = image_np.shape[0] * image_np.shape[1]
        coverage = (pixels / total_pixels) * 100
        avg_confidence = float(np.mean(scores))
        
        print(f"   ✅ Success! Found {len(masks)} objects")
        
        return JSONResponse({
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
        })
        
    except Exception as e:
        print(f"❌ Segmentation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/v1/segment/box")
async def segment_box(
    file: UploadFile = File(...),
    box: str = Form(...),
    class_name: str = Form(None)
):
    """
    Segment image using a bounding box prompt (SAM).
    Expects 'box' as a JSON list [x1, y1, x2, y2].
    """
    try:
        # Parse the bounding box JSON
        box_list = json.loads(box)
        if len(box_list) != 4:
            raise ValueError("Box must be a list of four numbers [x1, y1, x2, y2]")
        
        print(f"🎯 Box segmentation request: box={box_list}, class={class_name}")

        # Read and prepare image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        print(f"   Image size: {w}x{h}")

        # Ensure box coordinates are integers within image bounds
        x1, y1, x2, y2 = map(int, box_list)
        x1 = np.clip(x1, 0, w-1); y1 = np.clip(y1, 0, h-1)
        x2 = np.clip(x2, 0, w-1); y2 = np.clip(y2, 0, h-1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid box coordinates after clipping")

        # Get SAM predictor
        engine = get_sam_engine()
        pred = engine.predictor
        pred.set_image(image_np)

        # Run SAM with box prompt
        print("   Running SAM prediction with bounding box...")
        box_np = np.array(box_list, dtype=np.float32)
        masks, scores, logits = pred.predict(
            box=box_np,
            multimask_output=False
        )
        mask = masks[0]  # single mask
        score = scores[0] if len(scores) > 0 else None

        # Create outputs directory if not exists
        os.makedirs('outputs', exist_ok=True)
        timestamp = int(time.time() * 1000)

        # Save mask image (binary)
        mask_binary = (mask * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_binary)
        mask_path = f'outputs/mask_{timestamp}.png'
        mask_img.save(mask_path)
        print(f"   ✅ Mask saved: {mask_path}")

        # Save cropped image (RGB)
        crop_img = image.crop((x1, y1, x2, y2))
        crop_path = f'outputs/crop_{timestamp}.png'
        crop_img.save(crop_path)
        print(f"   ✅ Cropped image saved: {crop_path}")

        # Save object image with background transparent (RGBA)
        image_rgba = image.convert("RGBA")
        alpha_mask = Image.fromarray(mask_binary).convert("L")
        image_rgba.putalpha(alpha_mask)
        object_path = f'outputs/object_{timestamp}.png'
        image_rgba.save(object_path)
        print(f"   ✅ Object (background removed) saved: {object_path}")

        # Compute statistics
        pixels = int(np.sum(mask))
        total_pixels = h * w
        coverage = (pixels / total_pixels) * 100
        print(f"   Pixels: {pixels}, Coverage: {coverage:.2f}%, Confidence: {score:.2f}")

        # Build JSON response with download URLs
        return JSONResponse({
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
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Segmentation (box) error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation error: {e}")


# ============================================================
# VIDEO PROCESSING ENDPOINTS
# ============================================================


@app.post('/api/v1/video/upload')
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    try:
        print(f"\n📹 Video upload: {file.filename}")
        
        # Validate file is video
        valid_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in valid_video_formats:
            return JSONResponse(
                status_code=400,
                content={'error': f'Invalid format. Supported: {valid_video_formats}'}
            )
        
        # Save uploaded file
        video_path = f"{settings.UPLOAD_FOLDER}/{file.filename}"
        contents = await file.read()
        
        with open(video_path, 'wb') as f:
            f.write(contents)
        
        # Get file size
        file_size = len(contents) / (1024 * 1024)  # MB
        
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
            return JSONResponse(
                status_code=400,
                content={'error': 'video_path required'}
            )
        
        if not os.path.exists(video_path):
            return JSONResponse(
                status_code=404,
                content={'error': 'Video file not found'}
            )
        
        print(f"\n🎬 Processing video: {video_path}")
        
        if yolo_model is None:
            return JSONResponse(
                status_code=500,
                content={'error': 'YOLOv8 model not loaded'}
            )
        
        # Generate output path
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        output_path = f"{settings.UPLOAD_FOLDER}/{name}_detected.mp4"
        
        # Get processor
        processor = get_video_processor(yolo_model)
        
        # Process video
        stats = processor.process_video(
            video_path,
            output_path=output_path,
            conf_threshold=conf_threshold
        )
        
        print(f"✅ Video processing complete")
        
        return JSONResponse({
            'status': 'completed',
            'input_file': filename,
            'output_file': os.path.basename(output_path),
            'stats': stats,
            'download_url': f'/api/v1/video/download?file={os.path.basename(output_path)}'
        })
        
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
            return JSONResponse(
                status_code=404,
                content={'error': 'File not found'}
            )
        
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
            return JSONResponse(
                status_code=500,
                content={'error': 'YOLOv8 model not loaded'}
            )
        
        processor = get_video_processor(yolo_model)
        
        progress_data = []
        
        def on_frame(data):
            progress_data.append(data)
        
        processor.process_video_streaming(
            video_path,
            conf_threshold=conf_threshold,
            callback=on_frame
        )
        
        return JSONResponse({
            'status': 'completed',
            'total_frames_processed': len(progress_data),
            'frames': progress_data[-10:]  # Last 10 frames
        })
        
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


# ============================================================
# ERROR HANDLERS
# ============================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={'error': str(exc)}
    )


# ============================================================
# SERVER STARTUP
# ============================================================


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