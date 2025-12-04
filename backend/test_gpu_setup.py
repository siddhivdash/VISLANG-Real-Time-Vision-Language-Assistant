
import sys
import torch
from utils.gpu_monitor import GPUMonitor

def main():
    print('\n' + '='*60)
    print('VisLang GPU Setup Verification')
    print('='*60 + '\n')

    # Step 1: Check GPU Availability
    print('Step 1: Checking GPU...')
    gpu_info = GPUMonitor.check_gpu_availability()

    if gpu_info['cuda_available']:
        print(f'✅ CUDA Available: YES')
        print(f'✅ GPU Count: {gpu_info["gpu_count"]}')
        for mem in gpu_info['gpu_memory']:
            print(f'   - {mem["name"]}: {mem["total_gb"]} GB')
        if gpu_info.get('cuda_version'):
            print(f'✅ CUDA Version: {gpu_info["cuda_version"]}')
    else:
        print(f'⚠️  CUDA Available: NO (CPU mode)')
        if gpu_info['error']:
            print(f'   {gpu_info["error"]}')

    # Step 2: Check System Memory
    print('\nStep 2: Checking System Memory...')
    sys_mem = GPUMonitor.get_system_memory_usage()
    print(f'✅ RAM: {sys_mem["used_gb"]}GB / {sys_mem["total_gb"]}GB ({sys_mem["percent_used"]:.1f}%)')

    if gpu_info['cuda_available']:
        gpu_mem = GPUMonitor.get_gpu_memory_usage()
        if gpu_mem['success']:
            for gpu in gpu_mem['gpus']:
                print(f'✅ GPU {gpu["gpu_id"]}: {gpu["memory_used_mb"]}MB / {gpu["memory_total_mb"]}MB')

    # Step 3: Test Model Loading
    print('\nStep 3: Testing Model Loading...')

    # Test YOLOv8
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print('✅ YOLOv8n loaded successfully')
    except Exception as e:
        print(f'❌ YOLOv8n failed: {str(e)[:100]}')
        return False

    # Test CLIP
    try:
        import clip
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, preprocess = clip.load('ViT-B/32', device=device)
        print('✅ CLIP loaded successfully')
    except Exception as e:
        print(f'❌ CLIP failed: {str(e)[:100]}')
        return False

    # Test PaddleOCR (Fixed version)
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print('✅ PaddleOCR loaded successfully')
    except Exception as e:
        print(f'❌ PaddleOCR failed: {str(e)[:100]}')
        return False

    print('\n' + '='*60)
    print('✅ Setup verification complete!')
    print('='*60 + '\n')
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)