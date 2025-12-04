import torch
import psutil
import GPUtil
from typing import Dict, Any

class GPUMonitor:
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """Check GPU availability and CUDA setup"""
        info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': [],
            'pytorch_version': torch.__version__,
            'error': None
        }
        
        try:
            cuda_available = torch.cuda.is_available()
            info['cuda_available'] = cuda_available
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                info['gpu_count'] = gpu_count
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    
                    info['gpu_names'].append(gpu_name)
                    info['gpu_memory'].append({
                        'gpu_id': i,
                        'name': gpu_name,
                        'total_gb': round(gpu_memory, 2)
                    })
                
                info['cuda_version'] = torch.version.cuda
                info['cudnn_version'] = torch.backends.cudnn.version()
            else:
                info['error'] = 'CUDA not available. Using CPU.'
                
        except Exception as e:
            info['error'] = f'GPU check failed: {str(e)}'
        
        return info
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[str, Any]:
        """Get current GPU memory usage"""
        try:
            gpus = GPUtil.getGPUs()
            memory_info = []
            
            for gpu in gpus:
                memory_info.append({
                    'gpu_id': gpu.id,
                    'gpu_name': gpu.name,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_free_mb': gpu.memoryFree,
                    'gpu_load_percent': gpu.load * 100,
                })
            
            return {'success': True, 'gpus': memory_info}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def get_system_memory_usage() -> Dict[str, float]:
        """Get system RAM usage"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent
        }

if __name__ == '__main__':
    print('=== GPU Check ===')
    gpu_info = GPUMonitor.check_gpu_availability()
    print(f'CUDA Available: {gpu_info["cuda_available"]}')
    print(f'GPU Count: {gpu_info["gpu_count"]}')
    if gpu_info['gpu_memory']:
        for gpu in gpu_info['gpu_memory']:
            print(f'  {gpu["name"]}: {gpu["total_gb"]} GB')
        