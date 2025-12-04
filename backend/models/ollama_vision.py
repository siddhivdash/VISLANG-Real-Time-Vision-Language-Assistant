import requests
import base64
import io
from PIL import Image
import subprocess
import time

class OllamaVisionEngine:
    """Ollama LLaVA - CPU mode forced"""
    
    def __init__(self, model_name="phi3"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.connected = False
        
        # Test connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.connected = True
                print("✅ Ollama connected and ready")
            else:
                print("⚠️  Ollama not responding properly")
        except:
            print("⚠️  Cannot connect to Ollama. Make sure:")
            print("   1. Ollama is running: 'ollama serve'")
            print("   2. Environment: set OLLAMA_NUM_GPU=0")
    
    def chat(self, image_data, question, max_tokens=256):
        """Chat using Ollama LLaVA on CPU"""
        try:
            if not self.connected:
                return "Error: Ollama not connected. Start Ollama with: ollama serve"
            
            # Convert image to base64
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            print(f"⏳ Sending to Ollama (CPU)...")
            
            # Call Ollama API with timeout
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": question,
                    "images": [img_base64],
                    "stream": False
                },
                timeout=600  # 10 minutes for CPU inference
            )
            
            print(f"📍 Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "No response generated")
                print(f"✅ Response received!")
                return answer
            else:
                error_text = response.text
                print(f"❌ Ollama error: {error_text}")
                
                # Check if it's GPU memory error
                if "memory" in error_text.lower() or "gpu" in error_text.lower():
                    return "Error: GPU memory issue. Restart Ollama with: set OLLAMA_NUM_GPU=0 && ollama serve"
                return f"Error: {error_text}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timeout. Ollama inference took too long (>10 min)"
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure it's running on localhost:11434"
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return f"Error: {str(e)}"


# Singleton
_engine = None

def get_ollama_engine():
    global _engine
    if _engine is None:
        _engine = OllamaVisionEngine()
    return _engine
