import os
import requests
import base64
import io
from PIL import Image
import time

class OllamaVisionEngine:
    """Ollama Vision - Tuned for Moondream (Maximum Speed)"""
    
    def __init__(self, model_name="moondream"): 
        self.model_name = model_name
        
        # --- DOCKER FIX: Use Environment Variable ---
        # If running in Docker, this will pick up 'http://host.docker.internal:11434'
        # If running locally, it falls back to 'http://localhost:11434'
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.generate_url = f"{self.base_url}/api/generate"
        self.connected = False
        
        print(f"🔌 Connecting to Ollama at: {self.base_url}...")
        
        # Test connection
        try:
            # We use /api/tags to list models and verify connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.connected = True
                print(f"✅ Ollama connected. Target Model: {self.model_name}")
                
                # Check if model exists (Optional but helpful)
                models = [m['name'] for m in response.json().get('models', [])]
                if not any(self.model_name in m for m in models):
                    print(f"⚠️ Warning: Model '{self.model_name}' not found in Ollama list. You might need to run 'ollama pull {self.model_name}'")
            else:
                print(f"⚠️ Ollama connected but returned status: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Cannot connect to Ollama at {self.base_url}. Error: {e}")
            print("   (If using Docker, ensure OLLAMA_HOST is set correctly in docker-compose.yml)")

    def process_image(self, image_data):
        """Pipeline: Load -> RGB -> Resize (512px for Moondream)"""
        try:
            # 1. Load
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = image_data
            
            # 2. Universal Transparency Fix (Catches RGBA, P, LA, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # 3. Aggressive Resize (Crucial for Moondream Stability)
            # Moondream native resolution is small (378x378). 
            # Sending 512px is the perfect balance of detail vs. confusion.
            max_size = 512
            if max(image.size) > max_size:
                # Use thumbnail to preserve aspect ratio
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            return image
        except Exception as e:
            print(f"❌ Image processing failed: {e}")
            raise e

    def chat(self, image_data, question, max_tokens=300):
        try:
            # Re-check connection if it failed initially
            if not self.connected:
                print("🔄 Retrying Ollama connection...")
                try:
                    if requests.get(f"{self.base_url}/api/tags", timeout=2).ok:
                        self.connected = True
                except:
                    return "Error: Ollama not connected. Make sure 'ollama serve' is running and accessible."
            
            # Run Optimization Pipeline
            image = self.process_image(image_data)
            
            # Convert to Base64
            buffered = io.BytesIO()
            # JPEG is faster and smaller for LLMs than PNG
            image.save(buffered, format="JPEG", quality=85) 
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            print(f"⏳ Sending to Ollama ({self.model_name}) at {self.generate_url}...")
            
            payload = {
                "model": self.model_name,
                "prompt": question, 
                "images": [img_base64],
                "stream": False,
                "keep_alive": "5m", 
                "options": {
                    "num_predict": max_tokens, 
                    "temperature": 0.1, # Low temp for factual answers
                    "top_p": 0.5,       
                    "top_k": 20         
                }
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120 # Give it time to think
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                # Safety Check
                if not answer:
                    return "I saw the image, but I couldn't generate a text response."
                
                print(f"✅ Response received!")
                return answer
            else:
                return f"Error from Ollama ({response.status_code}): {response.text}"
                
        except Exception as e:
            print(f"❌ Chat Error: {str(e)}")
            return f"System Error: {str(e)}"

# Singleton pattern to reuse connection
_engine = None

def get_ollama_engine():
    global _engine
    if _engine is None:
        # Defaults to moondream for speed, but can be swapped to 'llava'
        _engine = OllamaVisionEngine(model_name="moondream") 
    return _engine