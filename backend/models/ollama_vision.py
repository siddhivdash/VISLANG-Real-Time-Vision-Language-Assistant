import requests
import base64
import io
from PIL import Image
import time

class OllamaVisionEngine:
    """Ollama Vision - Tuned for Moondream (Maximum Speed)"""
    
    def __init__(self, model_name="moondream"): 
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.connected = False
        
        # Test connection
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                self.connected = True
                print(f"✅ Ollama connected. Target Model: {self.model_name}")
            else:
                print("⚠️  Ollama connected but returned non-200 status")
        except:
            print("⚠️  Cannot connect to Ollama. Make sure 'ollama serve' is running.")

    def process_image(self, image_data):
        """Pipeline: Load -> RGB -> Resize (512px for Moondream)"""
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
            image = image.resize((max_size, max_size), Image.Resampling.LANCZOS)
            
        return image

    def chat(self, image_data, question, max_tokens=300):
        try:
            if not self.connected:
                return "Error: Ollama not connected. Run 'ollama serve'."
            
            # Run Optimization Pipeline
            image = self.process_image(image_data)
            
            # Convert to Base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=80) # Lower quality slightly for speed
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            print(f"⏳ Sending to Ollama ({self.model_name})...")
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": question, # Raw question works best for Moondream
                    "images": [img_base64],
                    "stream": False,
                    "keep_alive": "10m", # Keep loaded longer to prevent re-loading lag
                    "options": {
                        "num_predict": max_tokens, 
                        "temperature": 0.0, # ZERO temp = No hallucinations, strict logic
                        "top_p": 0.5,       # Stricter sampling
                        "top_k": 10         # Limit vocabulary to most likely words
                    }
                },
                timeout=600 
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
                return f"Error from Ollama: {response.text}"
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return f"Error: {str(e)}"

# Singleton
_engine = None

def get_ollama_engine():
    global _engine
    if _engine is None:
        _engine = OllamaVisionEngine(model_name="moondream") 
    return _engine