# VISLANG – Real-Time Vision & Language Assistant

VISLANG is a real-time AI system that combines **Object Detection**, **Video Processing**, and **Vision-Language Conversational Understanding** into a single application.  
It is built for **learning**, **research**, and **production-ready experimentation**.

The project demonstrates how to integrate:
- YOLO Object Detection  
- Frame-by-frame Video Processing  
- Vision-Language Models (VLMs)  
- FastAPI Backend  
- Lightweight Frontend (HTML / CSS / JavaScript)

The goal of VISLANG is simple:  
**Learn how modern AI systems are built end-to-end.**


---

## 🚀 Features

### ✔️ 1. Image Object Detection
- Upload images and detect objects instantly  
- Returns bounding boxes, labels, and confidence scores  
- Adjustable detection threshold  
- Powered by **Ultralytics YOLO models**

---

### ✔️ 2. Video Processing (Frame-by-Frame)
- Upload any video (MP4, AVI, MKV, etc.)
- Extracts frames using OpenCV
- Runs YOLO on every frame
- Displays:
  - Total frames  
  - FPS achieved  
  - Total detected objects  
  - Processing duration  
- Download the processed, annotated video

---

### ✔️ 3. Vision–Language Chat Assistant
Upload an image and chat with an AI that can **see**, **interpret**, and **reason**.

Ask questions like:
- “Describe this image”
- “What objects do you see?”
- “How many people and vehicles are present?”
- “What is happening in this scene?”

### 🔥 GPU Upgrade Option  
If your system has a **powerful GPU**, you can load **larger VLMs** such as:
- LLaVA 1.6
- Qwen-VL
- Phi-3 Vision
- InternVL2
- Any OpenVLM compatible model  

Better GPU = Better reasoning + faster responses.

If your system is weak, lightweight models are used automatically.

---

# 🧩 Technologies & Tools Used

VISLANG focuses heavily on explaining the tools involved so learners can understand the entire pipeline.

---

## ⚙️ Backend Technologies

### **FastAPI**
- High-performance backend  
- Async support  
- Built-in API documentation  
- Ideal for ML model serving  

### **YOLO Object Detection**
Used for both image and video recognition.
- Fast real-time predictions  
- Supports multiple model sizes (nano → xlarge)  
- Flexible for low or high GPU power  

### **Vision-Language Models (VLMs)**
Used for the chat feature.  
The model:
- Accepts images + text  
- Performs reasoning  
- Outputs contextual answers  

### **OpenCV**
Handles:
- Reading video frames  
- Drawing bounding boxes  
- Exporting processed video  

### **Torch**
Backend for YOLO & VLM inference.

---

## 🎨 Frontend Technologies (No React)
The project uses a clean, lightweight frontend:

### **HTML5**
- Provides the interface structure  
- Forms, tabs, upload boxes  

### **CSS / Tailwind or Custom CSS**
- Modern UI  
- Responsive design  
- Minimal styling complexity  

### **JavaScript**
- Handles API requests to FastAPI  
- Updates results dynamically  
- Displays progress bars and detection metrics  

---

# 🔄 How VISLANG Works (Learning Explanation)

### **1. User uploads an image/video or sends a chat query**
HTML/JS frontend sends the request to FastAPI.

### **2. FastAPI receives the request**
Loads:
- YOLO model (for detection)
- Vision-Language model (for chat)

### **3. Inference happens**
- Image/video frames processed  
- Text + image reasoning for chat  

### **4. Results returned to frontend**
- Bounding boxes  
- Video analytics  
- VLM text response  

### **5. UI displays everything cleanly**

This end-to-end flow is how real AI products work.

---
