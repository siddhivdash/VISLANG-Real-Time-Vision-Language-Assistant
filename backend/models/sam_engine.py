import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
import time
from segment_anything import sam_model_registry, SamPredictor
import os

class SAMSegmentationEngine:
    def __init__(self, model_type="vit_b", device="cpu"):
        self.device = "cpu"
        self.model_type = "vit_b"

        # --- ROBUST PATH FINDING STRATEGY ---
        current_script_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_script_path)
        
        # Check all possible locations
        possible_paths = [
            os.path.join(current_dir, "models", "sam", "sam_vit_b_01ec64.pth"),
            "/app/backend/models/sam/sam_vit_b_01ec64.pth",
            os.path.join(os.getcwd(), "backend", "models", "sam", "sam_vit_b_01ec64.pth"),
            os.path.join(current_dir, "sam", "sam_vit_b_01ec64.pth")
        ]
        
        checkpoint = None
        print(f"🔍 Searching for SAM model...")
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint = path
                print(f"✅ Found model at: {checkpoint}")
                break
        
        if checkpoint is None:
            print(f"❌ CRITICAL ERROR: Model file not found.")
            raise FileNotFoundError(f"Could not find 'sam_vit_b_01ec64.pth'")

        try:
            print(f"📦 Loading SAM model (vit_b) on cpu...")
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            print(f"✅ SAM Model Loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading SAM: {str(e)}")
            raise
    
    # ... (Rest of your functions: process_image, segment_with_point, etc. Keep them as they were) ...
    # I am not pasting the rest to save space, but DO NOT DELETE the other methods from your file.
    
    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        self.image_array = np.array(image)
        self.predictor.set_image(self.image_array)
        return self.image_array

    def segment_with_point(self, points, labels, image_path):
        self.process_image(image_path)
        input_point = np.array(points)
        input_label = np.array(labels)
        masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
        return {'masks': masks, 'scores': scores}

    def segment_with_box(self, box, image_path):
        self.process_image(image_path)
        input_box = np.array(box)
        masks, scores, logits = self.predictor.predict(point_coords=None, point_labels=None, box=input_box, multimask_output=True)
        return {'masks': masks, 'scores': scores}