import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
import time
from segment_anything import sam_model_registry, SamPredictor

class SAMSegmentationEngine:
    """
    Segment Anything Model Integration for VisLang
    Enables pixel-level interactive segmentation
    CPU-OPTIMIZED VERSION (ViT-B model)
    """
    
    def __init__(self, model_type="vit_b", device="cpu"):
        """
        Initialize SAM model
        
        Args:
            model_type: "vit_b" (small - CPU optimized)
            device: "cpu" (CPU only - no GPU)
        """
        # Force CPU (no GPU available)
        self.device = "cpu"
        self.model_type = "vit_b"
        
        # HARDCODED checkpoint for ViT-B
        checkpoint = "models/sam/sam_vit_b_01ec64.pth"
        
        try:
            # Load model
            print(f"📦 Loading SAM model (vit_b) on cpu...")
            sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
            sam.to(device=self.device)
            
            self.predictor = SamPredictor(sam)
            self.image_array = None
            self.current_image_path = None
            
            print(f"✅ SAM Model Loaded successfully!")
            print(f"   - Model: vit_b")
            print(f"   - Device: cpu")
            print(f"   - Ready for inference")
        
        except Exception as e:
            print(f"❌ Error loading SAM: {str(e)}")
            raise
    
    def process_image(self, image_path):
        """
        Load and prepare image for segmentation
        
        Args:
            image_path: Path to image file
            
        Returns:
            image_array: Loaded image as numpy array
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            self.image_array = np.array(image)
            self.current_image_path = str(image_path)
            
            # Set image for SAM predictor
            self.predictor.set_image(self.image_array)
            
            print(f"✅ Image loaded: {image_path.name if hasattr(image_path, 'name') else image_path}")
            print(f"   - Shape: {self.image_array.shape}")
            print(f"   - Ready for prompts")
            
            return self.image_array
        
        except Exception as e:
            print(f"❌ Error loading image: {str(e)}")
            raise
    
    def segment_with_point(self, points, labels, image_path):
        """
        Segment image using point prompts (click-based)
        
        Args:
            points: List of [x, y] coordinates [[x1, y1], [x2, y2], ...]
            labels: List of labels [1=include, 0=exclude] (same length as points)
            image_path: Path to image
            
        Returns:
            Dictionary with masks, scores, and metadata
        """
        try:
            # Prepare image
            self.process_image(image_path)
            
            # Convert to numpy arrays
            input_point = np.array(points)
            input_label = np.array(labels)
            
            print(f"🎯 Processing {len(points)} point prompt(s)...")
            start_time = time.time()
            
            # Run inference
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True  # Returns 3 mask candidates
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Segmentation complete in {elapsed:.2f}s")
            print(f"   - Masks generated: {len(masks)}")
            print(f"   - Best confidence: {scores[0]:.4f}")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits,
                'points': points,
                'labels': labels,
                'processing_time': elapsed
            }
        
        except Exception as e:
            print(f"❌ Error in segmentation: {str(e)}")
            raise
    
    def segment_with_box(self, box, image_path):
        """
        Segment image using bounding box prompt
        
        Args:
            box: [x_min, y_min, x_max, y_max]
            image_path: Path to image
            
        Returns:
            Dictionary with masks, scores, and metadata
        """
        try:
            self.process_image(image_path)
            
            input_box = np.array(box)
            
            print(f"📦 Processing bounding box: {box}")
            start_time = time.time()
            
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Box segmentation complete in {elapsed:.2f}s")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits,
                'box': box,
                'processing_time': elapsed
            }
        
        except Exception as e:
            print(f"❌ Error in box segmentation: {str(e)}")
            raise
    
    @staticmethod
    def visualize_mask(image, mask, alpha=0.5):
        """
        Create visualization of mask overlay on image
        
        Args:
            image: Input image (BGR format from OpenCV)
            mask: Binary mask
            alpha: Transparency of overlay
            
        Returns:
            Visualization image
        """
        try:
            # Ensure image is BGR
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Create colored mask (green)
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = [0, 255, 0]  # Green
            
            # Blend with original image
            result = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
            
            # Draw contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            return result
        
        except Exception as e:
            print(f"❌ Error in visualization: {str(e)}")
            raise
    
    @staticmethod
    def extract_region(image, mask):
        """
        Extract segmented region from image
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Cropped region with transparency
        """
        try:
            # Convert to RGBA
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Create RGBA image
            b, g, r = cv2.split(image)
            a = np.where(mask, 255, 0).astype(np.uint8)
            extracted = cv2.merge([b, g, r, a])
            
            # Get bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not (np.any(rows) and np.any(cols)):
                print("⚠️ Empty mask - no region to extract")
                return None
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Crop
            cropped = extracted[ymin:ymax+1, xmin:xmax+1]
            
            print(f"✅ Region extracted: {cropped.shape}")
            
            return cropped
        
        except Exception as e:
            print(f"❌ Error in region extraction: {str(e)}")
            raise
    
    @staticmethod
    def mask_to_image(mask):
        """
        Convert binary mask to image (0=black, 1=white)
        """
        return (mask * 255).astype(np.uint8)


# Test function
if __name__ == "__main__":
    print("Testing SAM Engine...")
    
    try:
        engine = SAMSegmentationEngine(model_type="vit_b", device="cpu")
        print("✅ SAM Engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize SAM: {e}")