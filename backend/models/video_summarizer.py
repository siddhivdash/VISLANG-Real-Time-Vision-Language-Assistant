import cv2
import numpy as np
from PIL import Image, ImageStat
import os
from models.ollama_vision import get_ollama_engine

class VideoSummarizer:
    def __init__(self):
        self.llava = get_ollama_engine()

    def is_image_black(self, pil_img):
        """Check if an image is mostly black/empty"""
        stat = ImageStat.Stat(pil_img)
        if sum(stat.mean) / len(stat.mean) < 10: 
            return True
        return False

    def summarize(self, video_path):
        print(f"🎬 Summarizing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError("Video has 0 frames or is corrupted")

        # Extract 4 keyframes (Reduced from 6 to prevent model confusion)
        indices = np.linspace(0, total_frames - 1, 4, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Check for black frames
                if self.is_image_black(pil_img):
                    print(f"⚠️ Warning: Skipped a black frame at index {idx}")
                    continue

                # FIX: Use thumbnail to preserve aspect ratio (Supports Vertical Videos)
                # This ensures turtles don't get squashed into a wide rectangle
                pil_img.thumbnail((512, 512)) 
                frames.append(pil_img)
        
        cap.release()

        if not frames:
            raise ValueError("Could not extract any valid frames")

        # Create Grid Image
        # Dynamically calculate grid size based on actual frame size
        w, h = frames[0].size
        cols = 2
        rows = 2
        
        grid = Image.new('RGB', (w * cols, h * rows))
        
        for i, img in enumerate(frames):
            if i >= cols * rows: break
            x = (i % cols) * w
            y = (i // cols) * h
            grid.paste(img, (x, y))

        # Save debug image (Check this file to see if turtles look correct!)
        grid.save("debug_summary_grid.jpg")
        print("✅ Grid saved to debug_summary_grid.jpg")

        # FIX: Generic Prompt
        # We removed the "City Street" instructions so it works for ANY video.
        prompt = (
            "Analyze this sequence of video frames. "
            "Describe the main subjects (people, animals, or vehicles) and their actions. "
            "What is the setting?"
        )

        return self.llava.chat(grid, prompt)

# Singleton pattern
_summarizer = None
def get_video_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = VideoSummarizer()
    return _summarizer