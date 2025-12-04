import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
from datetime import datetime
import threading
from collections import deque
import time

class VideoProcessor:
    """Real-time video processing with object detection"""
    
    def __init__(self, yolo_model):
        self.yolo_model = yolo_model
        self.processing = False
        self.frame_queue = deque(maxlen=30)  # Last 30 frames
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'fps': 0,
            'avg_inference_time': 0,
            'detections_count': 0
        }
    
    def process_video(self, input_path, output_path=None, conf_threshold=0.5):
        """Process video file with detection"""
        print(f"📹 Loading video: {input_path}")
        
        # Load video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info - FPS: {fps}, Resolution: {width}x{height}, Total Frames: {total_frames}")
        
        # Setup output video writer if output path provided
        output_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        inference_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            
            results = self.yolo_model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False
            )
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Draw bounding boxes
            annotated_frame = frame.copy()
            frame_detections = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf)
                        cls = int(box.cls)
                        class_name = self.yolo_model.names[cls]
                        
                        # Draw bounding box (green)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with background
                        label = f"{class_name} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_size[1] - 8),
                            (x1 + label_size[0] + 8, y1),
                            (0, 255, 0),
                            -1
                        )
                        cv2.putText(
                            annotated_frame, label, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                        )
                        
                        total_detections += 1
                        frame_detections += 1
            
            # Add FPS and frame counter
            current_fps = 1 / inference_time if inference_time > 0 else 0
            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames} | Detections: {frame_detections}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated_frame,
                timestamp,
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
            # Write frame to output
            if output_writer:
                output_writer.write(annotated_frame)
            
            # Store frame
            self.frame_queue.append(annotated_frame)
            
            # Progress update every 30 frames
            if frame_count % 30 == 0:
                progress_pct = 100 * frame_count / total_frames
                print(f"⏳ Processing: {frame_count}/{total_frames} ({progress_pct:.1f}%)")
        
        # Cleanup
        cap.release()
        if output_writer:
            output_writer.release()
        
        # Calculate stats
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        avg_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
        
        stats = {
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'avg_fps': round(avg_fps, 2),
            'avg_inference_time': round(avg_inference_time, 4),
            'total_detections': total_detections,
            'output_path': output_path,
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'duration': round(total_frames / fps, 2) if fps > 0 else 0
            }
        }
        
        print(f"✅ Video processing complete!")
        print(f"📊 Stats: {frame_count} frames, {total_detections} detections, {avg_fps:.1f} avg FPS")
        
        return stats
    
    def process_video_streaming(self, input_path, conf_threshold=0.5, callback=None):
        """Process video with streaming updates (for WebSocket)"""
        print(f"📹 Loading video (streaming mode): {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"error": "Cannot open video file"}
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.yolo_model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False
            )
            
            # Extract detections
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detections.append({
                            'bbox': box.xyxy[0].tolist(),
                            'confidence': float(box.conf),
                            'class': int(box.cls),
                            'class_name': self.yolo_model.names[int(box.cls)]
                        })
            
            # Call callback with frame data
            if callback:
                callback({
                    'frame_number': frame_count,
                    'total_frames': total_frames,
                    'progress': round((frame_count / total_frames) * 100, 2),
                    'detections': detections,
                    'detection_count': len(detections)
                })
        
        cap.release()
        return {'status': 'completed', 'total_frames': frame_count}


# Singleton pattern for video processor
_processor = None

def get_video_processor(yolo_model):
    """Get or create video processor instance"""
    global _processor
    if _processor is None:
        _processor = VideoProcessor(yolo_model)
    return _processor