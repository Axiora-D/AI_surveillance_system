# utils.py - Utility functions for AI Surveillance System

import cv2
import numpy as np
from django.http import JsonResponse
from django.core.files.base import ContentFile
import base64
import io
from PIL import Image
import os
from django.conf import settings
import pandas as pd
from datetime import datetime
from django.utils import timezone

def create_recording_session(camera_name):
    """
    Create a recording folder based on date and start time
    """
    now = timezone.localtime()
    date_str = now.strftime("%Y-%m-%d")
    start_time_str = now.strftime("%H-%M-%S")

    base_dir = os.path.join(settings.MEDIA_ROOT, "recordings", date_str)
    os.makedirs(base_dir, exist_ok=True)

    folder_name = f"{camera_name}_{start_time_str}"
    session_path = os.path.join(base_dir, folder_name)
    os.makedirs(session_path, exist_ok=True)

    return session_path, start_time_str

def create_video_writer(session_path, fps, frame_size):
    video_path = os.path.join(session_path, "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    return writer, video_path

def save_detections_to_excel(detections, session_path):
    excel_path = os.path.join(session_path, "detections.xlsx")

    df = pd.DataFrame(detections, columns=[
        "timestamp",
        "object_class",
        "confidence",
        "x", "y", "width", "height"
    ])

    df.to_excel(excel_path, index=False)
    return excel_path

def finalize_session(session_path, start_time_str):
    end_time_str = timezone.localtime().strftime("%H-%M-%S")
    final_path = f"{session_path}_to_{end_time_str}"
    os.rename(session_path, final_path)
    return final_path, end_time_str


def process_frame(frame):
    """
    Process a video frame for surveillance analysis
    """
    # Convert BGR to RGB if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    return frame_rgb

def encode_frame_to_base64(frame):
    """
    Encode frame to base64 for web transmission
    """
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

def decode_base64_to_frame(base64_string):
    """
    Decode base64 string back to frame
    """
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding base64 frame: {e}")
        return None

def save_detection_image(frame, filename=None):
    """
    Save detection frame to media directory
    """
    if filename is None:
        import time
        filename = f"detection_{int(time.time())}.jpg"
    
    # Ensure media directory exists
    media_dir = os.path.join(settings.MEDIA_ROOT, 'detections')
    os.makedirs(media_dir, exist_ok=True)
    
    filepath = os.path.join(media_dir, filename)
    cv2.imwrite(filepath, frame)
    
    return os.path.join('detections', filename)  # Return relative path for URL

def create_response(success=True, message="", data=None):
    """
    Create standardized JSON response
    """
    response_data = {
        'success': success,
        'message': message,
    }
    
    if data is not None:
        response_data['data'] = data
    
    return JsonResponse(response_data)

def validate_image_upload(uploaded_file):
    """
    Validate uploaded image file
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    max_size = 10 * 1024 * 1024  # 10MB
    
    # Check file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in valid_extensions:
        return False, "Invalid file format. Please upload JPG, PNG, or BMP files."
    
    # Check file size
    if uploaded_file.size > max_size:
        return False, "File too large. Maximum size is 10MB."
    
    return True, "Valid image file."

def get_camera_info():
    """
    Get available camera information
    """
    cameras = []
    for i in range(4):  # Check first 4 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            cameras.append({
                'index': i,
                'width': width,
                'height': height,
                'fps': fps
            })
            cap.release()
    
    return cameras

def resize_frame(frame, max_width=640, max_height=480):
    """
    Resize frame while maintaining aspect ratio
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame
    
    return frame

def draw_detection_box(frame, x, y, w, h, label, confidence, color=(0, 255, 0)):
    """
    Draw detection bounding box on frame
    """
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw label
    label_text = f"{label}: {confidence:.2f}"
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Background for text
    cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
    
    # Text
    cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def process_video_django(video_path, output_path=None, detection_model=None):
    """
    Process video file with Django integration for AI surveillance
    """
    from ultralytics import YOLO
    import time
    
    # Load YOLO model if not provided
    if detection_model is None:
        detection_model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, etc.
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'success': False,
            'message': f'Could not open video file: {video_path}',
            'detections': []
        }
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output video writer if output path is provided
    out_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    detections = []
    frame_number = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection on frame
            results = detection_model(frame)
            
            # Process results
            frame_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = detection_model.names[class_id]
                        
                        # Store detection info
                        detection_info = {
                            'frame_number': frame_number,
                            'timestamp': frame_number / fps,
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # [x, y, width, height]
                        }
                        frame_detections.append(detection_info)
                        
                        # Draw detection box on frame
                        frame = draw_detection_box(
                            frame, 
                            int(x1), int(y1), int(x2-x1), int(y2-y1),
                            class_name, 
                            confidence
                        )
            
            detections.extend(frame_detections)
            
            # Write frame to output video if writer is available
            if out_writer:
                out_writer.write(frame)
            
            frame_number += 1
            
            # Optional: Print progress
            if frame_number % 30 == 0:  # Print every 30 frames
                progress = (frame_number / frame_count) * 100
                print(f"Processing video: {progress:.1f}% complete")
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error processing video: {str(e)}',
            'detections': detections
        }
    
    finally:
        # Clean up
        cap.release()
        if out_writer:
            out_writer.release()
    
    return {
        'success': True,
        'message': f'Video processed successfully. Found {len(detections)} detections.',
        'detections': detections,
        'video_info': {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps
        },
        'output_path': output_path if output_path else None
    }