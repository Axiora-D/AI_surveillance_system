from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from pathlib import Path
import base64 ,json , os ,sys,base64,cv2,numpy as np,uuid
from MVP.models import DetectionLog, UserProfile
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image
from ultralytics import YOLO
import psutil
from django.shortcuts import render
import pandas as pd 
import cv2
from ultralytics import YOLO
from django.http import StreamingHttpResponse
from django.utils import timezone
from MVP.utils.utils import (
    create_recording_session,
    create_video_writer,
    save_detections_to_excel,
    finalize_session,
    draw_detection_box
)
from django.shortcuts import render

def analytics(request):
    return render(request, 'analytics.html')



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MVP.utils.utils import process_video_django
from MVP.utils.inference_utils import (
    load_keras_model,
    load_yolo_model,
    crop_from_box,
    preprocess_for_classifier,
    draw_annotations,
    CLASS_MAP,
)
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .utils.two_factor import TwoFactorAuth


def dashboard(request):
    return render(request, "dashboard.html")

@login_required
def setup_2fa(request):
    auth = TwoFactorAuth()
    if request.method == 'POST':
        # Handle 2FA verification
        pass
    else:
        # Show QR code
        secret = auth.generate_secret()
        qr_code = auth.get_qr_code(request.user.email, secret)
        return render(request, 'setup_2fa.html', {
            'qr_code': qr_code,
            'secret': secret
        })
    


# Reuse the same YOLO model across requests to match the working test2_yolo pipeline
YOLO_MODEL_PATH = Path(r"C:\Users\Asus\OneDrive\Desktop\ai_surveillance\models\yolov8n.pt")
_yolo_model = None

# Face detection YOLO model (for gender classification within person detections)
YOLO_FACE_MODEL_PATH = Path(r"C:\Users\Asus\OneDrive\Desktop\ai_surveillance\models\yolo11m-pose.pt")  # Use YOLO11n for face detection
_yolo_face_model = None

# Gender classifier assets
KERAS_MODEL_PATH = getattr(settings, "KERAS_GENDER_MODEL", Path(settings.BASE_DIR) / "models" / "Gender_prediction_final.keras")
THRESHOLD_PATH = getattr(settings, "KERAS_THRESHOLD_PATH", Path(settings.BASE_DIR) / "models" / "best_threshold.txt")
_keras_model = None
_keras_threshold = 0.5


def get_yolo_model():
    """Load and cache the YOLO model so we do not re-load it per request."""
    global _yolo_model
    if _yolo_model is None:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
    return _yolo_model


def get_yolo_face_model():
    """Load and cache the face detection YOLO model."""
    global _yolo_face_model
    if _yolo_face_model is None:
        # Try to load face model, fallback to main model if not available
        if YOLO_FACE_MODEL_PATH.exists():
            _yolo_face_model = YOLO(str(YOLO_FACE_MODEL_PATH))
        else:
            # Fallback to main model if face model not found
            _yolo_face_model = get_yolo_model()
    return _yolo_face_model


def get_gender_model():
    """Lazy-load the gender classifier and threshold with graceful fallback."""
    global _keras_model, _keras_threshold
    if _keras_model is None:
        try:
            _keras_model, _keras_threshold = load_keras_model(str(KERAS_MODEL_PATH), str(THRESHOLD_PATH))
        except Exception as e:
            print(f"Warning: Could not load gender model: {e}. Gender detection disabled.")
            _keras_model = None
            _keras_threshold = 0.5
    return _keras_model, _keras_threshold

@login_required
def options(request):
    return render(request, 'options.html')

@login_required
def settings(request):
    """System settings page with 2FA integration"""
    try:
        profile, created = UserProfile.objects.get_or_create(user=request.user)
    except:
        profile = UserProfile.objects.create(user=request.user)
    
    # Generate secret if not exists (for QR code display)
    if not profile.two_factor_secret:
        auth = TwoFactorAuth()
        profile.two_factor_secret = auth.generate_secret()
        profile.save()
    
    # Generate QR code
    auth = TwoFactorAuth()
    qr_code = auth.get_qr_code(
        user_email=request.user.email or request.user.username,
        secret=profile.two_factor_secret,
        app_name="AI Surveillance System"
    )
    
    context = {
        'qr_code': qr_code,
        'secret': profile.two_factor_secret,
        'user_email': request.user.email or request.user.username,
        'two_factor_enabled': profile.two_factor_enabled,
        'profile': profile,
    }
    
    return render(request, 'settings.html', context)

@login_required
def live(request):
    return render(request, 'live.html')

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

@csrf_exempt
@require_http_methods(["POST"])
def process_video_yolo(request):
    """API endpoint for YOLO video processing"""
    try:
        data = json.loads(request.body)
        video_path = data.get('video_path')
        
        if not video_path or not os.path.exists(video_path):
            return JsonResponse({'error': 'Video file not found'}, status=400)
        
        # Load YOLO model (shared instance, same as working test2_yolo)
        model = get_yolo_model()
        
        # Process video with YOLO
        detections = process_video_django(video_path, detection_model=model)
        
        # Filter for people, vehicles, and other objects
        target_objects = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'train']
        filtered_detections = [d for d in detections if d.get('class_name') in target_objects]
        
        return JsonResponse({
            'success': True,
            'detections': filtered_detections,
            'total_detections': len(filtered_detections),
            'message': 'YOLO processing completed'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
@require_http_methods(["POST"])
def process_frame_yolo(request):
    """API endpoint for YOLO frame processing with optional gender detection"""
    try:
        data = json.loads(request.body)
        frame_data = data.get('frame_data')  # Base64 encoded frame
        session_id = data.get('session_id', '')
        enable_gender = data.get('enable_gender', True)  # Enable gender detection by default
        
        if not frame_data:
            return JsonResponse({'error': 'No frame data provided'}, status=400)
        
        # Load YOLO model (shared instance, same as working test2_yolo)
        model = get_yolo_model()
        
        # Load gender models if enabled
        face_model = None
        keras_model = None
        keras_threshold = 0.5
        if enable_gender:
            try:
                face_model = get_yolo_face_model()
                keras_model, keras_threshold = get_gender_model()
            except Exception as e:
                print(f"Warning: Gender detection disabled due to error: {e}")
                enable_gender = False
        
        # Decode frame from base64
        
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'error': 'Invalid image data'}, status=400)
        
        # Run YOLO detection
        results = model(frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    # Detect people, vehicles, and other objects
                    if class_name in ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'train']:
                        detection_data = {
                            'class_name': class_name,
                            'confidence': float(confidence),
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                        }
                        
                        # Add gender detection for person detections
                        if class_name == 'person' and enable_gender and face_model and keras_model:
                            try:
                                # Crop person region
                                person_crop = crop_from_box(frame, [x1, y1, x2, y2])
                                if person_crop is not None:
                                    # Detect faces within person crop
                                    face_results = face_model(person_crop, conf=0.25, verbose=False)
                                    if face_results and len(face_results) > 0:
                                        face_res = face_results[0]
                                        if face_res.boxes is not None and len(face_res.boxes) > 0:
                                            # Take the first (most confident) face
                                            face_box = face_res.boxes[0]
                                            fx1, fy1, fx2, fy2 = face_box.xyxy[0].cpu().numpy()
                                            
                                            # Adjust face coordinates to full frame coordinates
                                            fx1_full = int(x1 + fx1)
                                            fy1_full = int(y1 + fy1)
                                            fx2_full = int(x1 + fx2)
                                            fy2_full = int(y1 + fy2)
                                            
                                            # Crop face region
                                            face_crop = crop_from_box(person_crop, [fx1, fy1, fx2, fy2])
                                            if face_crop is not None:
                                                # Classify gender
                                                x_in = preprocess_for_classifier(face_crop)
                                                gender_score = float(keras_model.predict(x_in, verbose=0).flatten()[0])
                                                pred_label = 1 if gender_score > keras_threshold else 0
                                                gender_label = CLASS_MAP[pred_label]
                                                
                                                detection_data['gender'] = gender_label
                                                detection_data['gender_score'] = float(gender_score)
                                                detection_data['gender_confidence'] = float(gender_score) if pred_label == 1 else float(1 - gender_score)
                            except Exception as e:
                                print(f"Error in gender detection: {e}")
                                # Continue without gender info
                        
                        detections.append(detection_data)
                        
                        # Log detection to database
                        if request.user.is_authenticated:
                            print(f"DEBUG: Saving detection for user {request.user.username}")
                            DetectionLog.objects.create(
                                user=request.user,
                                object_class=class_name,
                                confidence=float(confidence),
                                bbox_x=int(x1),
                                bbox_y=int(y1),
                                bbox_width=int(x2-x1),
                                bbox_height=int(y2-y1),
                                session_id=session_id
                            )
                        else:
                            print(f"DEBUG: User not authenticated, skipping database save")
        
        return JsonResponse({
            'success': True,
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def live_recent_detections(request):
    """Return the most recent detections for the current user (for live UI polling)."""
    logs = DetectionLog.objects.filter(user=request.user).order_by("-timestamp")[:10]

    return JsonResponse(
        {
            "detections": [
                {
                    "object": l.object_class,
                    "confidence": round(l.confidence * 100, 2),
                    "time": l.timestamp.strftime("%H:%M:%S"),
                }
                for l in logs
            ]
        }
    )


@login_required
def upload_and_detect(request):
    """
    Upload an image, run YOLO11 detection + Keras gender classifier, and render results.
    Supports JSON output when ?format=json is provided.
    """
    if request.method == "POST" and request.FILES.get("image"):
        f = request.FILES["image"]
        save_name = default_storage.save(f"uploads/{f.name}", f)
        abs_path = default_storage.path(save_name)

        # Read image (OpenCV)
        img_bgr = cv2.imread(abs_path)
        if img_bgr is None:
            return JsonResponse({"error": "Cannot read uploaded image"}, status=400)

        # Load models
        yolo_model = get_yolo_model()
        keras_model, best_threshold = get_gender_model()

        # Run YOLO detection
        results = yolo_model.predict(source=abs_path, imgsz=640, conf=0.25, iou=0.45)
        res = results[0]

        detections = []
        for box_obj in res.boxes:
            xyxy = box_obj.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box_obj.conf[0])

            crop = crop_from_box(img_bgr, xyxy)
            if crop is None:
                continue

            x_in = preprocess_for_classifier(crop)  # shape (1,224,224,3)
            score = float(keras_model.predict(x_in, verbose=0).flatten()[0])  # prob for Female (1)
            pred_label = 1 if score > best_threshold else 0
            detections.append({
                "box": [int(v) for v in xyxy],
                "yolo_conf": conf,
                "gender_score": score,
                "label": CLASS_MAP[pred_label],
            })

        # Annotate image
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        annotated = draw_annotations(pil_img, detections)

        out_name = f"uploads/out_{f.name}"
        out_path = default_storage.path(out_name)
        annotated.save(out_path)

        if request.GET.get("format") == "json":
            return JsonResponse({"detections": detections, "annotated_url": default_storage.url(out_name)})

        return render(request, "result.html", {"out_url": default_storage.url(out_name), "detections": detections})

    return render(request, "upload.html")

@login_required
def get_detection_report(request):
    """Get detection report data"""
    try:
        from MVP.models import DetectionLog
        from django.db.models import Count, Avg, Max, Min
        from datetime import datetime, timedelta
        
        # Get query parameters
        hours = int(request.GET.get('hours', 24))  # Default to last 24 hours
        session_id = request.GET.get('session_id', '')
        object_class = request.GET.get('object_class', '')
        min_confidence = float(request.GET.get('min_confidence', 0))
        start_date = request.GET.get('start_date', '')
        end_date = request.GET.get('end_date', '')
        
        # Calculate time range
        end_time = timezone.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Override time range if custom dates provided
        if start_date:
            try:
                start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                pass  # Use default start_time
        
        if end_date:
            try:
                end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                pass  # Use default end_time
        
        # Base queryset
        queryset = DetectionLog.objects.filter(
            user=request.user,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        )
        
        # Apply filters
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        
        if object_class:
            queryset = queryset.filter(object_class=object_class)
        
        if min_confidence > 0:
            queryset = queryset.filter(confidence__gte=min_confidence)
        
        # Get recent detections
        recent_detections = queryset.order_by('-timestamp')[:50]
        
        # Get statistics
        stats = queryset.aggregate(
            total_detections=Count('id'),
            avg_confidence=Avg('confidence'),
            max_confidence=Max('confidence'),
            min_confidence=Min('confidence')
        )
        
        # Get detections by class
        class_stats = queryset.values('object_class').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence')
        ).order_by('-count')
        
        # Get hourly distribution (oldest first)
        hourly_stats = []
        for i in range(hours-1, -1, -1):  # Reverse the loop to get oldest first
            hour_start = end_time - timedelta(hours=i+1)
            hour_end = end_time - timedelta(hours=i)
            hour_count = queryset.filter(
                timestamp__gte=hour_start,
                timestamp__lt=hour_end
            ).count()
            hourly_stats.append({
                'hour': hour_start.strftime('%H:00'),
                'count': hour_count
            })
        
        return JsonResponse({
            'success': True,
            'recent_detections': [
                {
                    'id': d.id,
                    'object_class': d.object_class,
                    'confidence': d.confidence,
                    'bbox': d.bbox,
                    'timestamp': d.timestamp.isoformat(),
                    'session_id': d.session_id
                } for d in recent_detections
            ],
            'statistics': stats,
            'class_breakdown': list(class_stats),
            'hourly_distribution': hourly_stats,
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'hours': hours
            }
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def analytics(request):
    """Analytics dashboard view"""
    return render(request, 'analytics.html')

def signup(request):
    """User registration/signup view - This is now the home page"""
    if request.user.is_authenticated:
        return redirect('options')
    
    if request.method == 'POST':
        from django.contrib.auth.forms import UserCreationForm
        from django.contrib.auth import login
        
        # Create user using Django's UserCreationForm
        form = UserCreationForm(request.POST)
        
        if form.is_valid():
            user = form.save()
            
            # Create UserProfile for the new user
            UserProfile.objects.get_or_create(user=user)
            
            # Automatically log in the user
            login(request, user)
            messages.success(request, 'Account created successfully! Welcome to the Surveillance System.')
            return redirect('options')
        else:
            # Show form errors in a user-friendly way
            for field, errors in form.errors.items():
                for error in errors:
                    # Format field names nicely
                    field_name = field.replace('_', ' ').title()
                    if field == 'password1':
                        field_name = 'Password'
                    elif field == 'password2':
                        field_name = 'Password Confirmation'
                    messages.error(request, f"{field_name}: {error}")
    else:
        form = None
    
    return render(request, 'signup.html', {'form': form})

def debug_login(request):
    """Login view with 2FA support - users can login with their signup credentials"""
    if request.user.is_authenticated:
        return redirect('options')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            
            # Check if user has 2FA enabled
            try:
                from MVP.models import UserProfile
                profile = UserProfile.objects.get(user=user)
                if profile.two_factor_enabled:
                    return redirect('verify-2fa')
            except:
                pass
            
            return redirect('options')
        else:
            messages.error(request, 'Invalid credentials')
    return render(request, 'login.html')

def yolo_camera_stream(camera_index=0):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start recording session
    session_path, start_time = create_recording_session("CAM-01")
    writer, video_path = create_video_writer(session_path, fps, (width, height))
    detections_buffer = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    # Draw box
                    draw_detection_box(
                        frame,
                        x1, y1, x2 - x1, y2 - y1,
                        label,
                        conf
                    )

                    detections_buffer.append([
                        timezone.localtime(),
                        label,
                        conf,
                        x1, y1, x2 - x1, y2 - y1
                    ])

            # Save frame to video
            writer.write(frame)

            # Encode frame for browser
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpeg.tobytes() + b'\r\n'
            )

    finally:
        cap.release()
        writer.release()

        save_detections_to_excel(detections_buffer, session_path)
        finalize_session(session_path, start_time)

def live_stream_view(request):
    return StreamingHttpResponse(
        yolo_camera_stream(0),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
def finalize_session(session_path, start_time_str):
    end_time_str = timezone.localtime().strftime("%H-%M-%S")
    final_path = f"{session_path}_to_{end_time_str}"
    os.rename(session_path, final_path)
    return final_path, end_time_str

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
