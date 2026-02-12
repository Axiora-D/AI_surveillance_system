from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    two_factor_secret = models.CharField(max_length=255, blank=True)
    two_factor_enabled = models.BooleanField(default=False)
    backup_codes = models.TextField(blank=True, help_text="JSON list of backup codes")
    
    def __str__(self):
        return f"{self.user.username} Profile"
    
    def get_backup_codes(self):
        """Get backup codes as a list"""
        import json
        if self.backup_codes:
            return json.loads(self.backup_codes)
        return []
    
    def set_backup_codes(self, codes):
        """Set backup codes from a list"""
        import json
        self.backup_codes = json.dumps(codes)
    
    def generate_backup_codes(self):
        """Generate 10 backup codes"""
        import secrets
        import string
        codes = []
        for _ in range(10):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            codes.append(code)
        self.set_backup_codes(codes)
        return codes
    
class Camera(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    camera_id = models.CharField(max_length=50, blank=True, help_text="Camera ID e.g. CAM-01")
    name = models.CharField(max_length=100)
    source = models.CharField(
        max_length=255,
        help_text="Camera index (0), RTSP URL, or video file path"
    )

    is_active = models.BooleanField(default=True)
    is_online = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.camera_id or self.name

class DetectionLog(models.Model):
    """Model to store detection data from live video"""
    # Detection Id is automatically provided by Django as 'id' (primary key)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(default=timezone.now)
    object_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    bbox_x = models.IntegerField()
    bbox_y = models.IntegerField()
    bbox_width = models.IntegerField()
    bbox_height = models.IntegerField()
    session_id = models.CharField(max_length=100, blank=True)
    
    # New fields as requested
    cam_id = models.CharField(max_length=100, blank=True, help_text="Camera ID or video identifier")
    detection_type = models.CharField(max_length=50, blank=True, help_text="Type of detection: 'video' or 'live'")
    detected_items = models.JSONField(default=list, blank=True, help_text="All detected items in this frame as JSON array")
    video_path = models.CharField(max_length=500, blank=True, help_text="Path to video file when detection is from uploaded video")
    recording_path = models.CharField(max_length=500, blank=True, help_text="Path to recording file when detection is from live recording")
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['object_class']),
            models.Index(fields=['session_id']),
            models.Index(fields=['cam_id']),
            models.Index(fields=['detection_type']),
        ]
    
    def __str__(self):
        return f"{self.object_class} detected at {self.timestamp.strftime('%H:%M:%S')} (conf: {self.confidence:.2f})"
    
    @property
    def bbox(self):
        return [self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height]
    
    @property
    def detection_id(self):
        """Return the detection ID (primary key)"""
        return self.id
    
    def get_detected_items(self):
        """Get detected_items as a list"""
        if isinstance(self.detected_items, str):
            return json.loads(self.detected_items)
        return self.detected_items or []
    
class RecordingSession(models.Model):
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)

    folder_path = models.CharField(max_length=500)
    video_path = models.CharField(max_length=500)
    excel_path = models.CharField(max_length=500)

    def __str__(self):
        return f"{self.camera.name} | {self.start_time.strftime('%H:%M:%S')}"


class VideoPath(models.Model):
    """
    Table to store paths of processed/annotated videos (detection via video).
    Visible in Django admin for auditing which videos were processed.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    path = models.CharField(max_length=500, help_text="Filesystem path to processed/annotated video")
    created_at = models.DateTimeField(auto_now_add=True)
    source_name = models.CharField(max_length=255, blank=True, help_text="Original video filename or identifier")

    def __str__(self):
        return f"Video: {self.source_name or self.path}"


class RecordingPath(models.Model):
    """
    Table to store paths of saved live recordings.
    Visible in Django admin so admins can quickly see all recording files.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True, blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    path = models.CharField(max_length=500, help_text="Filesystem path to saved recording (video.mp4)")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        cam_label = self.camera.camera_id or self.camera.name if self.camera else "Unknown camera"
        return f"Recording: {cam_label} ({self.session_id})"
