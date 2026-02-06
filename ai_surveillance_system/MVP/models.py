from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

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
    name = models.CharField(max_length=100)
    source = models.CharField(
        max_length=255,
        help_text="Camera index (0), RTSP URL, or video file path"
    )

    is_active = models.BooleanField(default=True)
    is_online = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class DetectionLog(models.Model):
    """Model to store detection data from live video"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(default=timezone.now)
    object_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    bbox_x = models.IntegerField()
    bbox_y = models.IntegerField()
    bbox_width = models.IntegerField()
    bbox_height = models.IntegerField()
    session_id = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['object_class']),
            models.Index(fields=['session_id']),
        ]
    
    def __str__(self):
        return f"{self.object_class} detected at {self.timestamp.strftime('%H:%M:%S')} (conf: {self.confidence:.2f})"
    
    @property
    def bbox(self):
        return [self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height]
    
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