from django.contrib import admin
from .models import UserProfile, Camera, DetectionLog, RecordingSession, VideoPath, RecordingPath


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'two_factor_enabled', 'two_factor_secret')
    search_fields = ('user__username', 'user__email')
    raw_id_fields = ('user',)


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ('id', 'camera_id', 'name', 'user', 'source', 'is_active', 'is_online', 'created_at')
    list_filter = ('is_active', 'is_online')
    search_fields = ('camera_id', 'name', 'source', 'user__username')
    raw_id_fields = ('user',)
    list_editable = ('camera_id', 'name', 'is_active', 'is_online')


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'cam_id',
        'object_class',
        'detection_type',
        'video_path_short',
        'recording_path_short',
        'timestamp',
        'confidence',
    )
    list_filter = ('detection_type', 'object_class', 'timestamp')
    search_fields = ('user__username', 'cam_id', 'object_class', 'session_id', 'video_path', 'recording_path')
    raw_id_fields = ('user',)
    readonly_fields = ('timestamp', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'detected_items')
    date_hierarchy = 'timestamp'

    def video_path_short(self, obj):
        if not obj.video_path:
            return '-'
        return obj.video_path[:50] + '...' if len(obj.video_path) > 50 else obj.video_path
    video_path_short.short_description = 'Video path'

    def recording_path_short(self, obj):
        if not obj.recording_path:
            return '-'
        return obj.recording_path[:50] + '...' if len(obj.recording_path) > 50 else obj.recording_path
    recording_path_short.short_description = 'Recording path'


@admin.register(RecordingSession)
class RecordingSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'camera', 'start_time', 'end_time', 'video_path_short', 'folder_path_short')
    list_filter = ('start_time',)
    search_fields = ('user__username', 'camera__name', 'camera__camera_id', 'folder_path', 'video_path')
    raw_id_fields = ('user', 'camera')
    readonly_fields = ('start_time', 'folder_path', 'video_path', 'excel_path')

    def video_path_short(self, obj):
        if not obj.video_path:
            return '-'
        return obj.video_path[:60] + '...' if len(obj.video_path) > 60 else obj.video_path
    video_path_short.short_description = 'Recording path (video)'

    def folder_path_short(self, obj):
        if not obj.folder_path:
            return '-'
        return obj.folder_path[:60] + '...' if len(obj.folder_path) > 60 else obj.folder_path
    folder_path_short.short_description = 'Folder path'


@admin.register(VideoPath)
class VideoPathAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'source_name', 'path', 'created_at')
    search_fields = ('user__username', 'source_name', 'path')
    raw_id_fields = ('user',)
    readonly_fields = ('created_at',)


@admin.register(RecordingPath)
class RecordingPathAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'camera', 'session_id', 'path', 'created_at')
    search_fields = ('user__username', 'camera__name', 'camera__camera_id', 'session_id', 'path')
    raw_id_fields = ('user', 'camera')
    readonly_fields = ('created_at',)
