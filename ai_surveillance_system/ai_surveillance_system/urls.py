"""
URL configuration for ai_surveillance_system project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from MVP import views
from MVP import views, two_factor_views


urlpatterns = [
    path('admin/', admin.site.urls),
    # Auth
    path('login/', views.debug_login, name='login'),  # Using custom login with 2FA support
    path('signup/', views.signup, name='signup'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # 2FA
    path('2fa/setup/', two_factor_views.setup_2fa, name='setup-2fa'),
    path('2fa/verify-setup/', two_factor_views.verify_2fa_setup, name='verify-2fa-setup'),
    path('2fa/verify/', two_factor_views.verify_2fa_login, name='verify-2fa'),
    path('2fa/disable/', two_factor_views.disable_2fa, name='disable-2fa'),
    path('2fa/regenerate-codes/', two_factor_views.regenerate_backup_codes, name='regenerate-backup-codes'),

    # App
    path('', views.signup, name='home'),  # Show signup page first
    path('options/', views.options, name='options'),
    path('settings/', views.settings, name='settings'),
    path('live/', views.live, name='live'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('analytics/', views.analytics, name='analytics'),
    path('upload-detect/', views.upload_and_detect, name='upload_and_detect'),
     path("live-stream/", views.live_stream_view, name="live_stream"),
    
    # API endpoints
    path('api/process-video/', views.process_video_yolo, name='process_video_yolo'),
    path('api/process-frame/', views.process_frame_yolo, name='process_frame_yolo'),
    path('api/live-detections/', views.live_recent_detections, name='live_recent'),
    path('api/detection-report/', views.get_detection_report, name='detection_report'),
]