# forms.py - Django Forms for AI Surveillance System

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, Field, HTML
from crispy_forms.bootstrap import FormActions

class VideoUploadForm(forms.Form):
    """Form for uploading video files for processing"""
    video_file = forms.FileField(
        label='Select Video File',
        help_text='Upload MP4, AVI, MOV files (max 100MB)',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'video/*'
        })
    )
    
    process_realtime = forms.BooleanField(
        label='Process in Real-time',
        required=False,
        help_text='Enable real-time processing (slower but shows live results)',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    detection_confidence = forms.FloatField(
        label='Detection Confidence Threshold',
        initial=0.5,
        min_value=0.1,
        max_value=1.0,
        help_text='Higher values = fewer false positives (0.1 - 1.0)',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'max': '1.0'
        })
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            Field('video_file'),
            Row(
                Column('process_realtime', css_class='form-group col-md-6'),
                Column('detection_confidence', css_class='form-group col-md-6'),
            ),
            FormActions(
                Submit('submit', 'Upload and Process Video', css_class='btn btn-primary')
            )
        )
    
    def clean_video_file(self):
        video_file = self.cleaned_data.get('video_file')
        if video_file:
            # Check file size (100MB limit)
            if video_file.size > 100 * 1024 * 1024:
                raise forms.ValidationError('Video file too large. Maximum size is 100MB.')
            
            # Check file extension
            allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            file_extension = video_file.name.lower().split('.')[-1]
            if f'.{file_extension}' not in allowed_extensions:
                raise forms.ValidationError('Invalid video format. Please upload MP4, AVI, MOV, MKV, or WMV files.')
        
        return video_file

class ImageUploadForm(forms.Form):
    """Form for uploading single images for detection"""
    image_file = forms.ImageField(
        label='Select Image File',
        help_text='Upload JPG, PNG files (max 10MB)',
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )
    
    detection_confidence = forms.FloatField(
        label='Detection Confidence',
        initial=0.5,
        min_value=0.1,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1'
        })
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_enctype = 'multipart/form-data'
        self.helper.layout = Layout(
            'image_file',
            'detection_confidence',
            Submit('submit', 'Analyze Image', css_class='btn btn-success')
        )

class CameraSettingsForm(forms.Form):
    """Form for configuring camera settings"""
    camera_index = forms.IntegerField(
        label='Camera Index',
        initial=0,
        min_value=0,
        max_value=10,
        help_text='Camera device index (usually 0 for built-in camera)',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    resolution_width = forms.IntegerField(
        label='Width',
        initial=640,
        min_value=320,
        max_value=1920,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    resolution_height = forms.IntegerField(
        label='Height',
        initial=480,
        min_value=240,
        max_value=1080,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    fps = forms.IntegerField(
        label='Frames Per Second',
        initial=30,
        min_value=5,
        max_value=60,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    detection_enabled = forms.BooleanField(
        label='Enable Object Detection',
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'camera_index',
            Row(
                Column('resolution_width', css_class='col-md-6'),
                Column('resolution_height', css_class='col-md-6'),
            ),
            Row(
                Column('fps', css_class='col-md-6'),
                Column('detection_enabled', css_class='col-md-6'),
            ),
            Submit('submit', 'Update Camera Settings', css_class='btn btn-primary')
        )

class UserRegistrationForm(UserCreationForm):
    """Extended user registration form"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add CSS classes to form fields
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
        
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'username',
            Row(
                Column('first_name', css_class='col-md-6'),
                Column('last_name', css_class='col-md-6'),
            ),
            'email',
            'password1',
            'password2',
            Submit('submit', 'Register', css_class='btn btn-success')
        )
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
        return user

class AlertSettingsForm(forms.Form):
    """Form for configuring surveillance alerts"""
    ALERT_TYPES = [
        ('email', 'Email'),
        ('sms', 'SMS'),
        ('webhook', 'Webhook'),
        ('desktop', 'Desktop Notification'),
    ]
    
    alert_types = forms.MultipleChoiceField(
        choices=ALERT_TYPES,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-check-input'}),
        label='Alert Methods'
    )
    
    email_address = forms.EmailField(
        required=False,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter email for alerts'
        })
    )
    
    phone_number = forms.CharField(
        max_length=15,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '+1234567890'
        })
    )
    
    webhook_url = forms.URLField(
        required=False,
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://your-webhook-url.com'
        })
    )
    
    detection_threshold = forms.IntegerField(
        label='Alert after X detections',
        initial=5,
        min_value=1,
        max_value=100,
        help_text='Number of detections before triggering alert',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            'alert_types',
            'email_address',
            'phone_number',
            'webhook_url',
            'detection_threshold',
            Submit('submit', 'Save Alert Settings', css_class='btn btn-warning')
        )
    
    def clean(self):
        cleaned_data = super().clean()
        alert_types = cleaned_data.get('alert_types', [])
        
        # Validate email is provided if email alerts are enabled
        if 'email' in alert_types and not cleaned_data.get('email_address'):
            raise forms.ValidationError('Email address is required for email alerts.')
        
        # Validate phone number is provided if SMS alerts are enabled
        if 'sms' in alert_types and not cleaned_data.get('phone_number'):
            raise forms.ValidationError('Phone number is required for SMS alerts.')
        
        # Validate webhook URL is provided if webhook alerts are enabled
        if 'webhook' in alert_types and not cleaned_data.get('webhook_url'):
            raise forms.ValidationError('Webhook URL is required for webhook alerts.')
        
        return cleaned_data

class SearchDetectionForm(forms.Form):
    """Form for searching through detection history"""
    date_from = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={
            'class': 'form-control',
            'type': 'datetime-local'
        }),
        label='From Date'
    )
    
    date_to = forms.DateTimeField(
        required=False,
        widget=forms.DateTimeInput(attrs={
            'class': 'form-control',
            'type': 'datetime-local'
        }),
        label='To Date'
    )
    
    object_type = forms.CharField(
        required=False,
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., person, car, dog'
        }),
        label='Object Type'
    )
    
    min_confidence = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'placeholder': '0.5'
        }),
        label='Minimum Confidence'
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'get'
        self.helper.layout = Layout(
            Row(
                Column('date_from', css_class='col-md-6'),
                Column('date_to', css_class='col-md-6'),
            ),
            Row(
                Column('object_type', css_class='col-md-6'),
                Column('min_confidence', css_class='col-md-6'),
            ),
            Submit('submit', 'Search Detections', css_class='btn btn-info')
        )