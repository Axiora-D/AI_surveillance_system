# AI Surveillance System

**Repository:** [github.com/AHub2-dot/ai-surveillance-system](https://github.com/AHub2-dot/ai-surveillance-system)

A Django-based AI surveillance system with live video streaming, object detection (YOLO), recording, and two-factor authentication.

## Features

- **Live surveillance** – Real-time camera feed with object detection
- **Recordings** – Save and review video clips
- **Analytics** – Dashboard and detection logs
- **Security** – Login, signup, and optional 2FA (TOTP)
- **Upload** – Process uploaded videos with AI inference

## Tech Stack

- **Backend:** Django 6.x
- **AI:** Ultralytics YOLO, OpenCV
- **Frontend:** HTML templates, Crispy Forms
- **Auth:** Django auth + pyotp/qrcode for 2FA

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AHub2-dot/ai-surveillance-system.git
   cd ai-surveillance-system
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Start the server**
   ```bash
   python manage.py runserver
   ```

Open http://127.0.0.1:8000 in your browser.

## Project Structure

```
ai_surveillance_system/
├── ai_surveillance_system/   # Django project settings
├── MVP/                      # Main app (views, models, templates, utils)
├── media/                    # Uploaded/recorded files (not in git)
├── manage.py
└── requirements.txt
```

## License

MIT (or your preferred license)
