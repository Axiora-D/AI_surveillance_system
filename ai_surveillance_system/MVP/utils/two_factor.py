import pyotp
import qrcode
from io import BytesIO
import base64

class TwoFactorAuth:
    def __init__(self):
        pass
    
    def generate_secret(self):
        """Generate a new secret key for the user"""
        return pyotp.random_base32()
    
    def get_qr_code(self, user_email, secret, app_name="YourApp"):
        """Generate QR code for authenticator app setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=app_name
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 for web display
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def verify_token(self, secret, token):
        """Verify the 6-digit token from authenticator app"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 30s window
    
    def get_current_token(self, secret):
        """Get current token (for testing)"""
        totp = pyotp.TOTP(secret)
        return totp.now()

# Example usage
if __name__ == "__main__":
    auth = TwoFactorAuth()
    
    # Step 1: Generate secret for new user
    secret = auth.generate_secret()
    print(f"Secret key: {secret}")
    
    # Step 2: Generate QR code for user to scan
    qr_code = auth.get_qr_code("user@example.com", secret)
    print("QR Code generated (base64 data URI)")
    
    # Step 3: Verify token
    current_token = auth.get_current_token(secret)
    print(f"Current token: {current_token}")
    
    # Verify the token
    is_valid = auth.verify_token(secret, current_token)
    print(f"Token valid: {is_valid}")