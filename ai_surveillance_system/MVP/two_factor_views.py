from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
from .models import UserProfile
from MVP.utils.two_factor import TwoFactorAuth


def setup_2fa(request):
    """Setup 2FA for the current user"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    try:
        profile, created = UserProfile.objects.get_or_create(user=request.user)
    except:
        profile = UserProfile.objects.create(user=request.user)
    
    # Generate secret if not exists
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
    
    return render(request, 'two_factor_setup.html', context)

@require_http_methods(["POST"])
def verify_2fa_setup(request):
    """Verify 2FA setup with token"""
    print(f"2FA verification request from user: {request.user}")
    
    if not request.user.is_authenticated:
        print("User not authenticated")
        return JsonResponse({'error': 'Not authenticated'}, status=401)
    
    token = request.POST.get('token')
    print(f"Token received: {token}")
    
    if not token:
        print("No token provided")
        return JsonResponse({'error': 'Token required'}, status=400)
    
    # Handle test request to get current token
    if token == 'test':
        try:
            profile = UserProfile.objects.get(user=request.user)
            auth = TwoFactorAuth()
            current_token = auth.get_current_token(profile.two_factor_secret)
            return JsonResponse({
                'test': True,
                'current_token': current_token,
                'secret': profile.two_factor_secret,
                'message': f'Current valid token: {current_token}'
            })
        except Exception as e:
            return JsonResponse({'error': f'Test error: {str(e)}'}, status=500)
    
    try:
        profile = UserProfile.objects.get(user=request.user)
        print(f"Profile found: {profile}")
        print(f"Secret: {profile.two_factor_secret}")
        
        auth = TwoFactorAuth()
        
        # For testing, let's also get the current valid token
        current_token = auth.get_current_token(profile.two_factor_secret)
        print(f"Current valid token: {current_token}")
        print(f"Token being verified: {token}")
        print(f"Token type: {type(token)}")
        print(f"Current token type: {type(current_token)}")
        
        # Test if the current token matches what we expect
        print(f"Tokens match: {str(token) == str(current_token)}")
        print(f"Token length: {len(str(token))}")
        print(f"Current token length: {len(str(current_token))}")
        
        # Try verification with different token formats
        verification_result = auth.verify_token(profile.two_factor_secret, token)
        print(f"Verification result: {verification_result}")
        
        # Also try with string conversion
        if not verification_result:
            verification_result = auth.verify_token(profile.two_factor_secret, str(token))
            print(f"Verification result (string): {verification_result}")
        
        # Also try with zero-padded token
        if not verification_result and len(token) < 6:
            padded_token = token.zfill(6)
            verification_result = auth.verify_token(profile.two_factor_secret, padded_token)
            print(f"Verification result (padded): {verification_result}")
        
        if verification_result:
            print("Token verification successful")
            profile.two_factor_enabled = True
            # Generate backup codes
            backup_codes = profile.generate_backup_codes()
            profile.save()
            
            return JsonResponse({
                'success': True,
                'backup_codes': backup_codes,
                'message': '2FA setup successful!'
            })
        else:
            print("Token verification failed")
            return JsonResponse({
                'error': 'Invalid token', 
                'current_token': current_token,
                'provided_token': token
            }, status=400)
            
    except UserProfile.DoesNotExist:
        print("UserProfile not found")
        return JsonResponse({'error': 'Profile not found'}, status=404)
    except Exception as e:
        print(f"Exception in verify_2fa_setup: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def verify_2fa_login(request):
    """Verify 2FA token during login"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    try:
        profile = UserProfile.objects.get(user=request.user)
        if not profile.two_factor_enabled:
            return redirect('options')
    except UserProfile.DoesNotExist:
        return redirect('options')
    
    if request.method == 'POST':
        token = request.POST.get('token')
        backup_code = request.POST.get('backup_code')
        
        auth = TwoFactorAuth()
        
        # Try regular token first
        if token and auth.verify_token(profile.two_factor_secret, token):
            return redirect('options')
        
        # Try backup code
        elif backup_code:
            backup_codes = profile.get_backup_codes()
            if backup_code in backup_codes:
                # Remove used backup code
                backup_codes.remove(backup_code)
                profile.set_backup_codes(backup_codes)
                profile.save()
                return redirect('options')
            else:
                messages.error(request, 'Invalid backup code')
        else:
            messages.error(request, 'Invalid token')
    
    return render(request, 'two_factor_verify.html')

@login_required
def disable_2fa(request):
    """Disable 2FA for the current user"""
    if request.method == 'POST':
        try:
            profile = UserProfile.objects.get(user=request.user)
            profile.two_factor_enabled = False
            profile.two_factor_secret = ''
            profile.backup_codes = ''
            profile.save()
            messages.success(request, '2FA has been disabled successfully')
        except UserProfile.DoesNotExist:
            messages.error(request, 'Profile not found')
    
    return redirect('settings')

@login_required
def regenerate_backup_codes(request):
    """Regenerate backup codes for 2FA"""
    try:
        profile = UserProfile.objects.get(user=request.user)
        if profile.two_factor_enabled:
            backup_codes = profile.generate_backup_codes()
            profile.save()
            return JsonResponse({
                'success': True,
                'backup_codes': backup_codes,
                'message': 'Backup codes regenerated successfully'
            })
        else:
            return JsonResponse({'error': '2FA not enabled'}, status=400)
    except UserProfile.DoesNotExist:
        return JsonResponse({'error': 'Profile not found'}, status=404)
