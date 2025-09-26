# JWT Authentication for SmolRouter

SmolRouter supports JWT authentication for securing the admin dashboard when accessed externally.

## Quick Setup

```bash
# Generate a strong JWT secret (32+ characters)
export JWT_SECRET="your-super-secure-32-character-secret"

# Enable always-on authentication for external access
export WEBUI_SECURITY="ALWAYS_AUTH"

# Start SmolRouter
python -m smolrouter.cli
```

## Security Policies

| Policy | Description | When to Use |
|--------|-------------|-------------|
| `NONE` | No authentication required | Local development only |
| `AUTH_WHEN_PROXIED` | Auth required when reverse proxy detected | **Default** - good for most setups |
| `ALWAYS_AUTH` | Always require JWT | External access/production |

## JWT Secret Requirements

Your JWT_SECRET must meet these security requirements:

- ✅ **Minimum 32 characters long**  
- ✅ **At least 8 unique characters**
- ✅ **Not a common/weak password**
- ✅ **Cryptographically secure random**

### Generate a Secure Secret

```bash
# Option 1: OpenSSL
openssl rand -base64 48

# Option 2: Python
python -c "import secrets; print(secrets.token_urlsafe(48))"

# Option 3: Manual (ensure 32+ chars with good entropy)
export JWT_SECRET="2a8f4b1c-6e9d-4a7f-9b3e-1c5a8b2f7e4d-secure"
```

## Creating JWT Tokens

SmolRouter includes JWT token creation utilities:

```python
from smolrouter.auth import get_jwt_auth

# Get auth instance (requires JWT_SECRET to be set)
auth = get_jwt_auth()

# Create a token for a user  
token = auth.create_token({
    "sub": "admin_user",        # Subject (username)
    "name": "Admin User",       # Display name
    "role": "admin"             # Role/permissions
}, expires_in_hours=24)         # 24 hour expiry

print(f"Authorization: Bearer {token}")
```

## Using JWT Tokens

Include the JWT token in the `Authorization` header:

```bash
# Dashboard access
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://your-server:1234/

# API endpoints  
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://your-server:1234/api/stats
```

## Configuration Examples

### Development (No Auth)
```bash
export WEBUI_SECURITY="NONE"
```

### Production Behind Reverse Proxy
```bash
export WEBUI_SECURITY="AUTH_WHEN_PROXIED"  # Default
export JWT_SECRET="your-secure-secret-here"
```

### Production with Direct External Access
```bash
export WEBUI_SECURITY="ALWAYS_AUTH"
export JWT_SECRET="your-secure-secret-here"
```

## Security Features

- ✅ **Strong secret validation** - Rejects weak/common passwords
- ✅ **Automatic token expiry** - 24 hour default, configurable  
- ✅ **Case-insensitive proxy detection** - Prevents header bypass attacks
- ✅ **Detailed security logging** - Audit access attempts
- ✅ **Graceful fallbacks** - Secure defaults on configuration errors

## Troubleshooting

### "JWT_SECRET is too short"
```bash
# Your secret needs to be at least 32 characters
export JWT_SECRET="make-this-secret-at-least-32-characters-long"
```

### "JWT_SECRET appears to be a weak/default secret"
```bash
# Don't use common passwords - generate a random secret
export JWT_SECRET=$(openssl rand -base64 48)
```

### "WebUI access denied: jwt_verification_not_available"
```bash
# JWT_SECRET is not properly configured
export JWT_SECRET="your-secure-secret-here"
export WEBUI_SECURITY="ALWAYS_AUTH"
```

### Dashboard shows 403 errors
```bash
# Include JWT token in Authorization header
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:1234/
```

## Integration Notes

- JWT validation is **only required for WebUI/dashboard access**
- **Proxy endpoints** (`/v1/chat/completions`, etc.) are **not affected**
- Works with any reverse proxy (nginx, Traefik, Cloudflare, etc.)
- Compatible with standard JWT libraries and tools