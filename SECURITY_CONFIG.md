# 🔐 Security Configuration Guide

## ⚠️ CRITICAL: Never commit passwords or API keys to Git!

### Environment Variables Setup

Create a `.env` file in your project root (this file should be in `.gitignore`):

```bash
# SoccerNet Dataset Access
SOCCERNET_PASSWORD=your_actual_password_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/godseye_ai
MONGODB_URL=mongodb://localhost:27017/godseye_ai

# AWS Configuration (for production)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=your_s3_bucket_name

# API Keys
OPENAI_API_KEY=your_openai_api_key
STRIPE_SECRET_KEY=your_stripe_secret_key

# Security
SECRET_KEY=your_django_secret_key
JWT_SECRET_KEY=your_jwt_secret_key
```

### Usage in Scripts

The training scripts now use environment variables:

```python
# Secure way to get password
password = os.getenv("SOCCERNET_PASSWORD", input("Enter SoccerNet password: "))
```

### Setting Environment Variables

#### Local Development:
```bash
export SOCCERNET_PASSWORD="your_password_here"
```

#### Google Colab:
```python
import os
os.environ["SOCCERNET_PASSWORD"] = "your_password_here"
```

#### Production:
Set environment variables in your deployment platform (AWS, Heroku, etc.)

### Security Best Practices

1. **Never commit `.env` files**
2. **Use environment variables for all secrets**
3. **Rotate passwords regularly**
4. **Use different passwords for different environments**
5. **Monitor access logs**
6. **Use strong, unique passwords**

### Git History Cleanup

If you've already committed sensitive data:

```bash
# Remove from git history (DANGEROUS - use with caution)
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch .env' \
--prune-empty --tag-name-filter cat -- --all

# Force push (will rewrite history)
git push origin --force --all
```

**Warning**: This rewrites git history and can cause issues for collaborators.
