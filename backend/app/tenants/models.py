import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()


class Tenant(models.Model):
    """
    Multi-tenant organization model.
    """
    PLAN_CHOICES = [
        ('free', 'Free'),
        ('starter', 'Starter'),
        ('professional', 'Professional'),
        ('enterprise', 'Enterprise'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('suspended', 'Suspended'),
        ('cancelled', 'Cancelled'),
        ('trial', 'Trial'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True)
    plan = models.CharField(max_length=50, choices=PLAN_CHOICES, default='free')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='trial')
    
    # Billing and limits
    monthly_video_limit = models.IntegerField(default=10)
    monthly_processing_minutes = models.IntegerField(default=60)
    storage_limit_gb = models.IntegerField(default=5)
    api_calls_per_month = models.IntegerField(default=1000)
    
    # Metadata
    description = models.TextField(blank=True, null=True)
    website = models.URLField(blank=True, null=True)
    logo = models.ImageField(upload_to='tenant_logos/', blank=True, null=True)
    settings = models.JSONField(default=dict)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    trial_ends_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        db_table = 'tenants'
        verbose_name = 'Tenant'
        verbose_name_plural = 'Tenants'
        indexes = [
            models.Index(fields=['slug']),
            models.Index(fields=['status']),
            models.Index(fields=['plan']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.slug})"
    
    def is_trial_expired(self):
        if not self.trial_ends_at:
            return False
        return timezone.now() > self.trial_ends_at
    
    def get_usage_stats(self):
        """Get current usage statistics for the tenant."""
        from django.db.models import Sum, Count
        
        # This would be implemented with actual usage tracking
        return {
            'videos_uploaded': 0,
            'processing_minutes_used': 0,
            'storage_used_gb': 0,
            'api_calls_made': 0,
        }


class TenantUser(models.Model):
    """
    Many-to-many relationship between users and tenants with roles.
    """
    ROLE_CHOICES = [
        ('owner', 'Owner'),
        ('admin', 'Admin'),
        ('manager', 'Manager'),
        ('analyst', 'Analyst'),
        ('viewer', 'Viewer'),
    ]
    
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='tenant_users')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tenant_users')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='viewer')
    is_active = models.BooleanField(default=True)
    joined_at = models.DateTimeField(auto_now_add=True)
    invited_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='invited_users')
    
    class Meta:
        db_table = 'tenant_users'
        verbose_name = 'Tenant User'
        verbose_name_plural = 'Tenant Users'
        unique_together = ['tenant', 'user']
        indexes = [
            models.Index(fields=['tenant', 'user']),
            models.Index(fields=['role']),
        ]
    
    def __str__(self):
        return f"{self.user.email} - {self.tenant.name} ({self.role})"


class TenantInvitation(models.Model):
    """
    Invitations for users to join tenants.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('declined', 'Declined'),
        ('expired', 'Expired'),
    ]
    
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='invitations')
    email = models.EmailField()
    role = models.CharField(max_length=20, choices=TenantUser.ROLE_CHOICES, default='viewer')
    invited_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_invitations')
    token = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    message = models.TextField(blank=True, null=True)
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    responded_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        db_table = 'tenant_invitations'
        verbose_name = 'Tenant Invitation'
        verbose_name_plural = 'Tenant Invitations'
        indexes = [
            models.Index(fields=['token']),
            models.Index(fields=['email']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.email} -> {self.tenant.name} ({self.status})"
    
    def is_expired(self):
        return timezone.now() > self.expires_at


class TenantSettings(models.Model):
    """
    Tenant-specific configuration settings.
    """
    tenant = models.OneToOneField(Tenant, on_delete=models.CASCADE, related_name='settings_config')
    
    # Video processing settings
    default_fps = models.IntegerField(default=25)
    max_video_duration_minutes = models.IntegerField(default=60)
    auto_process_videos = models.BooleanField(default=True)
    
    # Model settings
    default_detection_model = models.CharField(max_length=100, default='yolov8n')
    default_pose_model = models.CharField(max_length=100, default='movenet')
    default_action_model = models.CharField(max_length=100, default='slowfast')
    
    # Notification settings
    email_notifications = models.BooleanField(default=True)
    slack_webhook_url = models.URLField(blank=True, null=True)
    webhook_url = models.URLField(blank=True, null=True)
    
    # Data retention
    data_retention_days = models.IntegerField(default=365)
    auto_delete_processed_videos = models.BooleanField(default=False)
    
    # API settings
    api_rate_limit_per_minute = models.IntegerField(default=60)
    webhook_secret = models.CharField(max_length=64, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'tenant_settings'
        verbose_name = 'Tenant Settings'
        verbose_name_plural = 'Tenant Settings'
    
    def __str__(self):
        return f"Settings for {self.tenant.name}"


class TenantUsage(models.Model):
    """
    Track tenant usage for billing and limits.
    """
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name='usage_records')
    date = models.DateField()
    
    # Usage metrics
    videos_uploaded = models.IntegerField(default=0)
    processing_minutes = models.FloatField(default=0.0)
    storage_used_mb = models.BigIntegerField(default=0)
    api_calls = models.IntegerField(default=0)
    inference_requests = models.IntegerField(default=0)
    
    # Cost tracking
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'tenant_usage'
        verbose_name = 'Tenant Usage'
        verbose_name_plural = 'Tenant Usage Records'
        unique_together = ['tenant', 'date']
        indexes = [
            models.Index(fields=['tenant', 'date']),
            models.Index(fields=['date']),
        ]
    
    def __str__(self):
        return f"{self.tenant.name} - {self.date}"
