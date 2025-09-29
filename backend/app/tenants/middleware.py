"""
Tenant middleware for multi-tenancy support.
"""
from django.http import Http404
from django.shortcuts import get_object_or_404
from .models import Tenant, TenantUser


class TenantMiddleware:
    """
    Middleware to handle tenant context for multi-tenancy.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Set tenant context based on subdomain, header, or user
        tenant = self.get_tenant_from_request(request)
        request.tenant = tenant
        
        # Set user's role in current tenant
        if hasattr(request, 'user') and request.user.is_authenticated and tenant:
            try:
                tenant_user = TenantUser.objects.get(
                    tenant=tenant,
                    user=request.user,
                    is_active=True
                )
                request.tenant_user = tenant_user
                request.tenant_role = tenant_user.role
            except TenantUser.DoesNotExist:
                request.tenant_user = None
                request.tenant_role = None
        else:
            request.tenant_user = None
            request.tenant_role = None
        
        response = self.get_response(request)
        return response
    
    def get_tenant_from_request(self, request):
        """
        Determine tenant from request.
        Priority: subdomain > header > user's default tenant
        """
        # Method 1: Subdomain (e.g., tenant1.godseye.com)
        host = request.get_host()
        if '.' in host:
            subdomain = host.split('.')[0]
            if subdomain not in ['www', 'api', 'admin']:
                try:
                    return Tenant.objects.get(slug=subdomain, status='active')
                except Tenant.DoesNotExist:
                    pass
        
        # Method 2: X-Tenant-ID header
        tenant_id = request.META.get('HTTP_X_TENANT_ID')
        if tenant_id:
            try:
                return Tenant.objects.get(id=tenant_id, status='active')
            except (Tenant.DoesNotExist, ValueError):
                pass
        
        # Method 3: User's default tenant (if authenticated)
        if hasattr(request, 'user') and request.user.is_authenticated:
            try:
                tenant_user = TenantUser.objects.filter(
                    user=request.user,
                    is_active=True
                ).select_related('tenant').first()
                if tenant_user and tenant_user.tenant.status == 'active':
                    return tenant_user.tenant
            except TenantUser.DoesNotExist:
                pass
        
        return None


def require_tenant(view_func):
    """
    Decorator to require tenant context for a view.
    """
    def wrapper(request, *args, **kwargs):
        if not hasattr(request, 'tenant') or not request.tenant:
            raise Http404("Tenant not found")
        return view_func(request, *args, **kwargs)
    return wrapper


def require_tenant_role(*allowed_roles):
    """
    Decorator to require specific tenant role.
    """
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            if not hasattr(request, 'tenant_user') or not request.tenant_user:
                raise Http404("Access denied")
            
            if request.tenant_role not in allowed_roles:
                raise Http404("Insufficient permissions")
            
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator


def get_tenant_queryset(model_class, request):
    """
    Helper to filter queryset by tenant.
    """
    if not hasattr(request, 'tenant') or not request.tenant:
        return model_class.objects.none()
    
    # If model has tenant field, filter by it
    if hasattr(model_class, 'tenant'):
        return model_class.objects.filter(tenant=request.tenant)
    
    # If model has tenant_id field, filter by it
    if hasattr(model_class, 'tenant_id'):
        return model_class.objects.filter(tenant_id=request.tenant.id)
    
    # If no tenant field, return all (for global models)
    return model_class.objects.all()
