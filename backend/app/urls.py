"""
URL configuration for Godseye project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API Documentation
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # API v1
    path('api/v1/auth/', include('app.accounts.urls')),
    path('api/v1/tenants/', include('app.tenants.urls')),
    path('api/v1/videos/', include('app.videos.urls')),
    path('api/v1/jobs/', include('app.jobs.urls')),
    path('api/v1/models/', include('app.models.urls')),
    path('api/v1/datasets/', include('app.datasets.urls')),
    path('api/v1/analytics/', include('app.analytics.urls')),
    path('api/v1/billing/', include('app.billing.urls')),
    
    # Health check
    path('health/', include('django_prometheus.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
