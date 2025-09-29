import os
from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')

app = Celery('godseye')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

# Celery Beat schedule
app.conf.beat_schedule = {
    'cleanup-expired-jobs': {
        'task': 'app.jobs.tasks.cleanup_expired_jobs',
        'schedule': 3600.0,  # Run every hour
    },
    'process-pending-videos': {
        'task': 'app.videos.tasks.process_pending_videos',
        'schedule': 300.0,  # Run every 5 minutes
    },
    'update-model-metrics': {
        'task': 'app.models.tasks.update_model_metrics',
        'schedule': 1800.0,  # Run every 30 minutes
    },
    'sync-dataset-status': {
        'task': 'app.datasets.tasks.sync_dataset_status',
        'schedule': 600.0,  # Run every 10 minutes
    },
}

app.conf.timezone = 'UTC'

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
