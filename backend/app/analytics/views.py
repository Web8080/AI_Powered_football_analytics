"""
Analytics views for Godseye AI sports analytics platform.
Provides comprehensive statistics and insights for football match analysis.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.db.models import Q, Count, Avg, Sum, Max, Min
from django.core.paginator import Paginator
from django.conf import settings

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import AnalysisJob, Video, MatchStatistics, PlayerStatistics, TeamStatistics
from .serializers import (
    MatchStatisticsSerializer, 
    PlayerStatisticsSerializer, 
    TeamStatisticsSerializer,
    AnalyticsSummarySerializer
)
from .utils import StatisticsCalculator, StatisticsVisualizer

logger = logging.getLogger(__name__)


class AnalyticsAPIView(APIView):
    """Base analytics API view with common functionality."""
    
    permission_classes = [IsAuthenticated]
    
    def get_tenant_from_request(self, request):
        """Get tenant from request context."""
        return getattr(request, 'tenant', None)
    
    def get_analysis_job(self, job_id: str, tenant=None):
        """Get analysis job with tenant filtering."""
        try:
            job = AnalysisJob.objects.get(id=job_id)
            if tenant and job.tenant != tenant:
                raise AnalysisJob.DoesNotExist
            return job
        except AnalysisJob.DoesNotExist:
            return None


@method_decorator(csrf_exempt, name='dispatch')
class TeamStatisticsView(AnalyticsAPIView):
    """API endpoint for team statistics."""
    
    def get(self, request, job_id):
        """Get team statistics for a specific analysis job."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Get team statistics from job results
            results = job.results
            if not results:
                return Response(
                    {'error': 'No results available'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Calculate team statistics
            calculator = StatisticsCalculator()
            team_stats = calculator.calculate_team_statistics(
                results.get('tracking', {}).get('player_tracks', {}),
                results.get('tracking', {}).get('ball_trajectory', []),
                results.get('events', {}).get('events', []),
                results.get('processing_time', 90 * 60)
            )
            
            # Serialize and return
            serializer = TeamStatisticsSerializer(team_stats.values(), many=True)
            return Response({
                'job_id': job_id,
                'team_statistics': serializer.data,
                'generated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error calculating team statistics: {e}")
            return Response(
                {'error': 'Failed to calculate team statistics'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class PlayerStatisticsView(AnalyticsAPIView):
    """API endpoint for player statistics."""
    
    def get(self, request, job_id, player_id=None):
        """Get player statistics for a specific analysis job."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Get player statistics from job results
            results = job.results
            if not results:
                return Response(
                    {'error': 'No results available'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Calculate player statistics
            calculator = StatisticsCalculator()
            player_tracks = results.get('tracking', {}).get('player_tracks', {})
            
            if player_id:
                # Get specific player statistics
                if player_id not in player_tracks:
                    return Response(
                        {'error': 'Player not found'}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                player_stat = calculator.calculate_player_statistics(
                    player_tracks[player_id],
                    results.get('tracking', {}).get('ball_trajectory', []),
                    results.get('events', {}).get('events', []),
                    results.get('processing_time', 90 * 60)
                )
                
                serializer = PlayerStatisticsSerializer(player_stat)
                return Response({
                    'job_id': job_id,
                    'player_id': player_id,
                    'player_statistics': serializer.data,
                    'generated_at': datetime.now().isoformat()
                })
            else:
                # Get all player statistics
                player_stats = {}
                for pid, tracks in player_tracks.items():
                    if tracks:
                        player_stat = calculator.calculate_player_statistics(
                            tracks,
                            results.get('tracking', {}).get('ball_trajectory', []),
                            results.get('events', {}).get('events', []),
                            results.get('processing_time', 90 * 60)
                        )
                        player_stats[pid] = player_stat
                
                serializer = PlayerStatisticsSerializer(player_stats.values(), many=True)
                return Response({
                    'job_id': job_id,
                    'player_statistics': serializer.data,
                    'generated_at': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error calculating player statistics: {e}")
            return Response(
                {'error': 'Failed to calculate player statistics'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class HeatmapView(AnalyticsAPIView):
    """API endpoint for heatmap data."""
    
    def get(self, request, job_id):
        """Get heatmap data for a specific analysis job."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Get heatmap type from query parameters
            heatmap_type = request.GET.get('type', 'team')  # team, player, ball
            team_id = request.GET.get('team_id')
            player_id = request.GET.get('player_id')
            
            results = job.results
            if not results:
                return Response(
                    {'error': 'No results available'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Generate heatmap data
            calculator = StatisticsCalculator()
            heatmap_data = {}
            
            if heatmap_type == 'team' and team_id:
                # Team heatmap
                team_tracks = results.get('tracking', {}).get('player_tracks', {})
                if team_id in team_tracks:
                    positions = [track['center'] for track in team_tracks[team_id]]
                    heatmap_data = calculator._generate_heatmap(positions)
            elif heatmap_type == 'player' and player_id:
                # Player heatmap
                player_tracks = results.get('tracking', {}).get('player_tracks', {})
                if player_id in player_tracks:
                    positions = [track['center'] for track in player_tracks[player_id]]
                    heatmap_data = calculator._generate_heatmap(positions)
            elif heatmap_type == 'ball':
                # Ball heatmap
                ball_positions = results.get('tracking', {}).get('ball_trajectory', [])
                positions = [pos['position'] for pos in ball_positions]
                heatmap_data = calculator._generate_heatmap(positions)
            
            return Response({
                'job_id': job_id,
                'heatmap_type': heatmap_type,
                'heatmap_data': heatmap_data.tolist() if hasattr(heatmap_data, 'tolist') else heatmap_data,
                'generated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return Response(
                {'error': 'Failed to generate heatmap'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class FormationAnalysisView(AnalyticsAPIView):
    """API endpoint for formation analysis."""
    
    def get(self, request, job_id):
        """Get formation analysis for a specific analysis job."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            results = job.results
            if not results:
                return Response(
                    {'error': 'No results available'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Analyze formations
            calculator = StatisticsCalculator()
            team_tracks = results.get('tracking', {}).get('player_tracks', {})
            
            formation_analysis = {}
            for team_id, tracks in team_tracks.items():
                if tracks:
                    formation = calculator._analyze_formation(tracks)
                    formation_metrics = calculator._calculate_formation_metrics(tracks)
                    
                    formation_analysis[team_id] = {
                        'formation': formation,
                        'width': formation_metrics['width'],
                        'height': formation_metrics['height'],
                        'changes': formation_metrics['changes'],
                        'player_positions': [track['center'] for track in tracks]
                    }
            
            return Response({
                'job_id': job_id,
                'formation_analysis': formation_analysis,
                'generated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error analyzing formations: {e}")
            return Response(
                {'error': 'Failed to analyze formations'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class EventTimelineView(AnalyticsAPIView):
    """API endpoint for event timeline."""
    
    def get(self, request, job_id):
        """Get event timeline for a specific analysis job."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            results = job.results
            if not results:
                return Response(
                    {'error': 'No results available'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get events and organize by time
            events = results.get('events', {}).get('events', [])
            
            # Group events by time periods
            timeline = {}
            for event in events:
                timestamp = event.get('timestamp', 0)
                time_period = int(timestamp // 900) * 15  # 15-minute periods
                
                if time_period not in timeline:
                    timeline[time_period] = {
                        'period': f"{time_period}-{time_period + 15}",
                        'events': [],
                        'goals': 0,
                        'cards': 0,
                        'substitutions': 0
                    }
                
                timeline[time_period]['events'].append(event)
                
                event_name = event.get('event_name', '')
                if event_name == 'goal':
                    timeline[time_period]['goals'] += 1
                elif event_name in ['yellow_card', 'red_card']:
                    timeline[time_period]['cards'] += 1
                elif event_name == 'substitution':
                    timeline[time_period]['substitutions'] += 1
            
            return Response({
                'job_id': job_id,
                'event_timeline': list(timeline.values()),
                'total_events': len(events),
                'generated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating event timeline: {e}")
            return Response(
                {'error': 'Failed to generate event timeline'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class ExportResultsView(AnalyticsAPIView):
    """API endpoint for exporting analysis results."""
    
    def get(self, request, job_id):
        """Export analysis results in various formats."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            export_format = request.GET.get('format', 'json')
            
            if export_format == 'json':
                # Return JSON data
                return Response({
                    'job_id': job_id,
                    'results': job.results,
                    'exported_at': datetime.now().isoformat()
                })
            
            elif export_format == 'csv':
                # Generate CSV data
                from .utils import export_statistics_to_csv
                import tempfile
                import os
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    temp_path = f.name
                
                # Export to CSV
                export_statistics_to_csv(job.results, temp_path)
                
                # Read and return CSV content
                with open(temp_path, 'r') as f:
                    csv_content = f.read()
                
                # Clean up
                os.unlink(temp_path)
                
                response = HttpResponse(csv_content, content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="analysis_{job_id}.csv"'
                return response
            
            elif export_format == 'pdf':
                # Generate PDF report
                from .utils import generate_pdf_report
                
                pdf_content = generate_pdf_report(job.results, job_id)
                
                response = HttpResponse(pdf_content, content_type='application/pdf')
                response['Content-Disposition'] = f'attachment; filename="analysis_{job_id}.pdf"'
                return response
            
            else:
                return Response(
                    {'error': 'Unsupported export format'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return Response(
                {'error': 'Failed to export results'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name='dispatch')
class AnalyticsSummaryView(AnalyticsAPIView):
    """API endpoint for analytics summary."""
    
    def get(self, request, job_id):
        """Get comprehensive analytics summary."""
        tenant = self.get_tenant_from_request(request)
        job = self.get_analysis_job(job_id, tenant)
        
        if not job:
            return Response(
                {'error': 'Analysis job not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        if job.status != 'completed':
            return Response(
                {'error': 'Analysis not completed yet'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            results = job.results
            if not results:
                return Response(
                    {'error': 'No results available'}, 
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Generate comprehensive summary
            from .utils import generate_statistics_report
            
            summary = generate_statistics_report(results)
            
            return Response({
                'job_id': job_id,
                'summary': summary,
                'generated_at': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating analytics summary: {e}")
            return Response(
                {'error': 'Failed to generate analytics summary'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analytics_health_check(request):
    """Health check endpoint for analytics service."""
    return Response({
        'status': 'healthy',
        'service': 'analytics',
        'timestamp': datetime.now().isoformat()
    })


# URL patterns would be defined in urls.py
# Example:
# urlpatterns = [
#     path('analytics/teams/<str:job_id>/', TeamStatisticsView.as_view(), name='team_statistics'),
#     path('analytics/players/<str:job_id>/', PlayerStatisticsView.as_view(), name='player_statistics'),
#     path('analytics/players/<str:job_id>/<str:player_id>/', PlayerStatisticsView.as_view(), name='player_statistics_detail'),
#     path('analytics/heatmaps/<str:job_id>/', HeatmapView.as_view(), name='heatmaps'),
#     path('analytics/formations/<str:job_id>/', FormationAnalysisView.as_view(), name='formation_analysis'),
#     path('analytics/events/<str:job_id>/', EventTimelineView.as_view(), name='event_timeline'),
#     path('analytics/export/<str:job_id>/', ExportResultsView.as_view(), name='export_results'),
#     path('analytics/summary/<str:job_id>/', AnalyticsSummaryView.as_view(), name='analytics_summary'),
#     path('analytics/health/', analytics_health_check, name='analytics_health'),
# ]
