import React, { useState, useEffect } from 'react';
import { 
  Play, 
  BarChart3, 
  Users, 
  Target, 
  Clock, 
  Trophy,
  AlertCircle,
  CheckCircle,
  Loader2,
  Upload
} from 'lucide-react';
import VideoPlayer from './VideoPlayer';
import StatisticsDashboard from './StatisticsDashboard';

interface AnalysisResultsProps {
  jobId: string;
  onBack: () => void;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ jobId, onBack }) => {
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [events, setEvents] = useState<any[]>([]);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'video' | 'stats'>('video');

  useEffect(() => {
    fetchAnalysisResults();
  }, [jobId]);

  const fetchAnalysisResults = async () => {
    try {
      setLoading(true);
      
      // Fetch analysis results
      const resultsResponse = await fetch(`http://localhost:8001/analysis/${jobId}`);
      if (!resultsResponse.ok) {
        throw new Error('Failed to fetch analysis results');
      }
      const results = await resultsResponse.json();
      setAnalysisData(results.results);

      // Fetch events
      const eventsResponse = await fetch(`http://localhost:8001/events/${jobId}`);
      if (eventsResponse.ok) {
        const eventsData = await eventsResponse.json();
        setEvents(eventsData.all_events || []);
      }

      // Set video URL
      setVideoUrl(`http://localhost:8001/video/${jobId}`);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Loading Analysis Results</h2>
          <p className="text-gray-600">Processing your video analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Error Loading Results</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={onBack}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (!analysisData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-yellow-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No Analysis Data</h2>
          <p className="text-gray-600">Analysis results not found.</p>
        </div>
      </div>
    );
  }

  const videoInfo = analysisData.video_info || {};
  const detection = analysisData.detection || {};
  const statistics = analysisData.statistics || {};

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <button
                onClick={onBack}
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                ← Back
              </button>
              <h1 className="text-xl font-semibold text-gray-900">
                Analysis Results
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span className="text-sm text-gray-600">Analysis Complete</span>
              </div>
              <button
                onClick={onBack}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
              >
                <Upload className="w-4 h-4" />
                <span>Upload New Video</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Video Info Summary */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <Clock className="w-8 h-8 text-blue-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {videoInfo.duration_minutes?.toFixed(1) || 0}m
              </div>
              <div className="text-sm text-gray-600">Duration</div>
            </div>
            <div className="text-center">
              <Users className="w-8 h-8 text-green-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {detection.total_players || 0}
              </div>
              <div className="text-sm text-gray-600">Players Detected</div>
            </div>
            <div className="text-center">
              <Target className="w-8 h-8 text-purple-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {events.length}
              </div>
              <div className="text-sm text-gray-600">Events Detected</div>
            </div>
            <div className="text-center">
              <Trophy className="w-8 h-8 text-yellow-600 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900">
                {events.filter(e => e.type === 'goal').length}
              </div>
              <div className="text-sm text-gray-600">Goals</div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-sm mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6">
              <button
                onClick={() => setActiveTab('video')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'video'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Play className="w-4 h-4 inline mr-2" />
                Video Analysis
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'stats'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <BarChart3 className="w-4 h-4 inline mr-2" />
                Statistics
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'video' ? (
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Annotated Video with Real-time Detection
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Watch your video with AI-powered bounding boxes showing player detection, 
                    team classification, and goal notifications.
                  </p>
                </div>
                
                {videoUrl ? (
                  <div className="space-y-4">
                    <VideoPlayer
                      videoUrl={videoUrl}
                      events={events}
                      className="w-full h-96"
                    />
                    
                    {/* Video Info */}
                    <div className="bg-blue-50 rounded-lg p-4">
                      <h4 className="font-semibold text-blue-900 mb-2">🎯 AI Detection Accuracy</h4>
                      <p className="text-blue-800 text-sm">
                        This video shows real-time AI detection with color-coded bounding boxes:
                      </p>
                      <ul className="text-blue-700 text-sm mt-2 space-y-1">
                        <li>• <span className="font-medium">Green boxes</span> = Team A players</li>
                        <li>• <span className="font-medium">Blue boxes</span> = Team B players</li>
                        <li>• <span className="font-medium">Yellow boxes</span> = Referees</li>
                        <li>• <span className="font-medium">Cyan boxes</span> = Ball tracking</li>
                        <li>• <span className="font-medium">Numbers</span> = Unique player IDs for tracking</li>
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="w-full h-96 bg-gray-200 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                      <p className="text-gray-600">Annotated video not available</p>
                      <p className="text-gray-500 text-sm mt-1">The analysis may still be processing...</p>
                    </div>
                  </div>
                )}

                {/* Detection Legend */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 mb-3">Detection Legend</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-green-500 rounded"></div>
                      <span className="text-sm text-gray-700">Team A Players</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-blue-500 rounded"></div>
                      <span className="text-sm text-gray-700">Team B Players</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                      <span className="text-sm text-gray-700">Referees</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-4 h-4 bg-cyan-500 rounded"></div>
                      <span className="text-sm text-gray-700">Ball</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Detailed Statistics
                </h3>
                <StatisticsDashboard analysisResults={analysisData} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
