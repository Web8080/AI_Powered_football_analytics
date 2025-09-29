import React, { useState } from 'react';
import { 
  Upload, 
  History, 
  Settings, 
  BarChart3,
  Video,
  Camera,
  Users,
  Target
} from 'lucide-react';
import VideoUpload from './VideoUpload';
import ResultsHistory from './ResultsHistory';
import AnalysisResults from './AnalysisResults';

type TabType = 'upload' | 'history' | 'settings';

const EnhancedMainDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [showAnalysisResults, setShowAnalysisResults] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string>('');

  const handleAnalysisComplete = (results: any) => {
    // Save results to localStorage for history
    const analysisResult = {
      id: `analysis_${Date.now()}`,
      filename: 'Current Analysis',
      uploadDate: new Date().toISOString(),
      duration: results.video_info?.duration_minutes || 0,
      status: 'completed',
      stats: {
        totalPlayers: results.detection?.total_players || 0,
        teamAPlayers: results.detection?.team_a_players || 0,
        teamBPlayers: results.detection?.team_b_players || 0,
        referees: results.detection?.referees || 0,
        balls: results.detection?.balls || 0,
        events: results.events?.length || 0,
        goals: results.events?.filter((e: any) => e.type === 'goal').length || 0
      }
    };

    // Save to localStorage
    const existingResults = JSON.parse(localStorage.getItem('godseye_analysis_results') || '[]');
    existingResults.unshift(analysisResult); // Add to beginning
    localStorage.setItem('godseye_analysis_results', JSON.stringify(existingResults));
  };

  const handleBackToUpload = () => {
    setShowAnalysisResults(false);
    setCurrentJobId('');
  };

  if (showAnalysisResults) {
    return (
      <AnalysisResults
        jobId={currentJobId}
        onBack={handleBackToUpload}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-8">
              <div className="flex items-center space-x-2">
                <Target className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Godseye AI</h1>
              </div>
              
              {/* Navigation Tabs */}
              <nav className="flex space-x-8">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                    activeTab === 'upload'
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Upload className="w-4 h-4" />
                  <span>Upload & Analyze</span>
                </button>
                
                <button
                  onClick={() => setActiveTab('history')}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                    activeTab === 'history'
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <History className="w-4 h-4" />
                  <span>Analysis History</span>
                </button>
                
                <button
                  onClick={() => setActiveTab('settings')}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors ${
                    activeTab === 'settings'
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Settings className="w-4 h-4" />
                  <span>Settings</span>
                </button>
              </nav>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                <span className="font-medium">AI-Powered</span> Sports Analytics
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Upload & Analyze Football Video
              </h2>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                Upload your football video (up to 2GB) and get comprehensive AI-powered analysis 
                with player tracking, team classification, event detection, and real-time statistics.
              </p>
            </div>
            
            <VideoUpload
              onAnalysisComplete={handleAnalysisComplete}
              onUploadSuccess={(videoId) => {
                setCurrentJobId(videoId);
              }}
            />
          </div>
        )}

        {activeTab === 'history' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Analysis History
              </h2>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                View and manage your previous video analyses. Access detailed results, 
                download data, and compare different matches.
              </p>
            </div>
            
            <ResultsHistory />
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="space-y-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                System Settings
              </h2>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                Configure your analysis preferences, model settings, and system options.
              </p>
            </div>
            
            <div className="bg-white rounded-lg shadow-sm p-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Model Settings */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Settings</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Detection Confidence Threshold
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.1"
                        defaultValue="0.5"
                        className="w-full"
                      />
                      <div className="text-sm text-gray-600 mt-1">0.5 (Recommended)</div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Frame Processing Interval
                      </label>
                      <select className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="5">Every 5th frame (High accuracy, slower)</option>
                        <option value="10" selected>Every 10th frame (Balanced)</option>
                        <option value="25">Every 25th frame (Fast, lower accuracy)</option>
                      </select>
                    </div>
                  </div>
                </div>
                
                {/* Analysis Settings */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Settings</h3>
                  <div className="space-y-4">
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        id="goal-detection"
                        defaultChecked
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="goal-detection" className="text-sm font-medium text-gray-700">
                        Enable Goal Detection
                      </label>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        id="jersey-detection"
                        defaultChecked
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="jersey-detection" className="text-sm font-medium text-gray-700">
                        Enable Jersey Number Detection
                      </label>
                    </div>
                    
                    <div className="flex items-center space-x-3">
                      <input
                        type="checkbox"
                        id="real-time-notifications"
                        defaultChecked
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="real-time-notifications" className="text-sm font-medium text-gray-700">
                        Real-time Event Notifications
                      </label>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-8 pt-6 border-t border-gray-200">
                <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  Save Settings
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedMainDashboard;
