import React, { useState, useEffect } from 'react';
import { 
  Clock, 
  Play, 
  Download, 
  Trash2, 
  Eye, 
  Calendar,
  FileVideo,
  BarChart3,
  Users,
  Target
} from 'lucide-react';

interface AnalysisResult {
  id: string;
  filename: string;
  uploadDate: string;
  duration: number;
  status: 'completed' | 'processing' | 'failed';
  thumbnail?: string;
  stats: {
    totalPlayers: number;
    teamAPlayers: number;
    teamBPlayers: number;
    referees: number;
    balls: number;
    events: number;
    goals: number;
  };
}

const ResultsHistory: React.FC = () => {
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    try {
      // Load from localStorage for now (in production, this would be from API)
      const savedResults = localStorage.getItem('godseye_analysis_results');
      if (savedResults) {
        setResults(JSON.parse(savedResults));
      }
    } catch (error) {
      console.error('Error loading results:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteResult = (id: string) => {
    const updatedResults = results.filter(result => result.id !== id);
    setResults(updatedResults);
    localStorage.setItem('godseye_analysis_results', JSON.stringify(updatedResults));
    
    if (selectedResult?.id === id) {
      setSelectedResult(null);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDuration = (minutes: number) => {
    if (minutes < 60) {
      return `${minutes.toFixed(1)}m`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins.toFixed(0)}m`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Analysis History</h2>
          <p className="text-gray-600">View and manage your previous video analyses</p>
        </div>
        <div className="text-sm text-gray-500">
          {results.length} {results.length === 1 ? 'analysis' : 'analyses'} saved
        </div>
      </div>

      {results.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg">
          <FileVideo className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analyses Yet</h3>
          <p className="text-gray-600 mb-4">
            Upload and analyze your first football video to see results here.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Results List */}
          <div className="lg:col-span-2 space-y-4">
            {results.map((result) => (
              <div
                key={result.id}
                className={`bg-white rounded-lg shadow-sm border-2 transition-all cursor-pointer ${
                  selectedResult?.id === result.id 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedResult(result)}
              >
                <div className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <FileVideo className="w-5 h-5 text-gray-500" />
                        <h3 className="font-semibold text-gray-900 truncate">
                          {result.filename}
                        </h3>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          result.status === 'completed' 
                            ? 'bg-green-100 text-green-800'
                            : result.status === 'processing'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {result.status}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>{formatDate(result.uploadDate)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Clock className="w-4 h-4" />
                          <span>{formatDuration(result.duration)}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          // View analysis
                          window.open(`/analysis/${result.id}`, '_blank');
                        }}
                        className="p-2 text-blue-600 hover:bg-blue-100 rounded-lg transition-colors"
                        title="View Analysis"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteResult(result.id);
                        }}
                        className="p-2 text-red-600 hover:bg-red-100 rounded-lg transition-colors"
                        title="Delete Analysis"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                  
                  {/* Quick Stats */}
                  <div className="mt-3 grid grid-cols-4 gap-4 text-center">
                    <div className="bg-gray-50 rounded-lg p-2">
                      <div className="text-lg font-bold text-gray-900">
                        {result.stats.totalPlayers}
                      </div>
                      <div className="text-xs text-gray-600">Players</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-2">
                      <div className="text-lg font-bold text-gray-900">
                        {result.stats.referees}
                      </div>
                      <div className="text-xs text-gray-600">Referees</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-2">
                      <div className="text-lg font-bold text-gray-900">
                        {result.stats.events}
                      </div>
                      <div className="text-xs text-gray-600">Events</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-2">
                      <div className="text-lg font-bold text-gray-900">
                        {result.stats.goals}
                      </div>
                      <div className="text-xs text-gray-600">Goals</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Selected Result Details */}
          <div className="lg:col-span-1">
            {selectedResult ? (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 sticky top-4">
                <h3 className="font-semibold text-gray-900 mb-4">Analysis Details</h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Video Info</h4>
                    <div className="space-y-1 text-sm text-gray-600">
                      <div>Duration: {formatDuration(selectedResult.duration)}</div>
                      <div>Uploaded: {formatDate(selectedResult.uploadDate)}</div>
                      <div>Status: <span className="capitalize">{selectedResult.status}</span></div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Detection Results</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Team A Players:</span>
                        <span className="font-medium">{selectedResult.stats.teamAPlayers}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Team B Players:</span>
                        <span className="font-medium">{selectedResult.stats.teamBPlayers}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Referees:</span>
                        <span className="font-medium">{selectedResult.stats.referees}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Balls:</span>
                        <span className="font-medium">{selectedResult.stats.balls}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Events</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Total Events:</span>
                        <span className="font-medium">{selectedResult.stats.events}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Goals:</span>
                        <span className="font-medium text-green-600">{selectedResult.stats.goals}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="pt-4 border-t border-gray-200">
                    <div className="flex space-x-2">
                      <button
                        onClick={() => {
                          // View full analysis
                          window.open(`/analysis/${selectedResult.id}`, '_blank');
                        }}
                        className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
                      >
                        <Eye className="w-4 h-4" />
                        <span>View Full</span>
                      </button>
                      <button
                        onClick={() => {
                          // Download results
                          const dataStr = JSON.stringify(selectedResult, null, 2);
                          const dataBlob = new Blob([dataStr], { type: 'application/json' });
                          const url = URL.createObjectURL(dataBlob);
                          const link = document.createElement('a');
                          link.href = url;
                          link.download = `${selectedResult.filename}_analysis.json`;
                          link.click();
                        }}
                        className="flex-1 flex items-center justify-center space-x-2 px-3 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm"
                      >
                        <Download className="w-4 h-4" />
                        <span>Download</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-gray-50 rounded-lg p-8 text-center">
                <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <p className="text-gray-600">Select an analysis to view details</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsHistory;
