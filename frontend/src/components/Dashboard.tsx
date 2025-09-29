import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Pause, 
  Upload, 
  BarChart3, 
  Users, 
  Target, 
  Zap, 
  TrendingUp,
  Activity,
  Clock,
  Download,
  Settings,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import { useQuery, useMutation } from '@tanstack/react-query';
import VideoUpload from './VideoUpload';
import { 
  analysisAPI, 
  analyticsAPI, 
  type AnalysisResults, 
  type AnalysisJob 
} from '../services/api';

interface DashboardProps {
  className?: string;
}

const Dashboard: React.FC<DashboardProps> = ({ className }) => {
  const [selectedJob, setSelectedJob] = useState<AnalysisJob | null>(null);
  const [showVideoUpload, setShowVideoUpload] = useState(false);
  const [expandedPanel, setExpandedPanel] = useState<string | null>(null);
  const [realTimeMode, setRealTimeMode] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState('all');

  // Fetch jobs
  const { data: jobs, refetch: refetchJobs } = useQuery({
    queryKey: ['jobs'],
    queryFn: analysisAPI.getJobs,
    refetchInterval: 5000, // Poll every 5 seconds
  });

  // Fetch analysis results for selected job
  const { data: analysisResults, isLoading: resultsLoading } = useQuery({
    queryKey: ['analysis-results', selectedJob?.id],
    queryFn: () => analysisAPI.getJobResults(selectedJob!.id),
    enabled: !!selectedJob && selectedJob.status === 'completed',
  });

  // Fetch team statistics
  const { data: teamStats } = useQuery({
    queryKey: ['team-stats', selectedJob?.id],
    queryFn: () => analyticsAPI.getTeamStatistics(selectedJob!.id),
    enabled: !!selectedJob && selectedJob.status === 'completed',
  });

  // Fetch heatmaps
  const { data: heatmaps } = useQuery({
    queryKey: ['heatmaps', selectedJob?.id],
    queryFn: () => analyticsAPI.getHeatmaps(selectedJob!.id),
    enabled: !!selectedJob && selectedJob.status === 'completed',
  });

  // Auto-select latest job
  useEffect(() => {
    if (jobs && jobs.length > 0 && !selectedJob) {
      setSelectedJob(jobs[0]);
    }
  }, [jobs, selectedJob]);

  const handleJobSelect = (job: AnalysisJob) => {
    setSelectedJob(job);
  };

  const handleExportResults = async (format: 'json' | 'csv' | 'pdf') => {
    if (!selectedJob) return;

    try {
      const data = await analyticsAPI.exportResults(selectedJob.id, format);
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([data]));
      const link = document.createElement('a');
      link.href = url;
      link.download = `analysis_results_${selectedJob.id}.${format}`;
      link.click();
      window.URL.revokeObjectURL(url);
      
      toast.success(`Results exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Export failed');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'processing': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <Target className="w-4 h-4" />;
      case 'processing': return <Activity className="w-4 h-4 animate-pulse" />;
      case 'failed': return <Zap className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Godseye AI Dashboard</h1>
              <p className="text-gray-600">Professional Sports Analytics Platform</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowVideoUpload(true)}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                <Upload className="w-4 h-4" />
                <span>Upload Video</span>
              </button>
              
              <button
                onClick={() => setRealTimeMode(!realTimeMode)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  realTimeMode 
                    ? 'bg-green-600 text-white hover:bg-green-700' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                <Activity className="w-4 h-4" />
                <span>{realTimeMode ? 'Live' : 'Static'}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar - Job List */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border">
              <div className="p-4 border-b">
                <h2 className="text-lg font-semibold text-gray-900">Analysis Jobs</h2>
                <p className="text-sm text-gray-600">Recent video analyses</p>
              </div>
              
              <div className="max-h-96 overflow-y-auto">
                {jobs?.map((job) => (
                  <motion.div
                    key={job.id}
                    whileHover={{ scale: 1.02 }}
                    className={`p-4 border-b cursor-pointer transition-colors ${
                      selectedJob?.id === job.id ? 'bg-blue-50 border-blue-200' : 'hover:bg-gray-50'
                    }`}
                    onClick={() => handleJobSelect(job)}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(job.status)}
                        <span className="text-sm font-medium text-gray-900">
                          Job #{job.id.slice(-8)}
                        </span>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                        {job.status}
                      </span>
                    </div>
                    
                    <div className="text-xs text-gray-600 space-y-1">
                      <div>Created: {new Date(job.created_at).toLocaleDateString()}</div>
                      {job.status === 'processing' && (
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-gray-200 rounded-full h-1">
                            <div 
                              className="bg-blue-600 h-1 rounded-full transition-all duration-300"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                          <span>{job.progress}%</span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3 space-y-6">
            {selectedJob ? (
              <>
                {/* Job Overview */}
                <div className="bg-white rounded-lg shadow-sm border p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        Analysis #{selectedJob.id.slice(-8)}
                      </h2>
                      <p className="text-gray-600">
                        Status: <span className={`font-medium ${getStatusColor(selectedJob.status).split(' ')[0]}`}>
                          {selectedJob.status}
                        </span>
                      </p>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {selectedJob.status === 'completed' && (
                        <>
                          <button
                            onClick={() => handleExportResults('json')}
                            className="flex items-center space-x-1 px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                          >
                            <Download className="w-4 h-4" />
                            <span>JSON</span>
                          </button>
                          <button
                            onClick={() => handleExportResults('pdf')}
                            className="flex items-center space-x-1 px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                          >
                            <Download className="w-4 h-4" />
                            <span>PDF</span>
                          </button>
                        </>
                      )}
                    </div>
                  </div>

                  {selectedJob.status === 'processing' && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span>Processing...</span>
                        <span>{selectedJob.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <motion.div
                          className="bg-blue-600 h-2 rounded-full"
                          initial={{ width: 0 }}
                          animate={{ width: `${selectedJob.progress}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>
                  )}

                  {selectedJob.status === 'failed' && selectedJob.error && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                      <p className="text-red-800 text-sm">
                        <strong>Error:</strong> {selectedJob.error}
                      </p>
                    </div>
                  )}
                </div>

                {/* Analysis Results */}
                {selectedJob.status === 'completed' && analysisResults && (
                  <div className="space-y-6">
                    {/* Key Metrics */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white rounded-lg shadow-sm border p-4"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-blue-100 rounded-lg">
                            <Users className="w-5 h-5 text-blue-600" />
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Total Players</p>
                            <p className="text-xl font-semibold text-gray-900">
                              {analysisResults.detection?.total_players || 0}
                            </p>
                          </div>
                        </div>
                      </motion.div>

                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="bg-white rounded-lg shadow-sm border p-4"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-green-100 rounded-lg">
                            <Target className="w-5 h-5 text-green-600" />
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Ball Positions</p>
                            <p className="text-xl font-semibold text-gray-900">
                              {analysisResults.tracking?.ball_trajectory?.length || 0}
                            </p>
                          </div>
                        </div>
                      </motion.div>

                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="bg-white rounded-lg shadow-sm border p-4"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-purple-100 rounded-lg">
                            <Zap className="w-5 h-5 text-purple-600" />
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Events Detected</p>
                            <p className="text-xl font-semibold text-gray-900">
                              {analysisResults.events?.total_events || 0}
                            </p>
                          </div>
                        </div>
                      </motion.div>

                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="bg-white rounded-lg shadow-sm border p-4"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="p-2 bg-orange-100 rounded-lg">
                            <TrendingUp className="w-5 h-5 text-orange-600" />
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Processing Time</p>
                            <p className="text-xl font-semibold text-gray-900">
                              {analysisResults.processing_time?.toFixed(1)}s
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    {/* Team Statistics */}
                    {teamStats && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-white rounded-lg shadow-sm border p-6"
                      >
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Team Statistics</h3>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div className="space-y-4">
                            <h4 className="font-medium text-red-600">Team A</h4>
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Players:</span>
                                <span className="font-medium">{teamStats.team_a?.players || 0}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Goalkeepers:</span>
                                <span className="font-medium">{teamStats.team_a?.goalkeepers || 0}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Total Distance:</span>
                                <span className="font-medium">{(teamStats.team_a?.total_distance || 0).toFixed(1)}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Avg Speed:</span>
                                <span className="font-medium">{(teamStats.team_a?.avg_speed || 0).toFixed(1)} km/h</span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="space-y-4">
                            <h4 className="font-medium text-blue-600">Team B</h4>
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Players:</span>
                                <span className="font-medium">{teamStats.team_b?.players || 0}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Goalkeepers:</span>
                                <span className="font-medium">{teamStats.team_b?.goalkeepers || 0}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Total Distance:</span>
                                <span className="font-medium">{(teamStats.team_b?.total_distance || 0).toFixed(1)}m</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm text-gray-600">Avg Speed:</span>
                                <span className="font-medium">{(teamStats.team_b?.avg_speed || 0).toFixed(1)} km/h</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </motion.div>
                    )}

                    {/* Detailed Results */}
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-white rounded-lg shadow-sm border p-6"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-gray-900">Detailed Analysis</h3>
                        <button
                          onClick={() => setExpandedPanel(expandedPanel === 'details' ? null : 'details')}
                          className="flex items-center space-x-1 text-sm text-gray-600 hover:text-gray-900"
                        >
                          {expandedPanel === 'details' ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                          <span>{expandedPanel === 'details' ? 'Hide' : 'Show'} Details</span>
                        </button>
                      </div>
                      
                      <AnimatePresence>
                        {expandedPanel === 'details' && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="overflow-hidden"
                          >
                            <div className="bg-gray-50 rounded-lg p-4">
                              <pre className="text-sm text-gray-700 overflow-auto max-h-96">
                                {JSON.stringify(analysisResults, null, 2)}
                              </pre>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-white rounded-lg shadow-sm border p-12 text-center">
                <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analysis Selected</h3>
                <p className="text-gray-600 mb-6">
                  Upload a video to start analyzing with Godseye AI
                </p>
                <button
                  onClick={() => setShowVideoUpload(true)}
                  className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Upload Your First Video
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Video Upload Modal */}
      <AnimatePresence>
        {showVideoUpload && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowVideoUpload(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-gray-900">Upload Video for Analysis</h2>
                  <button
                    onClick={() => setShowVideoUpload(false)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>
                
                <VideoUpload
                  onAnalysisComplete={(results) => {
                    setShowVideoUpload(false);
                    refetchJobs();
                    toast.success('Analysis completed!');
                  }}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Dashboard;
