import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Play, 
  Pause, 
  RotateCcw, 
  CheckCircle, 
  AlertCircle,
  X,
  FileVideo,
  BarChart3,
  Users,
  Target,
  Zap,
  Camera,
  Download,
  Eye
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import AnalysisResults from './AnalysisResults';

interface VideoUploadProps {
  onUploadSuccess?: (videoId: string) => void;
  onAnalysisComplete?: (results: any) => void;
  tenantId?: string;
}

interface UploadedFile {
  file: File;
  preview: string;
  id: string;
}

interface AnalysisJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  results?: any;
  error?: string;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onUploadSuccess, onAnalysisComplete, tenantId }) => {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [analysisJob, setAnalysisJob] = useState<AnalysisJob | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [showAnalysisPage, setShowAnalysisPage] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const preview = URL.createObjectURL(file);
      setUploadedFile({
        file,
        preview,
        id: Math.random().toString(36).substr(2, 9)
      });
      
      // Simulate upload process
      setIsUploading(true);
      toast.success('Video uploaded successfully!');
      
      setTimeout(() => {
        setIsUploading(false);
        onUploadSuccess?.(file.name);
      }, 2000);
    }
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
    maxSize: 2 * 1024 * 1024 * 1024 // 2GB
  });

  const handleStartAnalysis = async () => {
    if (uploadedFile) {
      const jobId = Math.random().toString(36).substr(2, 9);
      setAnalysisJob({
        id: jobId,
        status: 'processing',
        progress: 0
      });
      
      toast.success('Analysis started!');
      
      try {
        // Upload video to API
        const formData = new FormData();
        formData.append('file', uploadedFile.file);
        
        const response = await fetch('http://localhost:8001/upload-video', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error('Failed to upload video');
        }
        
        const result = await response.json();
        
        // Store job ID for later retrieval
        setAnalysisJob(prev => prev ? { ...prev, id: result.job_id } : null);
        
        // Poll for progress updates
        const pollInterval = setInterval(async () => {
          try {
            // First check progress
            const progressResponse = await fetch(`http://localhost:8001/progress/${result.job_id}`);
            if (progressResponse.ok) {
              const progressData = await progressResponse.json();
              
              // Update progress
              setAnalysisJob(prev => prev ? {
                ...prev,
                progress: progressData.progress,
                status: progressData.status
              } : null);
              
              // If completed, get results
              if (progressData.status === 'completed') {
                clearInterval(pollInterval);
                
                const resultsResponse = await fetch(`http://localhost:8001/analysis/${result.job_id}`);
                if (resultsResponse.ok) {
                  const analysisData = await resultsResponse.json();
                  
                  setAnalysisJob(prev => prev ? {
                    ...prev,
                    status: 'completed',
                    progress: 100,
                    results: analysisData.results
                  } : null);
                  
                  onAnalysisComplete?.(analysisData.results);
                  toast.success('Analysis completed!');
                  
                  // Automatically show the full analysis page
                  setTimeout(() => {
                    setShowAnalysisPage(true);
                  }, 1000);
                }
              } else if (progressData.status === 'failed') {
                clearInterval(pollInterval);
                setAnalysisJob(prev => prev ? {
                  ...prev,
                  status: 'failed',
                  error: progressData.message
                } : null);
                toast.error('Analysis failed');
              }
            }
          } catch (error) {
            console.error('Error polling for progress:', error);
          }
        }, 1000); // Poll every second for real-time updates
        
      } catch (error) {
        console.error('Analysis failed:', error);
        setAnalysisJob(prev => prev ? {
          ...prev,
          status: 'failed',
          error: error instanceof Error ? error.message : 'Unknown error'
        } : null);
        toast.error('Analysis failed');
      }
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    setAnalysisJob(null);
    setShowResults(false);
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600';
      case 'processing': return 'text-blue-600';
      case 'failed': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-5 h-5" />;
      case 'processing': return <Zap className="w-5 h-5 animate-pulse" />;
      case 'failed': return <AlertCircle className="w-5 h-5" />;
      default: return <FileVideo className="w-5 h-5" />;
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Godseye AI Sports Analytics
        </h1>
        <p className="text-gray-600">
          Upload your football video and get comprehensive AI-powered analysis
        </p>
      </div>

      {/* Upload Area */}
      {!uploadedFile && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 transition-colors cursor-pointer"
          {...getRootProps()}
        >
          <input {...getInputProps()} />
          <motion.div
            animate={{ scale: isDragActive ? 1.05 : 1 }}
            className="space-y-4"
          >
            <Upload className="w-16 h-16 mx-auto text-gray-400" />
            <div>
              <p className="text-xl font-semibold text-gray-700">
                {isDragActive ? 'Drop your video here' : 'Upload Football Video'}
              </p>
              <p className="text-gray-500 mt-2">
                Drag & drop or click to select (MP4, AVI, MOV, MKV, WebM)
              </p>
              <p className="text-sm text-gray-400 mt-1">
                Maximum file size: 2GB
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Upload Progress */}
      {isUploading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-lg p-6"
        >
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <span className="text-lg font-medium text-gray-900">Uploading video...</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <motion.div
              className="bg-blue-600 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: "100%" }}
              transition={{ duration: 2 }}
            />
          </div>
        </motion.div>
      )}

      {/* Video Player and Analysis */}
      {uploadedFile && !isUploading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Video Player */}
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            <div className="relative">
              <video
                src={uploadedFile.preview}
                width="100%"
                height="400px"
                controls
                className="w-full"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
              />
              
              {/* Video Info Overlay */}
              <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded">
                <span className="text-sm font-medium">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
              </div>
            </div>

            {/* Video Controls */}
            <div className="p-4 bg-gray-50 border-t">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    <span>{isPlaying ? 'Pause' : 'Play'}</span>
                  </button>
                  
                  <div className="text-sm text-gray-600">
                    <span className="font-medium">{uploadedFile.file.name}</span>
                    <span className="ml-2">
                      ({(uploadedFile.file.size / (1024 * 1024)).toFixed(1)} MB)
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={handleReset}
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>Reset</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Analysis Controls */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">
                AI Analysis
              </h2>
              <div className="flex items-center space-x-2">
                <Users className="w-5 h-5 text-blue-600" />
                <Target className="w-5 h-5 text-green-600" />
                <BarChart3 className="w-5 h-5 text-purple-600" />
              </div>
            </div>

            {!analysisJob && (
              <div className="text-center py-8">
                <p className="text-gray-600 mb-4">
                  Start comprehensive AI analysis of your football video
                </p>
                <button
                  onClick={handleStartAnalysis}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105"
                >
                  Start AI Analysis
                </button>
              </div>
            )}

            {/* Analysis Progress */}
            {analysisJob && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`${getStatusColor(analysisJob.status)}`}>
                      {getStatusIcon(analysisJob.status)}
                    </div>
                    <span className="font-medium">
                      {analysisJob.status === 'processing' && 'Analyzing Video...'}
                      {analysisJob.status === 'completed' && 'Analysis Complete'}
                      {analysisJob.status === 'failed' && 'Analysis Failed'}
                    </span>
                  </div>
                  <span className="text-sm text-gray-500">
                    Job ID: {analysisJob.id}
                  </span>
                </div>

                {analysisJob.status === 'processing' && (
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>Progress</span>
                      <span>{Math.round(analysisJob.progress)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <motion.div
                        className="bg-blue-600 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${analysisJob.progress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <div className="text-sm text-gray-600 text-center">
                      {analysisJob.progress < 10 && "Initializing model..."}
                      {analysisJob.progress >= 10 && analysisJob.progress < 90 && "Processing video frames..."}
                      {analysisJob.progress >= 90 && analysisJob.progress < 100 && "Loading results..."}
                    </div>
                  </div>
                )}

                {analysisJob.status === 'completed' && (
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={() => setShowAnalysisPage(true)}
                      className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                    >
                      <Eye className="w-4 h-4" />
                      <span>View Full Analysis</span>
                    </button>
                    
                    <button
                      onClick={() => setShowResults(!showResults)}
                      className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      <BarChart3 className="w-4 h-4" />
                      <span>{showResults ? 'Hide' : 'Show'} Quick Stats</span>
                    </button>
                    
                    <button
                      onClick={() => {
                        // Download results
                        const dataStr = JSON.stringify(analysisJob.results, null, 2);
                        const dataBlob = new Blob([dataStr], { type: 'application/json' });
                        const url = URL.createObjectURL(dataBlob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = 'analysis_results.json';
                        link.click();
                      }}
                      className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download Results</span>
                    </button>
                  </div>
                )}

                {analysisJob.status === 'failed' && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-800">
                      <strong>Error:</strong> {analysisJob.error}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Results Display */}
          <AnimatePresence>
            {showResults && analysisJob?.results && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="bg-white rounded-lg shadow-lg p-6"
              >
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Analysis Results
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {/* Player Detection */}
                  <div className="bg-blue-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Users className="w-5 h-5 text-blue-600" />
                      <span className="font-medium text-blue-900">Player Detection</span>
                    </div>
                    <p className="text-sm text-blue-700">
                      {analysisJob.results.detection?.total_players || 0} players detected
                    </p>
                    <p className="text-xs text-blue-600 mt-1">
                      Team A: {analysisJob.results.detection?.team_a_players || 0} | 
                      Team B: {analysisJob.results.detection?.team_b_players || 0}
                    </p>
                  </div>

                  {/* Ball Tracking */}
                  <div className="bg-green-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Target className="w-5 h-5 text-green-600" />
                      <span className="font-medium text-green-900">Ball Tracking</span>
                    </div>
                    <p className="text-sm text-green-700">
                      {analysisJob.results.tracking?.ball_trajectory?.length || 0} ball positions tracked
                    </p>
                    <p className="text-xs text-green-600 mt-1">
                      {analysisJob.results.tracking?.player_tracks || 0} player tracks
                    </p>
                  </div>

                  {/* Events Detected */}
                  <div className="bg-purple-50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className="w-5 h-5 text-purple-600" />
                      <span className="font-medium text-purple-900">Events</span>
                    </div>
                    <p className="text-sm text-purple-700">
                      {analysisJob.results.events?.length || 0} events detected
                    </p>
                    <p className="text-xs text-purple-600 mt-1">
                      Goals, shots, passes, tackles
                    </p>
                  </div>
                </div>

                {/* Statistics */}
                {analysisJob.results.statistics && (
                  <div className="mt-6">
                    <h4 className="font-medium text-gray-900 mb-3">Match Statistics</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-lg font-bold text-gray-900">
                          {analysisJob.results.statistics.possession?.team_a || 0}%
                        </div>
                        <div className="text-xs text-gray-600">Team A Possession</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-lg font-bold text-gray-900">
                          {analysisJob.results.statistics.possession?.team_b || 0}%
                        </div>
                        <div className="text-xs text-gray-600">Team B Possession</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-lg font-bold text-gray-900">
                          {(analysisJob.results.statistics.shots?.team_a || 0) + (analysisJob.results.statistics.shots?.team_b || 0)}
                        </div>
                        <div className="text-xs text-gray-600">Total Shots</div>
                      </div>
                      <div className="text-center p-3 bg-gray-50 rounded-lg">
                        <div className="text-lg font-bold text-gray-900">
                          {(analysisJob.results.statistics.passes?.team_a || 0) + (analysisJob.results.statistics.passes?.team_b || 0)}
                        </div>
                        <div className="text-xs text-gray-600">Total Passes</div>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* Analysis Results Page */}
      {showAnalysisPage && analysisJob && (
        <AnalysisResults
          jobId={analysisJob.id}
          onBack={() => setShowAnalysisPage(false)}
        />
      )}
    </div>
  );
};

export default VideoUpload;