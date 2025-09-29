import axios from 'axios';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const INFERENCE_API_URL = process.env.REACT_APP_INFERENCE_URL || 'http://localhost:8001';

// Create axios instances
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const inferenceClient = axios.create({
  baseURL: INFERENCE_API_URL,
  timeout: 60000, // Longer timeout for inference
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptors
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

inferenceClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Types
export interface VideoUploadResponse {
  id: string;
  file: File;
  preview: string;
  status: string;
  created_at: string;
}

export interface AnalysisJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  results?: AnalysisResults;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface AnalysisResults {
  detection: {
    total_players: number;
    team_a_players: number;
    team_b_players: number;
    team_a_goalkeepers: number;
    team_b_goalkeepers: number;
    referees: number;
    balls: number;
    detections: Detection[];
  };
  tracking: {
    total_tracks: number;
    ball_trajectory: BallPosition[];
    player_tracks: PlayerTrack[];
    team_statistics: TeamStatistics;
  };
  pose: {
    total_poses: number;
    poses: PoseEstimation[];
  };
  events: {
    total_events: number;
    events: Event[];
  };
  tactical: {
    formations: FormationAnalysis[];
    possession: PossessionData[];
  };
  processing_time: number;
  model_info: {
    detection_model: string;
    tracking_model: string;
    pose_model: string;
    event_model: string;
  };
}

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  team_id?: number;
  role?: string;
}

export interface BallPosition {
  frame: number;
  timestamp: number;
  position: [number, number];
  confidence: number;
  speed?: number;
}

export interface PlayerTrack {
  track_id: number;
  class_id: number;
  class_name: string;
  team_id?: number;
  role?: string;
  positions: [number, number][];
  speeds: number[];
  total_distance: number;
  max_speed: number;
  avg_speed: number;
}

export interface TeamStatistics {
  team_a: {
    players: number;
    goalkeepers: number;
    total_distance: number;
    avg_speed: number;
    possession_percentage: number;
  };
  team_b: {
    players: number;
    goalkeepers: number;
    total_distance: number;
    avg_speed: number;
    possession_percentage: number;
  };
}

export interface PoseEstimation {
  person_id: number;
  keypoints: Keypoint[];
  confidence: number;
  bbox: [number, number, number, number];
}

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
  name: string;
}

export interface Event {
  event_id: number;
  event_name: string;
  confidence: number;
  timestamp: number;
  frame: number;
  bbox?: [number, number, number, number];
  description?: string;
}

export interface FormationAnalysis {
  timestamp: number;
  team_a_formation: string;
  team_b_formation: string;
  confidence: number;
}

export interface PossessionData {
  timestamp: number;
  team_a_possession: number;
  team_b_possession: number;
  ball_team: number;
}

// Authentication API
export const authAPI = {
  login: async (email: string, password: string) => {
    const response = await apiClient.post('/api/v1/auth/login/', {
      email,
      password,
    });
    return response.data;
  },

  register: async (userData: any) => {
    const response = await apiClient.post('/api/v1/auth/register/', userData);
    return response.data;
  },

  refreshToken: async (refreshToken: string) => {
    const response = await apiClient.post('/api/v1/auth/refresh/', {
      refresh: refreshToken,
    });
    return response.data;
  },

  logout: async () => {
    const response = await apiClient.post('/api/v1/auth/logout/');
    return response.data;
  },
};

// Video API
export const videoAPI = {
  upload: async (file: File): Promise<VideoUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await apiClient.post('/api/v1/videos/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getVideos: async () => {
    const response = await apiClient.get('/api/v1/videos/');
    return response.data;
  },

  getVideo: async (id: string) => {
    const response = await apiClient.get(`/api/v1/videos/${id}/`);
    return response.data;
  },

  deleteVideo: async (id: string) => {
    const response = await apiClient.delete(`/api/v1/videos/${id}/`);
    return response.data;
  },
};

// Analysis API
export const analysisAPI = {
  startAnalysis: async (data: {
    video_id: string;
    analysis_types: string[];
    options?: any;
  }): Promise<AnalysisJob> => {
    const response = await apiClient.post('/api/v1/jobs/', data);
    return response.data;
  },

  getJobStatus: async (jobId: string): Promise<AnalysisJob> => {
    const response = await apiClient.get(`/api/v1/jobs/${jobId}/`);
    return response.data;
  },

  getJobResults: async (jobId: string): Promise<AnalysisResults> => {
    const response = await apiClient.get(`/api/v1/jobs/${jobId}/results/`);
    return response.data;
  },

  cancelJob: async (jobId: string) => {
    const response = await apiClient.post(`/api/v1/jobs/${jobId}/cancel/`);
    return response.data;
  },

  getJobs: async () => {
    const response = await apiClient.get('/api/v1/jobs/');
    return response.data;
  },
};

// Inference API
export const inferenceAPI = {
  detectObjects: async (file: File, options: any = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('request', JSON.stringify({
      model_type: 'detection',
      confidence_threshold: 0.5,
      iou_threshold: 0.45,
      max_detections: 1000,
      return_visualization: true,
      team_classification: true,
      ...options,
    }));

    const response = await inferenceClient.post('/inference/detect', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  trackObjects: async (file: File, options: any = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('request', JSON.stringify({
      model_type: 'tracking',
      confidence_threshold: 0.5,
      return_visualization: true,
      ...options,
    }));

    const response = await inferenceClient.post('/inference/track', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  estimatePose: async (file: File, options: any = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('request', JSON.stringify({
      model_type: 'pose',
      confidence_threshold: 0.5,
      return_visualization: true,
      ...options,
    }));

    const response = await inferenceClient.post('/inference/pose', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  detectEvents: async (file: File, options: any = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('request', JSON.stringify({
      model_type: 'events',
      confidence_threshold: 0.5,
      ...options,
    }));

    const response = await inferenceClient.post('/inference/events', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  batchInference: async (data: {
    video_path: string;
    output_path: string;
    model_type: string;
    frame_interval: number;
    confidence_threshold: number;
    return_annotated_video: boolean;
  }) => {
    const response = await inferenceClient.post('/inference/batch', data);
    return response.data;
  },

  getBatchStatus: async (taskId: string) => {
    const response = await inferenceClient.get(`/inference/batch/${taskId}`);
    return response.data;
  },
};

// Analytics API
export const analyticsAPI = {
  getTeamStatistics: async (jobId: string) => {
    const response = await apiClient.get(`/api/v1/analytics/teams/${jobId}/`);
    return response.data;
  },

  getPlayerStatistics: async (jobId: string, playerId?: string) => {
    const url = playerId 
      ? `/api/v1/analytics/players/${jobId}/${playerId}/`
      : `/api/v1/analytics/players/${jobId}/`;
    const response = await apiClient.get(url);
    return response.data;
  },

  getHeatmaps: async (jobId: string, type: 'team' | 'player' | 'ball' = 'team') => {
    const response = await apiClient.get(`/api/v1/analytics/heatmaps/${jobId}/?type=${type}`);
    return response.data;
  },

  getFormationAnalysis: async (jobId: string) => {
    const response = await apiClient.get(`/api/v1/analytics/formations/${jobId}/`);
    return response.data;
  },

  getEventTimeline: async (jobId: string) => {
    const response = await apiClient.get(`/api/v1/analytics/events/${jobId}/`);
    return response.data;
  },

  exportResults: async (jobId: string, format: 'json' | 'csv' | 'pdf' = 'json') => {
    const response = await apiClient.get(`/api/v1/analytics/export/${jobId}/?format=${format}`, {
      responseType: 'blob',
    });
    return response.data;
  },
};

// Model API
export const modelAPI = {
  getModels: async () => {
    const response = await inferenceClient.get('/models');
    return response.data;
  },

  getModelInfo: async (modelName: string) => {
    const response = await inferenceClient.get(`/models/${modelName}`);
    return response.data;
  },
};

// Health check
export const healthAPI = {
  checkBackend: async () => {
    const response = await apiClient.get('/health/');
    return response.data;
  },

  checkInference: async () => {
    const response = await inferenceClient.get('/health');
    return response.data;
  },
};

// Simple inference API functions for testing
export const uploadVideo = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await inferenceClient.post('/upload-video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const startAnalysis = async (data: { video_id: string; analysis_types: string[] }) => {
  // For now, just return a mock job ID since upload-video handles everything
  return { job_id: data.video_id, status: 'processing' };
};

export const getAnalysisResults = async (jobId: string) => {
  const response = await inferenceClient.get(`/analysis/${jobId}`);
  return response.data;
};

export default {
  auth: authAPI,
  video: videoAPI,
  analysis: analysisAPI,
  inference: inferenceAPI,
  analytics: analyticsAPI,
  model: modelAPI,
  health: healthAPI,
};
