import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Play, Camera, BarChart3, Settings } from 'lucide-react';

const SimpleDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('upload');

  const tabs = [
    { id: 'upload', label: 'Upload Video', icon: Upload },
    { id: 'live', label: 'Live Analytics', icon: Play },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <Camera className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Godseye AI</h1>
                  <p className="text-sm text-gray-600">Sports Analytics Platform</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:w-64 flex-shrink-0">
            <nav className="bg-white rounded-lg shadow-sm border p-4">
              <div className="space-y-2">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-left transition-colors ${
                        isActive
                          ? 'bg-blue-50 text-blue-700 border border-blue-200'
                          : 'text-gray-700 hover:bg-gray-50'
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      <span className="font-medium">{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className="bg-white rounded-lg shadow-sm border p-6"
            >
              {activeTab === 'upload' && (
                <div className="text-center py-12">
                  <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    Upload Football Video
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Drag and drop your football video here to start AI analysis
                  </p>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 hover:border-blue-400 transition-colors">
                    <p className="text-lg text-gray-700">Drop your video here or click to select</p>
                    <p className="text-sm text-gray-500 mt-2">MP4, AVI, MOV, MKV, WebM (Max 500MB)</p>
                  </div>
                </div>
              )}

              {activeTab === 'live' && (
                <div className="text-center py-12">
                  <Play className="w-16 h-16 mx-auto text-green-400 mb-4" />
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    Live Analytics
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Real-time match analytics like Premier League broadcasts
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-gray-900">0:00</div>
                      <div className="text-sm text-gray-600">Match Time</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-gray-900">OFFLINE</div>
                      <div className="text-sm text-gray-600">Status</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="text-2xl font-bold text-gray-900">0</div>
                      <div className="text-sm text-gray-600">Events</div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'analytics' && (
                <div className="text-center py-12">
                  <BarChart3 className="w-16 h-16 mx-auto text-purple-400 mb-4" />
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    Analytics Dashboard
                  </h2>
                  <p className="text-gray-600 mb-6">
                    View detailed analysis results and statistics
                  </p>
                  <div className="text-gray-500">
                    No analysis results available yet. Upload a video to get started.
                  </div>
                </div>
              )}

              {activeTab === 'settings' && (
                <div className="text-center py-12">
                  <Settings className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                    Settings
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Configure system settings and preferences
                  </p>
                  <div className="max-w-md mx-auto space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-700">Detection Model</span>
                      <select className="px-3 py-1 border border-gray-300 rounded">
                        <option>YOLOv8n</option>
                        <option>YOLOv8s</option>
                        <option>YOLOv8m</option>
                      </select>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-700">Tracking Algorithm</span>
                      <select className="px-3 py-1 border border-gray-300 rounded">
                        <option>DeepSort</option>
                        <option>ByteTrack</option>
                        <option>StrongSORT</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimpleDashboard;
