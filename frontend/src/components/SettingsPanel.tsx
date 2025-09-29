import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Settings,
  Save,
  RotateCcw,
  Camera,
  Wifi,
  HardDrive,
  Monitor,
  Cpu,
  MemoryStick,
  Volume2,
  VolumeX,
  Bell,
  BellOff,
  Shield,
  Key,
  User,
  Database,
  Cloud,
  Download,
  Upload,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react';
import { toast } from 'react-hot-toast';

const SettingsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'general' | 'hardware' | 'ai' | 'notifications' | 'security' | 'system'>('general');
  const [settings, setSettings] = useState({
    // General Settings
    language: 'en',
    theme: 'light',
    timezone: 'UTC',
    dateFormat: 'MM/DD/YYYY',
    autoSave: true,
    notifications: true,
    
    // Hardware Settings
    cameraResolution: '1080p',
    recordingQuality: 'high',
    frameRate: 30,
    audioEnabled: true,
    microphoneSensitivity: 75,
    
    // AI Settings
    detectionModel: 'yolov8n',
    trackingAlgorithm: 'deepsort',
    confidenceThreshold: 0.5,
    nmsThreshold: 0.4,
    maxFps: 30,
    enablePoseEstimation: true,
    enableEventDetection: true,
    enableHeatmap: true,
    
    // Notification Settings
    emailNotifications: true,
    pushNotifications: true,
    soundNotifications: true,
    analysisComplete: true,
    hardwareDisconnected: true,
    lowStorage: true,
    
    // Security Settings
    twoFactorAuth: false,
    sessionTimeout: 30,
    dataEncryption: true,
    auditLogging: true,
    
    // System Settings
    maxStorage: 100,
    autoCleanup: true,
    backupEnabled: true,
    cloudSync: false
  });

  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const handleSettingChange = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSaveSettings = async () => {
    setIsSaving(true);
    
    // Simulate save operation
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setIsSaving(false);
    setHasChanges(false);
    toast.success('Settings saved successfully!');
  };

  const handleResetSettings = () => {
    if (window.confirm('Are you sure you want to reset all settings to default?')) {
      // Reset to default values
      setSettings({
        language: 'en',
        theme: 'light',
        timezone: 'UTC',
        dateFormat: 'MM/DD/YYYY',
        autoSave: true,
        notifications: true,
        cameraResolution: '1080p',
        recordingQuality: 'high',
        frameRate: 30,
        audioEnabled: true,
        microphoneSensitivity: 75,
        detectionModel: 'yolov8n',
        trackingAlgorithm: 'deepsort',
        confidenceThreshold: 0.5,
        nmsThreshold: 0.4,
        maxFps: 30,
        enablePoseEstimation: true,
        enableEventDetection: true,
        enableHeatmap: true,
        emailNotifications: true,
        pushNotifications: true,
        soundNotifications: true,
        analysisComplete: true,
        hardwareDisconnected: true,
        lowStorage: true,
        twoFactorAuth: false,
        sessionTimeout: 30,
        dataEncryption: true,
        auditLogging: true,
        maxStorage: 100,
        autoCleanup: true,
        backupEnabled: true,
        cloudSync: false
      });
      setHasChanges(true);
      toast.info('Settings reset to default values');
    }
  };

  const tabs = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'hardware', label: 'Hardware', icon: Camera },
    { id: 'ai', label: 'AI Models', icon: Cpu },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'system', label: 'System', icon: Database }
  ];

  const renderGeneralTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">General Preferences</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Language
            </label>
            <select
              value={settings.language}
              onChange={(e) => handleSettingChange('language', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Theme
            </label>
            <select
              value={settings.theme}
              onChange={(e) => handleSettingChange('theme', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="auto">Auto</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Timezone
            </label>
            <select
              value={settings.timezone}
              onChange={(e) => handleSettingChange('timezone', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="UTC">UTC</option>
              <option value="EST">Eastern Time</option>
              <option value="PST">Pacific Time</option>
              <option value="GMT">Greenwich Mean Time</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Date Format
            </label>
            <select
              value={settings.dateFormat}
              onChange={(e) => handleSettingChange('dateFormat', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="MM/DD/YYYY">MM/DD/YYYY</option>
              <option value="DD/MM/YYYY">DD/MM/YYYY</option>
              <option value="YYYY-MM-DD">YYYY-MM-DD</option>
            </select>
          </div>
        </div>

        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Auto-save recordings</label>
              <p className="text-sm text-gray-500">Automatically save match recordings</p>
            </div>
            <input
              type="checkbox"
              checked={settings.autoSave}
              onChange={(e) => handleSettingChange('autoSave', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Enable notifications</label>
              <p className="text-sm text-gray-500">Receive system notifications</p>
            </div>
            <input
              type="checkbox"
              checked={settings.notifications}
              onChange={(e) => handleSettingChange('notifications', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderHardwareTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Camera Settings</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Camera Resolution
            </label>
            <select
              value={settings.cameraResolution}
              onChange={(e) => handleSettingChange('cameraResolution', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="720p">720p (HD)</option>
              <option value="1080p">1080p (Full HD)</option>
              <option value="4k">4K (Ultra HD)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Recording Quality
            </label>
            <select
              value={settings.recordingQuality}
              onChange={(e) => handleSettingChange('recordingQuality', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="low">Low Quality (Faster Processing)</option>
              <option value="medium">Medium Quality</option>
              <option value="high">High Quality</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Frame Rate: {settings.frameRate} FPS
            </label>
            <input
              type="range"
              min="15"
              max="60"
              step="5"
              value={settings.frameRate}
              onChange={(e) => handleSettingChange('frameRate', parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Microphone Sensitivity: {settings.microphoneSensitivity}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={settings.microphoneSensitivity}
              onChange={(e) => handleSettingChange('microphoneSensitivity', parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Enable Audio Recording</label>
              <p className="text-sm text-gray-500">Record audio with video</p>
            </div>
            <input
              type="checkbox"
              checked={settings.audioEnabled}
              onChange={(e) => handleSettingChange('audioEnabled', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderAITab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Model Configuration</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Detection Model
            </label>
            <select
              value={settings.detectionModel}
              onChange={(e) => handleSettingChange('detectionModel', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="yolov8n">YOLOv8 Nano (Fastest)</option>
              <option value="yolov8s">YOLOv8 Small</option>
              <option value="yolov8m">YOLOv8 Medium</option>
              <option value="yolov8l">YOLOv8 Large</option>
              <option value="yolov8x">YOLOv8 Extra Large (Most Accurate)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tracking Algorithm
            </label>
            <select
              value={settings.trackingAlgorithm}
              onChange={(e) => handleSettingChange('trackingAlgorithm', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="deepsort">DeepSort (Recommended)</option>
              <option value="bytetrack">ByteTrack</option>
              <option value="strongsort">StrongSORT</option>
              <option value="custom">Custom Tracker</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold: {settings.confidenceThreshold}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.1"
              value={settings.confidenceThreshold}
              onChange={(e) => handleSettingChange('confidenceThreshold', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              NMS Threshold: {settings.nmsThreshold}
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.1"
              value={settings.nmsThreshold}
              onChange={(e) => handleSettingChange('nmsThreshold', parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Processing FPS: {settings.maxFps}
            </label>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={settings.maxFps}
              onChange={(e) => handleSettingChange('maxFps', parseInt(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Enable Pose Estimation</label>
              <p className="text-sm text-gray-500">Analyze player body positions and movements</p>
            </div>
            <input
              type="checkbox"
              checked={settings.enablePoseEstimation}
              onChange={(e) => handleSettingChange('enablePoseEstimation', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Enable Event Detection</label>
              <p className="text-sm text-gray-500">Automatically detect goals, fouls, and other events</p>
            </div>
            <input
              type="checkbox"
              checked={settings.enableEventDetection}
              onChange={(e) => handleSettingChange('enableEventDetection', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Enable Heatmap Generation</label>
              <p className="text-sm text-gray-500">Generate player movement heatmaps</p>
            </div>
            <input
              type="checkbox"
              checked={settings.enableHeatmap}
              onChange={(e) => handleSettingChange('enableHeatmap', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Notification Preferences</h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Email Notifications</label>
              <p className="text-sm text-gray-500">Receive notifications via email</p>
            </div>
            <input
              type="checkbox"
              checked={settings.emailNotifications}
              onChange={(e) => handleSettingChange('emailNotifications', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Push Notifications</label>
              <p className="text-sm text-gray-500">Receive browser push notifications</p>
            </div>
            <input
              type="checkbox"
              checked={settings.pushNotifications}
              onChange={(e) => handleSettingChange('pushNotifications', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Sound Notifications</label>
              <p className="text-sm text-gray-500">Play sound for notifications</p>
            </div>
            <input
              type="checkbox"
              checked={settings.soundNotifications}
              onChange={(e) => handleSettingChange('soundNotifications', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="border-t pt-4">
            <h4 className="font-medium text-gray-900 mb-3">Notification Types</h4>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Analysis Complete</label>
                  <p className="text-sm text-gray-500">When video analysis is finished</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.analysisComplete}
                  onChange={(e) => handleSettingChange('analysisComplete', e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Hardware Disconnected</label>
                  <p className="text-sm text-gray-500">When camera or hardware disconnects</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.hardwareDisconnected}
                  onChange={(e) => handleSettingChange('hardwareDisconnected', e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-700">Low Storage Warning</label>
                  <p className="text-sm text-gray-500">When storage space is running low</p>
                </div>
                <input
                  type="checkbox"
                  checked={settings.lowStorage}
                  onChange={(e) => handleSettingChange('lowStorage', e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSecurityTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Security Settings</h3>
        
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Two-Factor Authentication</label>
              <p className="text-sm text-gray-500">Add an extra layer of security to your account</p>
            </div>
            <input
              type="checkbox"
              checked={settings.twoFactorAuth}
              onChange={(e) => handleSettingChange('twoFactorAuth', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Session Timeout: {settings.sessionTimeout} minutes
            </label>
            <input
              type="range"
              min="5"
              max="120"
              step="5"
              value={settings.sessionTimeout}
              onChange={(e) => handleSettingChange('sessionTimeout', parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Data Encryption</label>
              <p className="text-sm text-gray-500">Encrypt stored data and communications</p>
            </div>
            <input
              type="checkbox"
              checked={settings.dataEncryption}
              onChange={(e) => handleSettingChange('dataEncryption', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Audit Logging</label>
              <p className="text-sm text-gray-500">Log all system activities for security monitoring</p>
            </div>
            <input
              type="checkbox"
              checked={settings.auditLogging}
              onChange={(e) => handleSettingChange('auditLogging', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>
        </div>
      </div>
    </div>
  );

  const renderSystemTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Settings</h3>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Storage: {settings.maxStorage} GB
            </label>
            <input
              type="range"
              min="10"
              max="1000"
              step="10"
              value={settings.maxStorage}
              onChange={(e) => handleSettingChange('maxStorage', parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Auto Cleanup</label>
              <p className="text-sm text-gray-500">Automatically delete old recordings to free space</p>
            </div>
            <input
              type="checkbox"
              checked={settings.autoCleanup}
              onChange={(e) => handleSettingChange('autoCleanup', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Backup Enabled</label>
              <p className="text-sm text-gray-500">Automatically backup important data</p>
            </div>
            <input
              type="checkbox"
              checked={settings.backupEnabled}
              onChange={(e) => handleSettingChange('backupEnabled', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-700">Cloud Sync</label>
              <p className="text-sm text-gray-500">Sync data with cloud storage</p>
            </div>
            <input
              type="checkbox"
              checked={settings.cloudSync}
              onChange={(e) => handleSettingChange('cloudSync', e.target.checked)}
              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">System Version</span>
              <span className="font-medium">Godseye AI v1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Last Updated</span>
              <span className="font-medium">2025-01-28</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Storage Used</span>
              <span className="font-medium">45.2 GB / 100 GB</span>
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">CPU Usage</span>
              <span className="font-medium">23%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Memory Usage</span>
              <span className="font-medium">1.2 GB / 8 GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Network Status</span>
              <span className="font-medium text-green-600">Connected</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'general': return renderGeneralTab();
      case 'hardware': return renderHardwareTab();
      case 'ai': return renderAITab();
      case 'notifications': return renderNotificationsTab();
      case 'security': return renderSecurityTab();
      case 'system': return renderSystemTab();
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
              <p className="text-gray-600">Configure your Godseye AI experience</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={handleResetSettings}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Reset</span>
              </button>
              
              <button
                onClick={handleSaveSettings}
                disabled={!hasChanges || isSaving}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  hasChanges && !isSaving
                    ? 'bg-blue-600 text-white hover:bg-blue-700'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {isSaving ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                <span>{isSaving ? 'Saving...' : 'Save Changes'}</span>
              </button>
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
                      onClick={() => setActiveTab(tab.id as any)}
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

            {/* Settings Status */}
            <div className="mt-6 bg-white rounded-lg shadow-sm border p-4">
              <h3 className="font-semibold text-gray-900 mb-3">Settings Status</h3>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  {hasChanges ? (
                    <AlertTriangle className="w-4 h-4 text-yellow-500" />
                  ) : (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  )}
                  <span className="text-sm text-gray-600">
                    {hasChanges ? 'Unsaved changes' : 'All changes saved'}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <Info className="w-4 h-4 text-blue-500" />
                  <span className="text-sm text-gray-600">
                    {Object.keys(settings).length} settings configured
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
            >
              {renderTabContent()}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel;
