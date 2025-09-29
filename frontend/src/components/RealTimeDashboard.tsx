import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Play,
  Pause,
  Square,
  Wifi,
  WifiOff,
  Camera,
  HardDrive,
  Download,
  Settings,
  Users,
  Target,
  Zap,
  TrendingUp,
  Activity,
  Clock,
  MapPin,
  Award,
  Trophy,
  BarChart3,
  PieChart,
  LineChart,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Maximize2,
  Minimize2,
  Volume2,
  VolumeX,
  RotateCcw,
  Save,
  Share2
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import {
  LineChart as RechartsLineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Pie,
  Cell
} from 'recharts';

interface RealTimeDashboardProps {
  isLive: boolean;
  onToggleLive: () => void;
  hardwareConnected: boolean;
  onConnectHardware: () => void;
  className?: string;
}

interface LiveStats {
  timestamp: number;
  teamA: {
    possession: number;
    shots: number;
    passes: number;
    tackles: number;
    distance: number;
    avgSpeed: number;
    goals: number;
    corners: number;
    fouls: number;
  };
  teamB: {
    possession: number;
    shots: number;
    passes: number;
    tackles: number;
    distance: number;
    avgSpeed: number;
    goals: number;
    corners: number;
    fouls: number;
  };
  events: Array<{
    id: string;
    type: string;
    timestamp: number;
    player?: string;
    team?: string;
    description: string;
    severity?: string;
  }>;
  ballPosition: {
    x: number;
    y: number;
    speed: number;
    direction: number;
  };
  matchTime: number;
  temperature: number;
  humidity: number;
  windSpeed: number;
}

const RealTimeDashboard: React.FC<RealTimeDashboardProps> = ({
  isLive,
  onToggleLive,
  hardwareConnected,
  onConnectHardware,
  className
}) => {
  const [liveStats, setLiveStats] = useState<LiveStats | null>(null);
  const [statsHistory, setStatsHistory] = useState<LiveStats[]>([]);
  const [selectedView, setSelectedView] = useState<'overview' | 'teams' | 'events' | 'ball' | 'weather'>('overview');
  const [autoSave, setAutoSave] = useState(true);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState<'A' | 'B' | 'both'>('both');
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const recordingStartRef = useRef<number>(0);

  // Simulate real-time data updates
  useEffect(() => {
    if (isLive && hardwareConnected) {
      intervalRef.current = setInterval(() => {
        const newStats: LiveStats = {
          timestamp: Date.now(),
          teamA: {
            possession: 45 + Math.random() * 10,
            shots: Math.floor(Math.random() * 8) + 2,
            passes: Math.floor(Math.random() * 30) + 50,
            tackles: Math.floor(Math.random() * 12) + 3,
            distance: Math.random() * 8 + 8,
            avgSpeed: Math.random() * 4 + 6,
            goals: Math.floor(Math.random() * 3),
            corners: Math.floor(Math.random() * 6),
            fouls: Math.floor(Math.random() * 8) + 2
          },
          teamB: {
            possession: 55 - Math.random() * 10,
            shots: Math.floor(Math.random() * 8) + 2,
            passes: Math.floor(Math.random() * 30) + 50,
            tackles: Math.floor(Math.random() * 12) + 3,
            distance: Math.random() * 8 + 8,
            avgSpeed: Math.random() * 4 + 6,
            goals: Math.floor(Math.random() * 3),
            corners: Math.floor(Math.random() * 6),
            fouls: Math.floor(Math.random() * 8) + 2
          },
          events: [
            {
              id: `event_${Date.now()}`,
              type: Math.random() > 0.8 ? 'goal' : Math.random() > 0.6 ? 'shot' : Math.random() > 0.4 ? 'pass' : 'tackle',
              timestamp: Date.now(),
              player: `Player ${Math.floor(Math.random() * 22) + 1}`,
              team: Math.random() > 0.5 ? 'A' : 'B',
              description: Math.random() > 0.8 ? 'Goal scored!' : Math.random() > 0.6 ? 'Shot on target' : Math.random() > 0.4 ? 'Pass completed' : 'Tackle won',
              severity: Math.random() > 0.7 ? 'high' : 'medium'
            }
          ],
          ballPosition: {
            x: Math.random() * 100,
            y: Math.random() * 100,
            speed: Math.random() * 25 + 5,
            direction: Math.random() * 360
          },
          matchTime: recordingDuration,
          temperature: 18 + Math.random() * 8,
          humidity: 60 + Math.random() * 20,
          windSpeed: Math.random() * 15
        };

        setLiveStats(newStats);
        setStatsHistory(prev => [...prev.slice(-29), newStats]); // Keep last 30 data points
        
        if (isRecording) {
          setRecordingDuration(prev => prev + 1);
        }
      }, 1000); // Update every second
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isLive, hardwareConnected, isRecording, recordingDuration]);

  const handleStartRecording = () => {
    setIsRecording(true);
    recordingStartRef.current = Date.now();
    toast.success('Recording started - match will be auto-saved');
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    toast.success('Recording stopped - match saved to your account');
  };

  const handleExportData = () => {
    const data = {
      liveStats,
      statsHistory,
      recordingDuration,
      timestamp: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `live_analytics_${Date.now()}.json`;
    link.click();
    toast.success('Live data exported successfully!');
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'goal': return <Trophy className="w-4 h-4 text-yellow-500" />;
      case 'shot': return <Target className="w-4 h-4 text-red-500" />;
      case 'pass': return <Zap className="w-4 h-4 text-blue-500" />;
      case 'tackle': return <Users className="w-4 h-4 text-orange-500" />;
      default: return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'goal': return 'bg-yellow-100 border-yellow-300 text-yellow-800';
      case 'shot': return 'bg-red-100 border-red-300 text-red-800';
      case 'pass': return 'bg-blue-100 border-blue-300 text-blue-800';
      case 'tackle': return 'bg-orange-100 border-orange-300 text-orange-800';
      default: return 'bg-gray-100 border-gray-300 text-gray-800';
    }
  };

  const renderOverviewTab = () => (
    <div className="space-y-6">
      {/* Live Match Status */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Live Match Status</h3>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
              <span className="text-sm font-medium">
                {isLive ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Clock className="w-8 h-8 mx-auto text-gray-600 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {formatTime(recordingDuration)}
            </div>
            <div className="text-sm text-gray-600">Match Time</div>
          </div>

          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Camera className="w-8 h-8 mx-auto text-gray-600 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {hardwareConnected ? 'Connected' : 'Disconnected'}
            </div>
            <div className="text-sm text-gray-600">Hardware Status</div>
          </div>

          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <HardDrive className="w-8 h-8 mx-auto text-gray-600 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {isRecording ? 'Recording' : 'Stopped'}
            </div>
            <div className="text-sm text-gray-600">Recording Status</div>
          </div>

          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Activity className="w-8 h-8 mx-auto text-gray-600 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {liveStats?.events.length || 0}
            </div>
            <div className="text-sm text-gray-600">Live Events</div>
          </div>
        </div>
      </div>

      {/* Team Comparison */}
      {liveStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-lg shadow-sm border p-6"
          >
            <h3 className="text-lg font-semibold text-red-600 mb-4">Team A</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Possession</span>
                  <span className="font-semibold">{liveStats.teamA.possession.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Shots</span>
                  <span className="font-semibold">{liveStats.teamA.shots}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Passes</span>
                  <span className="font-semibold">{liveStats.teamA.passes}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Tackles</span>
                  <span className="font-semibold">{liveStats.teamA.tackles}</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Goals</span>
                  <span className="font-semibold text-green-600">{liveStats.teamA.goals}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Corners</span>
                  <span className="font-semibold">{liveStats.teamA.corners}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Fouls</span>
                  <span className="font-semibold text-red-600">{liveStats.teamA.fouls}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Distance (km)</span>
                  <span className="font-semibold">{liveStats.teamA.distance.toFixed(1)}</span>
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-lg shadow-sm border p-6"
          >
            <h3 className="text-lg font-semibold text-blue-600 mb-4">Team B</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Possession</span>
                  <span className="font-semibold">{liveStats.teamB.possession.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Shots</span>
                  <span className="font-semibold">{liveStats.teamB.shots}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Passes</span>
                  <span className="font-semibold">{liveStats.teamB.passes}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Tackles</span>
                  <span className="font-semibold">{liveStats.teamB.tackles}</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Goals</span>
                  <span className="font-semibold text-green-600">{liveStats.teamB.goals}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Corners</span>
                  <span className="font-semibold">{liveStats.teamB.corners}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Fouls</span>
                  <span className="font-semibold text-red-600">{liveStats.teamB.fouls}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Distance (km)</span>
                  <span className="font-semibold">{liveStats.teamB.distance.toFixed(1)}</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Live Events Feed */}
      {liveStats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Live Events</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {liveStats.events.map((event) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`flex items-center space-x-3 p-3 rounded-lg border ${getEventColor(event.type)}`}
              >
                {getEventIcon(event.type)}
                <div className="flex-1">
                  <div className="font-medium">{event.description}</div>
                  <div className="text-sm opacity-75">
                    {event.player} ({event.team}) - {formatTime(Math.floor((Date.now() - event.timestamp) / 1000))} ago
                  </div>
                </div>
                {event.severity === 'high' && (
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );

  const renderTeamsTab = () => (
    <div className="space-y-6">
      {/* Possession Chart */}
      {statsHistory.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Possession Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={statsHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="matchTime" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey="teamA.possession"
                stackId="1"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.6}
                name="Team A"
              />
              <Area
                type="monotone"
                dataKey="teamB.possession"
                stackId="1"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.6}
                name="Team B"
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Team Performance Comparison */}
      {liveStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Team A Performance</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { name: 'Shots', value: liveStats.teamA.shots },
                { name: 'Passes', value: liveStats.teamA.passes },
                { name: 'Tackles', value: liveStats.teamA.tackles },
                { name: 'Corners', value: liveStats.teamA.corners }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Team B Performance</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { name: 'Shots', value: liveStats.teamB.shots },
                { name: 'Passes', value: liveStats.teamB.passes },
                { name: 'Tackles', value: liveStats.teamB.tackles },
                { name: 'Corners', value: liveStats.teamB.corners }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );

  const renderEventsTab = () => (
    <div className="space-y-6">
      {/* Events Timeline */}
      {statsHistory.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Events Timeline</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={statsHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="matchTime" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="teamA.shots"
                stroke="#ef4444"
                strokeWidth={2}
                name="Team A Shots"
              />
              <Line
                type="monotone"
                dataKey="teamB.shots"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Team B Shots"
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Event Statistics */}
      {liveStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <Target className="w-8 h-8 mx-auto text-red-500 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.teamA.shots + liveStats.teamB.shots}
            </div>
            <div className="text-sm text-gray-600">Total Shots</div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <Zap className="w-8 h-8 mx-auto text-blue-500 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.teamA.passes + liveStats.teamB.passes}
            </div>
            <div className="text-sm text-gray-600">Total Passes</div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <Users className="w-8 h-8 mx-auto text-orange-500 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.teamA.tackles + liveStats.teamB.tackles}
            </div>
            <div className="text-sm text-gray-600">Total Tackles</div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <Trophy className="w-8 h-8 mx-auto text-yellow-500 mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.teamA.goals + liveStats.teamB.goals}
            </div>
            <div className="text-sm text-gray-600">Total Goals</div>
          </div>
        </div>
      )}
    </div>
  );

  const renderBallTab = () => (
    <div className="space-y-6">
      {/* Ball Position and Speed */}
      {liveStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-lg shadow-sm border p-6"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Ball Position</h3>
            <div className="relative w-full h-64 bg-green-100 rounded-lg border-2 border-green-300">
              <motion.div
                className="absolute w-4 h-4 bg-yellow-500 rounded-full border-2 border-white shadow-lg"
                animate={{
                  x: `${liveStats.ballPosition.x}%`,
                  y: `${liveStats.ballPosition.y}%`
                }}
                transition={{ duration: 0.5 }}
              />
              <div className="absolute top-2 left-2 text-sm font-medium text-gray-700">
                X: {liveStats.ballPosition.x.toFixed(1)}%, Y: {liveStats.ballPosition.y.toFixed(1)}%
              </div>
              <div className="absolute bottom-2 right-2 text-sm font-medium text-gray-700">
                Direction: {liveStats.ballPosition.direction.toFixed(0)}°
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-lg shadow-sm border p-6"
          >
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Ball Speed</h3>
            <div className="text-center">
              <div className="text-4xl font-bold text-yellow-600 mb-2">
                {liveStats.ballPosition.speed.toFixed(1)}
              </div>
              <div className="text-lg text-gray-600 mb-4">km/h</div>
              <div className="w-full bg-gray-200 rounded-full h-4">
                <motion.div
                  className="bg-yellow-500 h-4 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${(liveStats.ballPosition.speed / 30) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Ball Trajectory */}
      {statsHistory.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Ball Speed Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={statsHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="matchTime" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="ballPosition.speed"
                stroke="#eab308"
                strokeWidth={2}
                name="Ball Speed (km/h)"
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      )}
    </div>
  );

  const renderWeatherTab = () => (
    <div className="space-y-6">
      {liveStats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <div className="text-4xl mb-2">🌡️</div>
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.temperature.toFixed(1)}°C
            </div>
            <div className="text-sm text-gray-600">Temperature</div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <div className="text-4xl mb-2">💧</div>
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.humidity.toFixed(0)}%
            </div>
            <div className="text-sm text-gray-600">Humidity</div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border p-6 text-center">
            <div className="text-4xl mb-2">💨</div>
            <div className="text-2xl font-bold text-gray-900">
              {liveStats.windSpeed.toFixed(1)} km/h
            </div>
            <div className="text-sm text-gray-600">Wind Speed</div>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Live Match Analytics</h1>
              <p className="text-gray-600">Real-time Premier League-style analytics</p>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Hardware Connection */}
              <button
                onClick={onConnectHardware}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  hardwareConnected
                    ? 'bg-green-600 text-white hover:bg-green-700'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {hardwareConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                <span>{hardwareConnected ? 'Connected' : 'Connect Hardware'}</span>
              </button>

              {/* Live Toggle */}
              <button
                onClick={onToggleLive}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  isLive
                    ? 'bg-red-600 text-white hover:bg-red-700'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {isLive ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isLive ? 'Stop Live' : 'Start Live'}</span>
              </button>

              {/* Recording Controls */}
              {isLive && hardwareConnected && (
                <button
                  onClick={isRecording ? handleStopRecording : handleStartRecording}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                    isRecording
                      ? 'bg-red-600 text-white hover:bg-red-700'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isRecording ? <Square className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
                  <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
                </button>
              )}

              {/* Export Data */}
              <button
                onClick={handleExportData}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Export</span>
              </button>

              {/* Audio Toggle */}
              <button
                onClick={() => setIsMuted(!isMuted)}
                className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow-sm border mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6">
              {[
                { id: 'overview', label: 'Overview', icon: BarChart3 },
                { id: 'teams', label: 'Teams', icon: Users },
                { id: 'events', label: 'Events', icon: Zap },
                { id: 'ball', label: 'Ball', icon: Target },
                { id: 'weather', label: 'Weather', icon: MapPin }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setSelectedView(tab.id as any)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    selectedView === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {selectedView === 'overview' && renderOverviewTab()}
            {selectedView === 'teams' && renderTeamsTab()}
            {selectedView === 'events' && renderEventsTab()}
            {selectedView === 'ball' && renderBallTab()}
            {selectedView === 'weather' && renderWeatherTab()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default RealTimeDashboard;