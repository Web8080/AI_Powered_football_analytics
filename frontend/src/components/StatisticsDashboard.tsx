import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Users,
  Target,
  Zap,
  TrendingUp,
  Download,
  Share2,
  Filter,
  Search,
  Calendar,
  Clock,
  Award,
  Trophy,
  Activity,
  MapPin,
  PieChart,
  LineChart,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  RefreshCw,
  Save,
  FileText,
  Image,
  Video,
  Database,
  AlertTriangle
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
  Cell,
  ScatterChart,
  Scatter
} from 'recharts';

interface StatisticsDashboardProps {
  analysisResults: any;
  uploadedVideo: any;
  className?: string;
}

const StatisticsDashboard: React.FC<StatisticsDashboardProps> = ({
  analysisResults,
  uploadedVideo,
  className
}) => {
  const [selectedView, setSelectedView] = useState<'overview' | 'players' | 'teams' | 'events' | 'heatmap' | 'export'>('overview');
  const [selectedTimeframe, setSelectedTimeframe] = useState<'all' | 'first_half' | 'second_half' | 'custom'>('all');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [filters, setFilters] = useState({
    team: 'both',
    eventType: 'all',
    player: 'all',
    timeRange: [0, 100]
  });

  // Generate mock data if no real data
  const generateMockData = () => {
    if (analysisResults && analysisResults.statistics && analysisResults.statistics.possession) {
      return analysisResults;
    }
    
    return {
      detection: {
        total_players: 22,
        team_a_players: 11,
        team_b_players: 11,
        ball_detections: 150,
        referee_detections: 25
      },
      tracking: {
        ball_trajectory: Array.from({ length: 100 }, (_, i) => ({
          x: Math.random() * 100,
          y: Math.random() * 100,
          timestamp: i * 1000,
          speed: Math.random() * 20 + 5
        })),
        player_tracks: 22
      },
      events: [
        { type: 'goal', timestamp: 1200, player: 'Player 7', team: 'A', x: 85, y: 45 },
        { type: 'shot', timestamp: 800, player: 'Player 12', team: 'B', x: 75, y: 50 },
        { type: 'pass', timestamp: 600, player: 'Player 3', team: 'A', x: 60, y: 30 },
        { type: 'tackle', timestamp: 400, player: 'Player 8', team: 'B', x: 40, y: 60 },
        { type: 'corner', timestamp: 1000, player: 'Player 5', team: 'A', x: 90, y: 20 },
        { type: 'foul', timestamp: 300, player: 'Player 15', team: 'B', x: 35, y: 70 }
      ],
      statistics: {
        possession: { team_a: 45, team_b: 55 },
        shots: { team_a: 8, team_b: 12 },
        passes: { team_a: 156, team_b: 142 },
        tackles: { team_a: 23, team_b: 19 },
        corners: { team_a: 4, team_b: 6 },
        fouls: { team_a: 12, team_b: 15 }
      },
      playerStats: Array.from({ length: 22 }, (_, i) => ({
        id: i + 1,
        name: `Player ${i + 1}`,
        team: i < 11 ? 'A' : 'B',
        position: ['GK', 'DEF', 'MID', 'FWD'][Math.floor(Math.random() * 4)],
        distance: Math.random() * 10 + 5,
        speed: Math.random() * 5 + 15,
        passes: Math.floor(Math.random() * 50) + 10,
        shots: Math.floor(Math.random() * 5),
        tackles: Math.floor(Math.random() * 8) + 2,
        heatmap: Array.from({ length: 20 }, () => ({
          x: Math.random() * 100,
          y: Math.random() * 100,
          intensity: Math.random()
        }))
      }))
    };
  };

  const data = generateMockData();
  
  // Safety check to prevent undefined errors
  if (!data || !data.statistics || !data.statistics.possession) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertTriangle className="w-8 h-8 text-red-600" />
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No Analysis Data</h2>
          <p className="text-gray-600">Please upload a video and run analysis first.</p>
        </div>
      </div>
    );
  }

  // Ensure arrays exist to prevent .map() errors
  const safePlayerStats = data.playerStats || [];
  const safeEvents = data.events || [];
  const safeBallTrajectory = data.tracking?.ball_trajectory || [];

  const handleExportData = (format: 'json' | 'csv' | 'pdf') => {
    const exportData = {
      ...data,
      exportDate: new Date().toISOString(),
      videoFile: uploadedVideo || 'demo_video.mp4'
    };

    if (format === 'json') {
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `godseye_analysis_${Date.now()}.json`;
      link.click();
    } else if (format === 'csv') {
      // Convert to CSV format
      const csvData = [
        ['Metric', 'Team A', 'Team B'],
        ['Possession', data.statistics.possession.team_a, data.statistics.possession.team_b],
        ['Shots', data.statistics.shots.team_a, data.statistics.shots.team_b],
        ['Passes', data.statistics.passes.team_a, data.statistics.passes.team_b],
        ['Tackles', data.statistics.tackles.team_a, data.statistics.tackles.team_b]
      ].map(row => row.join(',')).join('\n');
      
      const blob = new Blob([csvData], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `godseye_analysis_${Date.now()}.csv`;
      link.click();
    }

    toast.success(`Data exported as ${format.toUpperCase()} successfully!`);
  };

  const renderOverviewTab = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Players</p>
              <p className="text-2xl font-bold text-gray-900">{data.detection.total_players}</p>
            </div>
            <Users className="w-8 h-8 text-blue-600" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Events Detected</p>
              <p className="text-2xl font-bold text-gray-900">{data.events.length}</p>
            </div>
            <Zap className="w-8 h-8 text-yellow-600" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Ball Positions</p>
              <p className="text-2xl font-bold text-gray-900">{data.tracking.ball_trajectory.length}</p>
            </div>
            <Target className="w-8 h-8 text-green-600" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Match Duration</p>
              <p className="text-2xl font-bold text-gray-900">90:00</p>
            </div>
            <Clock className="w-8 h-8 text-purple-600" />
          </div>
        </motion.div>
      </div>

      {/* Team Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Team A Statistics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Possession</span>
              <span className="font-semibold text-lg">{data.statistics.possession.team_a}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Shots</span>
              <span className="font-semibold text-lg">{data.statistics.shots.team_a}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Passes</span>
              <span className="font-semibold text-lg">{data.statistics.passes.team_a}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Tackles</span>
              <span className="font-semibold text-lg">{data.statistics.tackles.team_a}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Corners</span>
              <span className="font-semibold text-lg">{data.statistics.corners.team_a}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Fouls</span>
              <span className="font-semibold text-lg text-red-600">{data.statistics.fouls.team_a}</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-white rounded-lg shadow-sm border p-6"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Team B Statistics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Possession</span>
              <span className="font-semibold text-lg">{data.statistics.possession.team_b}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Shots</span>
              <span className="font-semibold text-lg">{data.statistics.shots.team_b}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Passes</span>
              <span className="font-semibold text-lg">{data.statistics.passes.team_b}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Tackles</span>
              <span className="font-semibold text-lg">{data.statistics.tackles.team_b}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Corners</span>
              <span className="font-semibold text-lg">{data.statistics.corners.team_b}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Fouls</span>
              <span className="font-semibold text-lg text-red-600">{data.statistics.fouls.team_b}</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Possession Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Possession Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={[
                { name: 'Team A', value: data.statistics.possession.team_a, color: '#ef4444' },
                { name: 'Team B', value: data.statistics.possession.team_b, color: '#3b82f6' }
              ]}
              cx="50%"
              cy="50%"
              outerRadius={100}
              dataKey="value"
              label={({ name, value }) => `${name}: ${value}%`}
            >
              {[
                { name: 'Team A', value: data.statistics.possession.team_a, color: '#ef4444' },
                { name: 'Team B', value: data.statistics.possession.team_b, color: '#3b82f6' }
              ].map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );

  const renderPlayersTab = () => (
    <div className="space-y-6">
      {/* Player Performance Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Player Performance</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data.playerStats.slice(0, 10)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="distance" fill="#3b82f6" name="Distance (km)" />
            <Bar dataKey="passes" fill="#10b981" name="Passes" />
            <Bar dataKey="tackles" fill="#f59e0b" name="Tackles" />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Player List */}
      <div className="bg-white rounded-lg shadow-sm border">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold text-gray-900">Player Statistics</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Player</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Jersey</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Team</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Position</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Distance</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Speed</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Passes</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Shots</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tackles</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {safePlayerStats.map((player, index) => (
                <motion.tr
                  key={player.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="hover:bg-gray-50"
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {player.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800">
                      #{player.jersey_number || 0}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`px-2 py-1 text-xs rounded-full ${
                      player.team === 'A' ? 'bg-red-100 text-red-800' : 
                      player.team === 'B' ? 'bg-blue-100 text-blue-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {player.team === 'REF' ? 'Referee' : `Team ${player.team}`}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {player.position}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {player.distance.toFixed(1)} km
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {player.speed.toFixed(1)} km/h
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {player.passes}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {player.shots}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {player.tackles}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderEventsTab = () => (
    <div className="space-y-6">
      {/* Events Timeline */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Events Timeline</h3>
        <div className="space-y-4">
          {safeEvents.map((event, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg"
            >
              <div className="flex-shrink-0">
                {event.type === 'goal' && <Trophy className="w-6 h-6 text-yellow-500" />}
                {event.type === 'shot' && <Target className="w-6 h-6 text-red-500" />}
                {event.type === 'pass' && <Zap className="w-6 h-6 text-blue-500" />}
                {event.type === 'tackle' && <Users className="w-6 h-6 text-orange-500" />}
                {event.type === 'corner' && <MapPin className="w-6 h-6 text-green-500" />}
                {event.type === 'foul' && <AlertTriangle className="w-6 h-6 text-red-600" />}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-gray-900 capitalize">{event.type}</h4>
                  <span className="text-sm text-gray-500">
                    {Math.floor(event.timestamp / 60)}:{(event.timestamp % 60).toString().padStart(2, '0')}
                  </span>
                </div>
                <p className="text-sm text-gray-600">
                  {event.player} (Team {event.team}) at position ({event.x}, {event.y})
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Event Distribution */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Event Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={[
            { name: 'Goals', value: data.events.filter(e => e.type === 'goal').length },
            { name: 'Shots', value: data.events.filter(e => e.type === 'shot').length },
            { name: 'Passes', value: data.events.filter(e => e.type === 'pass').length },
            { name: 'Tackles', value: data.events.filter(e => e.type === 'tackle').length },
            { name: 'Corners', value: data.events.filter(e => e.type === 'corner').length },
            { name: 'Fouls', value: data.events.filter(e => e.type === 'foul').length }
          ]}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );

  const renderHeatmapTab = () => (
    <div className="space-y-6">
      {/* Field Heatmap */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Field Heatmap</h3>
        <div className="relative w-full h-96 bg-green-100 rounded-lg border-2 border-green-300">
          {/* Field lines */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-full h-full border border-white opacity-30"></div>
          </div>
          
          {/* Heatmap points */}
          {safeBallTrajectory.map((point, index) => (
            <motion.div
              key={index}
              className="absolute w-2 h-2 bg-red-500 rounded-full opacity-60"
              style={{
                left: `${point.x}%`,
                top: `${point.y}%`,
              }}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 0.6, scale: 1 }}
              transition={{ delay: index * 0.01 }}
            />
          ))}
          
          {/* Event markers */}
          {safeEvents.map((event, index) => (
            <motion.div
              key={index}
              className="absolute w-4 h-4 rounded-full border-2 border-white"
              style={{
                left: `${event.x}%`,
                top: `${event.y}%`,
                backgroundColor: event.type === 'goal' ? '#fbbf24' : 
                               event.type === 'shot' ? '#ef4444' : 
                               event.type === 'pass' ? '#3b82f6' : '#10b981'
              }}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
            />
          ))}
        </div>
      </motion.div>

      {/* Player Heatmaps */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {safePlayerStats.slice(0, 6).map((player, index) => (
          <motion.div
            key={player.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="bg-white rounded-lg shadow-sm border p-4"
          >
            <h4 className="font-medium text-gray-900 mb-2">{player.name}</h4>
            <div className="relative w-full h-32 bg-green-100 rounded border">
              {(player.heatmap || []).map((point, pointIndex) => (
                <div
                  key={pointIndex}
                  className="absolute w-1 h-1 bg-red-500 rounded-full"
                  style={{
                    left: `${point.x}%`,
                    top: `${point.y}%`,
                    opacity: point.intensity
                  }}
                />
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );

  const renderExportTab = () => (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Export Analysis Data</h3>
        <p className="text-gray-600 mb-6">
          Download your analysis results in various formats for further analysis or reporting.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => handleExportData('json')}
            className="flex items-center space-x-3 p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <Database className="w-8 h-8 text-blue-600" />
            <div className="text-left">
              <div className="font-medium text-gray-900">JSON Format</div>
              <div className="text-sm text-gray-600">Complete raw data</div>
            </div>
          </button>
          
          <button
            onClick={() => handleExportData('csv')}
            className="flex items-center space-x-3 p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <FileText className="w-8 h-8 text-green-600" />
            <div className="text-left">
              <div className="font-medium text-gray-900">CSV Format</div>
              <div className="text-sm text-gray-600">Spreadsheet compatible</div>
            </div>
          </button>
          
          <button
            onClick={() => toast.info('PDF export coming soon!')}
            className="flex items-center space-x-3 p-4 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <FileText className="w-8 h-8 text-red-600" />
            <div className="text-left">
              <div className="font-medium text-gray-900">PDF Report</div>
              <div className="text-sm text-gray-600">Professional report</div>
            </div>
          </button>
        </div>
      </motion.div>

      {/* Data Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border p-6"
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">{data.detection.total_players}</div>
            <div className="text-sm text-gray-600">Players Tracked</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">{data.events.length}</div>
            <div className="text-sm text-gray-600">Events Detected</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">{data.tracking.ball_trajectory.length}</div>
            <div className="text-sm text-gray-600">Ball Positions</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">90:00</div>
            <div className="text-sm text-gray-600">Match Duration</div>
          </div>
        </div>
      </motion.div>
    </div>
  );

  return (
    <div className={`min-h-screen bg-gray-50 ${className}`}>
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Match Analytics</h1>
              <p className="text-gray-600">Detailed analysis and statistics</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                {showAdvanced ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                <span>{showAdvanced ? 'Hide' : 'Show'} Advanced</span>
              </button>
              
              <button
                onClick={() => setIsFullscreen(!isFullscreen)}
                className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
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
                { id: 'players', label: 'Players', icon: Users },
                { id: 'events', label: 'Events', icon: Zap },
                { id: 'heatmap', label: 'Heatmap', icon: MapPin },
                { id: 'export', label: 'Export', icon: Download }
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
            {selectedView === 'players' && renderPlayersTab()}
            {selectedView === 'events' && renderEventsTab()}
            {selectedView === 'heatmap' && renderHeatmapTab()}
            {selectedView === 'export' && renderExportTab()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};

export default StatisticsDashboard;