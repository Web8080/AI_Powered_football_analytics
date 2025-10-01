import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  Clock, 
  Users, 
  Target, 
  Activity, 
  MapPin, 
  TrendingUp,
  AlertTriangle,
  Trophy,
  Zap,
  Eye,
  BarChart3,
  PieChart,
  HeatMap,
  Route,
  Timer,
  Calendar,
  Cloud,
  Camera,
  Star
} from 'lucide-react';

interface Player {
  id: number;
  name: string;
  team: 'A' | 'B';
  position: string;
  jerseyNumber: number;
  speed: number;
  distance: number;
  heatmap: { x: number; y: number; intensity: number }[];
  keypoints: { x: number; y: number; confidence: number }[];
  performance: {
    passes: number;
    shots: number;
    tackles: number;
    interceptions: number;
  };
}

interface Event {
  id: string;
  type: 'goal' | 'foul' | 'card' | 'substitution' | 'corner' | 'throw_in' | 'offside';
  timestamp: string;
  player?: string;
  team?: 'A' | 'B';
  description: string;
  confidence: number;
}

interface MatchData {
  teamA: {
    name: string;
    score: number;
    possession: number;
    shots: number;
    passes: number;
    formation: string;
  };
  teamB: {
    name: string;
    score: number;
    possession: number;
    shots: number;
    passes: number;
    formation: string;
  };
  matchTime: string;
  weather: string;
  venue: string;
  competition: string;
}

interface TacticalData {
  formation: string;
  pressingIntensity: number;
  fieldZones: { zone: string; activity: number }[];
  passNetworks: { from: number; to: number; count: number }[];
}

const EnhancedAnalyticsDashboard: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState('00:00');
  const [matchData, setMatchData] = useState<MatchData>({
    teamA: { name: 'Team A', score: 0, possession: 45, shots: 3, passes: 120, formation: '4-4-2' },
    teamB: { name: 'Team B', score: 0, possession: 55, shots: 5, passes: 150, formation: '4-3-3' },
    matchTime: '00:00',
    weather: 'Clear',
    venue: 'Stadium',
    competition: 'Premier League'
  });

  const [players, setPlayers] = useState<Player[]>([
    {
      id: 1, name: 'Player 1', team: 'A', position: 'GK', jerseyNumber: 1,
      speed: 0, distance: 0, heatmap: [], keypoints: [],
      performance: { passes: 0, shots: 0, tackles: 0, interceptions: 0 }
    }
  ]);

  const [events, setEvents] = useState<Event[]>([]);
  const [tacticalData, setTacticalData] = useState<TacticalData>({
    formation: '4-4-2',
    pressingIntensity: 0.7,
    fieldZones: [],
    passNetworks: []
  });

  const [activeTab, setActiveTab] = useState('overview');
  const [showEventAlert, setShowEventAlert] = useState(false);
  const [lastEvent, setLastEvent] = useState<Event | null>(null);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (isPlaying) {
        const [minutes, seconds] = currentTime.split(':').map(Number);
        const newSeconds = seconds + 1;
        const newMinutes = minutes + Math.floor(newSeconds / 60);
        const finalSeconds = newSeconds % 60;
        setCurrentTime(`${newMinutes.toString().padStart(2, '0')}:${finalSeconds.toString().padStart(2, '0')}`);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isPlaying, currentTime]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleStop = () => {
    setIsPlaying(false);
    setCurrentTime('00:00');
  };

  const simulateEvent = (eventType: Event['type']) => {
    const newEvent: Event = {
      id: Date.now().toString(),
      type: eventType,
      timestamp: currentTime,
      player: `Player ${Math.floor(Math.random() * 22) + 1}`,
      team: Math.random() > 0.5 ? 'A' : 'B',
      description: `${eventType} detected`,
      confidence: 0.95
    };
    
    setEvents(prev => [newEvent, ...prev]);
    setLastEvent(newEvent);
    setShowEventAlert(true);
    
    setTimeout(() => setShowEventAlert(false), 3000);
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'players', label: 'Players', icon: Users },
    { id: 'events', label: 'Events', icon: AlertTriangle },
    { id: 'tactical', label: 'Tactical', icon: MapPin },
    { id: 'pose', label: 'Pose Analysis', icon: Activity },
    { id: 'heatmaps', label: 'Heatmaps', icon: HeatMap },
    { id: 'trajectories', label: 'Trajectories', icon: Route },
    { id: 'settings', label: 'Settings', icon: Star }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header with Scoreboard */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-2xl font-bold">Godseye AI</div>
              <div className="text-sm text-gray-400">
                {matchData.competition} • {matchData.venue} • {matchData.weather}
              </div>
            </div>
            
            {/* Live Scoreboard */}
            <div className="flex items-center space-x-8">
              <div className="text-center">
                <div className="text-lg font-semibold">{matchData.teamA.name}</div>
                <div className="text-3xl font-bold text-blue-400">{matchData.teamA.score}</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold">{currentTime}</div>
                <div className="text-sm text-gray-400">Match Time</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-semibold">{matchData.teamB.name}</div>
                <div className="text-3xl font-bold text-red-400">{matchData.teamB.score}</div>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={handlePlayPause}
                className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <button
                onClick={handleStop}
                className="p-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors"
              >
                <Square size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Event Alert */}
      {showEventAlert && lastEvent && (
        <div className="fixed top-20 right-4 z-50 bg-yellow-600 text-white p-4 rounded-lg shadow-lg animate-pulse">
          <div className="flex items-center space-x-2">
            <AlertTriangle size={20} />
            <div>
              <div className="font-bold">{lastEvent.type.toUpperCase()}</div>
              <div className="text-sm">{lastEvent.description} at {lastEvent.timestamp}</div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4">
          <div className="flex space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  <Icon size={16} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Match Statistics */}
            <div className="lg:col-span-2 bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">Match Statistics</h3>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold mb-3">{matchData.teamA.name}</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Possession:</span>
                      <span className="text-blue-400">{matchData.teamA.possession}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Shots:</span>
                      <span>{matchData.teamA.shots}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Passes:</span>
                      <span>{matchData.teamA.passes}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Formation:</span>
                      <span>{matchData.teamA.formation}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="text-lg font-semibold mb-3">{matchData.teamB.name}</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Possession:</span>
                      <span className="text-red-400">{matchData.teamB.possession}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Shots:</span>
                      <span>{matchData.teamB.shots}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Passes:</span>
                      <span>{matchData.teamB.passes}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Formation:</span>
                      <span>{matchData.teamB.formation}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={() => simulateEvent('goal')}
                  className="w-full p-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
                >
                  Simulate Goal
                </button>
                <button
                  onClick={() => simulateEvent('foul')}
                  className="w-full p-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
                >
                  Simulate Foul
                </button>
                <button
                  onClick={() => simulateEvent('card')}
                  className="w-full p-3 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                >
                  Simulate Card
                </button>
                <button
                  onClick={() => simulateEvent('substitution')}
                  className="w-full p-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                >
                  Simulate Substitution
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'players' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Player Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {players.map((player) => (
                <div key={player.id} className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-semibold">{player.name}</div>
                    <div className={`px-2 py-1 rounded text-xs ${
                      player.team === 'A' ? 'bg-blue-600' : 'bg-red-600'
                    }`}>
                      {player.team}
                    </div>
                  </div>
                  <div className="text-sm text-gray-400 mb-2">
                    #{player.jerseyNumber} • {player.position}
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span>Speed:</span>
                      <span>{player.speed} km/h</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Distance:</span>
                      <span>{player.distance} m</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Passes:</span>
                      <span>{player.performance.passes}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'events' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Event Timeline</h3>
            <div className="space-y-3">
              {events.map((event) => (
                <div key={event.id} className="flex items-center space-x-4 p-3 bg-gray-700 rounded-lg">
                  <div className="text-sm font-mono text-gray-400">{event.timestamp}</div>
                  <div className={`px-2 py-1 rounded text-xs ${
                    event.type === 'goal' ? 'bg-green-600' :
                    event.type === 'foul' ? 'bg-yellow-600' :
                    event.type === 'card' ? 'bg-red-600' :
                    'bg-blue-600'
                  }`}>
                    {event.type.toUpperCase()}
                  </div>
                  <div className="flex-1">{event.description}</div>
                  <div className="text-sm text-gray-400">
                    {Math.round(event.confidence * 100)}% confidence
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'tactical' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">Formation Analysis</h3>
              <div className="text-center">
                <div className="text-3xl font-bold mb-2">{tacticalData.formation}</div>
                <div className="text-gray-400">Current Formation</div>
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold mb-4">Pressing Intensity</h3>
              <div className="text-center">
                <div className="text-3xl font-bold mb-2">
                  {Math.round(tacticalData.pressingIntensity * 100)}%
                </div>
                <div className="text-gray-400">High Pressing</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'pose' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Pose Estimation Analysis</h3>
            <div className="text-center text-gray-400">
              <Activity size={48} className="mx-auto mb-4" />
              <p>17-keypoint pose estimation data will be displayed here</p>
              <p className="text-sm mt-2">Including gait analysis, injury prevention, and performance tracking</p>
            </div>
          </div>
        )}

        {activeTab === 'heatmaps' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Player Heatmaps</h3>
            <div className="text-center text-gray-400">
              <HeatMap size={48} className="mx-auto mb-4" />
              <p>Player movement heatmaps will be displayed here</p>
              <p className="text-sm mt-2">Showing field coverage and movement patterns</p>
            </div>
          </div>
        )}

        {activeTab === 'trajectories' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Ball & Player Trajectories</h3>
            <div className="text-center text-gray-400">
              <Route size={48} className="mx-auto mb-4" />
              <p>Ball and player trajectory analysis will be displayed here</p>
              <p className="text-sm mt-2">Including speed, acceleration, and direction changes</p>
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Detection Confidence</label>
                <input type="range" min="0" max="1" step="0.1" className="w-full" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Event Sensitivity</label>
                <input type="range" min="0" max="1" step="0.1" className="w-full" />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Update Frequency</label>
                <select className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2">
                  <option>Real-time</option>
                  <option>1 second</option>
                  <option>5 seconds</option>
                  <option>10 seconds</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedAnalyticsDashboard;

