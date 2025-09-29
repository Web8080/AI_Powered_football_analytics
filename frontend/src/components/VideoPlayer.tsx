import React, { useRef, useEffect, useState } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize, RotateCcw } from 'lucide-react';

interface VideoPlayerProps {
  videoUrl: string;
  events: any[];
  onTimeUpdate?: (currentTime: number) => void;
  className?: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ 
  videoUrl, 
  events, 
  onTimeUpdate, 
  className = "" 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showGoalNotification, setShowGoalNotification] = useState(false);
  const [goalEvent, setGoalEvent] = useState<any>(null);

  // Check for goal events at current time
  useEffect(() => {
    if (!events || events.length === 0) return;

    const currentEvent = events.find(event => 
      Math.abs(event.timestamp - currentTime) < 2 && event.type === 'goal'
    );

    if (currentEvent && currentEvent !== goalEvent) {
      setGoalEvent(currentEvent);
      setShowGoalNotification(true);
      
      // Hide notification after 5 seconds
      setTimeout(() => {
        setShowGoalNotification(false);
      }, 5000);
    }
  }, [currentTime, events, goalEvent]);

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (videoRef.current) {
      const newTime = parseFloat(e.target.value);
      videoRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    }
  };

  const handleRestart = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      setCurrentTime(0);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`relative bg-black rounded-lg overflow-hidden ${className}`}>
      {/* Video Element */}
      <video
        ref={videoRef}
        src={videoUrl}
        className="w-full h-full"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Goal Notification */}
      {showGoalNotification && goalEvent && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50">
          <div className="bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg animate-bounce">
            <div className="text-center">
              <div className="text-2xl font-bold">⚽ GOAL!</div>
              <div className="text-sm">
                {goalEvent.team} - {goalEvent.player}
              </div>
              <div className="text-xs opacity-75">
                {formatTime(goalEvent.timestamp)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Controls Overlay */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
        {/* Progress Bar */}
        <div className="mb-4">
          <input
            type="range"
            min="0"
            max={duration || 0}
            value={currentTime}
            onChange={handleSeek}
            className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
          />
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={handlePlayPause}
              className="text-white hover:text-gray-300 transition-colors"
            >
              {isPlaying ? <Pause size={24} /> : <Play size={24} />}
            </button>

            <button
              onClick={handleMute}
              className="text-white hover:text-gray-300 transition-colors"
            >
              {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
            </button>

            <button
              onClick={handleRestart}
              className="text-white hover:text-gray-300 transition-colors"
            >
              <RotateCcw size={20} />
            </button>

            <span className="text-white text-sm">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleFullscreen}
              className="text-white hover:text-gray-300 transition-colors"
            >
              <Maximize size={20} />
            </button>
          </div>
        </div>
      </div>

      {/* Event Timeline */}
      <div className="absolute top-4 right-4 bg-black/70 text-white p-3 rounded-lg max-w-xs">
        <h4 className="text-sm font-semibold mb-2">Events Timeline</h4>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {events
            .filter(event => event.timestamp <= currentTime + 10)
            .slice(-5)
            .map((event, index) => (
              <div
                key={index}
                className={`text-xs p-1 rounded ${
                  event.type === 'goal' 
                    ? 'bg-green-600' 
                    : event.type === 'shot'
                    ? 'bg-yellow-600'
                    : 'bg-gray-600'
                }`}
              >
                <span className="font-medium">{event.type.toUpperCase()}</span>
                <span className="ml-2">{formatTime(event.timestamp)}</span>
                {event.team && (
                  <span className="ml-2">Team {event.team}</span>
                )}
              </div>
            ))}
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
        }
        
        .slider::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
};

export default VideoPlayer;
