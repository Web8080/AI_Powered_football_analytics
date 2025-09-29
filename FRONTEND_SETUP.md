# 🎯 Godseye AI Frontend Setup Guide

## 🚀 Quick Start

### Prerequisites
- **Node.js 18+** (Download from [nodejs.org](https://nodejs.org/))
- **npm 9+** (comes with Node.js)
- **Backend running** on `http://localhost:8000`

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
# or use the startup script
../start_frontend.sh
```

### 3. Open in Browser
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **ML Inference**: http://localhost:8001

## 🎨 Features

### ✅ **Video Upload & Analysis**
- **Drag & Drop Interface**: Upload football videos (MP4, AVI, MOV, MKV, WebM)
- **Real-time Progress**: See analysis progress with live updates
- **Multiple Analysis Types**: Detection, tracking, pose estimation, event detection
- **Results Visualization**: View bounding boxes, player tracks, and statistics

### ✅ **Live Analytics Dashboard**
- **Real-time Match Stats**: Like Premier League live broadcasts
- **Team Comparison**: Possession, shots, passes, tackles, distance, speed
- **Live Events Feed**: Goals, shots, passes, tackles with timestamps
- **Ball Tracking**: Real-time ball position and speed
- **Interactive Charts**: Possession over time, team performance

### ✅ **Hardware Integration**
- **VeoCam Support**: Connect to VeoCam or similar hardware
- **Auto-recording**: Automatically save 90+ minute matches
- **Real-time Streaming**: Live analytics during matches
- **Multiple Deployment**: Jetson, Raspberry Pi, cloud support

### ✅ **Professional Analytics**
- **8-Class Detection**: Team A/B players, goalkeepers, referee, ball, other, staff
- **Advanced Tracking**: DeepSort, ByteTrack, StrongSORT algorithms
- **Pose Estimation**: Player movement analysis and fatigue detection
- **Event Detection**: Goals, fouls, cards, offsides, corners
- **Statistics**: Comprehensive KPIs and performance metrics

## 🛠️ Technical Stack

### **Frontend Technologies**
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **React Query** for data fetching
- **Recharts** for data visualization
- **React Player** for video playback
- **React Dropzone** for file uploads

### **UI Components**
- **Lucide React** for icons
- **Headless UI** for accessible components
- **React Hot Toast** for notifications
- **React Hook Form** for form handling

### **Development Tools**
- **ESLint** for code linting
- **TypeScript** for type safety
- **Vitest** for testing
- **PostCSS** for CSS processing

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── MainDashboard.tsx # Main dashboard
│   │   ├── VideoUpload.tsx   # Video upload interface
│   │   ├── RealTimeDashboard.tsx # Live analytics
│   │   └── StatisticsDashboard.tsx # Results display
│   ├── services/            # API services
│   ├── utils/               # Utility functions
│   ├── types/               # TypeScript types
│   ├── hooks/               # Custom React hooks
│   ├── assets/              # Static assets
│   ├── App.tsx              # Main app component
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static files
├── package.json             # Dependencies
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind configuration
└── tsconfig.json            # TypeScript configuration
```

## 🎯 Usage Guide

### **1. Upload Video**
1. Go to **Upload Video** tab
2. Drag & drop your football video or click to select
3. Wait for upload to complete
4. Click **Start AI Analysis**
5. Monitor progress in real-time

### **2. Live Analytics**
1. Connect hardware (VeoCam, Jetson, etc.)
2. Go to **Live Analytics** tab
3. Click **Start Live** to begin real-time analysis
4. Click **Start Recording** to save the match
5. View live statistics, events, and ball tracking

### **3. View Results**
1. Go to **Analytics** tab after analysis completes
2. View comprehensive statistics and visualizations
3. Download results as JSON
4. Export annotated video

### **4. Configure Settings**
1. Go to **Settings** tab
2. Adjust detection model (YOLOv8n/s/m/l/x)
3. Select tracking algorithm (DeepSort/ByteTrack/StrongSORT)
4. Configure confidence thresholds
5. Set recording preferences

## 🔧 Configuration

### **Environment Variables**
Create `.env.local` in the frontend directory:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_INFERENCE_BASE_URL=http://localhost:8001
VITE_APP_NAME=Godseye AI
VITE_APP_VERSION=1.0.0
```

### **API Endpoints**
The frontend expects these backend endpoints:
- `POST /api/videos/upload` - Upload video
- `POST /api/analysis/start` - Start analysis
- `GET /api/analysis/{id}/status` - Get analysis status
- `GET /api/analysis/{id}/results` - Get analysis results
- `POST /api/hardware/connect` - Connect hardware
- `POST /api/live/start` - Start live analytics

## 🚀 Deployment

### **Development**
```bash
npm run dev
```

### **Production Build**
```bash
npm run build
npm run preview
```

### **Docker Deployment**
```bash
# Build Docker image
docker build -t godseye-frontend .

# Run container
docker run -p 3000:3000 godseye-frontend
```

## 🐛 Troubleshooting

### **Common Issues**

1. **"Module not found" errors**
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Port 3000 already in use**
   ```bash
   # Kill process on port 3000
   lsof -ti:3000 | xargs kill -9
   # Or use different port
   npm run dev -- --port 3001
   ```

3. **Backend connection errors**
   - Ensure backend is running on `http://localhost:8000`
   - Check CORS settings in backend
   - Verify API endpoints are accessible

4. **Video upload fails**
   - Check file size (max 500MB)
   - Verify file format (MP4, AVI, MOV, MKV, WebM)
   - Ensure backend storage is configured

### **Performance Optimization**

1. **Large video files**
   - Use video compression before upload
   - Consider chunked upload for large files
   - Implement progress indicators

2. **Real-time analytics**
   - Adjust FPS settings in configuration
   - Use WebSocket for real-time updates
   - Implement data throttling

## 📊 Analytics Features

### **Real-time Statistics**
- **Team A vs Team B**: Possession, shots, passes, tackles
- **Player Performance**: Distance covered, speed, acceleration
- **Ball Tracking**: Position, speed, trajectory
- **Event Detection**: Goals, fouls, cards, offsides

### **Visualizations**
- **Possession Chart**: Area chart showing possession over time
- **Performance Bars**: Team comparison charts
- **Events Timeline**: Line chart of match events
- **Ball Speed**: Real-time ball speed tracking
- **Heat Maps**: Player movement patterns

### **Export Options**
- **JSON Results**: Complete analysis data
- **CSV Reports**: Statistics in spreadsheet format
- **Annotated Video**: Video with overlays
- **PDF Reports**: Professional match reports

## 🔮 Future Features

- **Multi-camera Support**: Multiple camera angles
- **3D Visualization**: Three.js powered 3D field view
- **Mobile App**: React Native mobile application
- **AR Overlays**: Augmented reality statistics
- **Voice Commands**: Voice-controlled interface
- **AI Insights**: Automated match insights and recommendations

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review the troubleshooting section
3. Check backend logs
4. Verify all services are running
5. Contact the development team

---

**Ready to analyze football like never before! ⚽🤖**
