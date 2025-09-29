#!/usr/bin/env python3
"""
================================================================================
GODSEYE AI - SIMPLE INFERENCE API
================================================================================

Author: Victor Ibhafidon
Date: January 28, 2025
Version: 1.0.0

DESCRIPTION:
Simple FastAPI server to run model inference on uploaded videos.
This provides a bridge between the frontend and the trained model.

USAGE:
    python simple_inference_api.py
"""

import os
import sys
import json
import tempfile
import subprocess
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI(title="Godseye AI Inference API", version="1.0.0")

# Configure file size limits (2GB+)
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store analysis results and progress
analysis_results = {}
analysis_progress = {}

@app.get("/")
async def root():
    return {"message": "Godseye AI Inference API", "status": "running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """Upload and analyze a football video"""
    try:
        # Generate unique job ID
        job_id = f"job_{int(time.time())}"
        
        print(f"📁 Starting upload: {file.filename}")
        
        # Check file size by reading in chunks
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        # Save uploaded file temporarily with streaming
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024*1024)}GB")
                
                tmp_file.write(chunk)
            
            tmp_file_path = tmp_file.name
        
        print(f"✅ Video uploaded successfully: {file.filename} ({total_size / (1024*1024):.1f} MB)")
        
        # Initialize progress tracking
        analysis_progress[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting analysis...",
            "filename": file.filename
        }
        
        # Store initial job info
        analysis_results[job_id] = {
            "status": "processing",
            "filename": file.filename,
            "output_dir": f"temp_analysis_{job_id}"
        }
        
        # Run analysis in background
        import threading
        analysis_thread = threading.Thread(
            target=run_analysis_background,
            args=(job_id, tmp_file_path, file.filename)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Analysis started successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_analysis_background(job_id, tmp_file_path, filename):
    """Run analysis in background with progress tracking"""
    try:
        output_dir = f"temp_analysis_{job_id}"
        
        # Update progress
        analysis_progress[job_id]["message"] = "Initializing model..."
        analysis_progress[job_id]["progress"] = 5
        
        # Run realistic model inference
        result = subprocess.run([
            sys.executable, "realistic_inference.py", 
            "--video", tmp_file_path,
            "--model", "models/yolov8_improved_referee.pt",
            "--output", output_dir
        ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode != 0:
            raise Exception(f"Model inference failed: {result.stderr}")
        
        # Update progress
        analysis_progress[job_id]["message"] = "Loading results..."
        analysis_progress[job_id]["progress"] = 90
        
        # Load results
        results_file = Path(f"{output_dir}/analysis_results.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                analysis_data = json.load(f)
            
            # Update results
            analysis_results[job_id] = {
                "status": "completed",
                "results": analysis_data,
                "filename": filename,
                "output_dir": output_dir
            }
            
            # Update progress
            analysis_progress[job_id]["status"] = "completed"
            analysis_progress[job_id]["progress"] = 100
            analysis_progress[job_id]["message"] = "Analysis completed successfully"
            
            print(f"✅ Analysis completed for job {job_id}")
        else:
            raise Exception("Analysis results not found")
            
    except Exception as e:
        # Update progress with error
        analysis_progress[job_id]["status"] = "failed"
        analysis_progress[job_id]["message"] = f"Analysis failed: {str(e)}"
        analysis_results[job_id]["status"] = "failed"
        analysis_results[job_id]["error"] = str(e)
        print(f"❌ Analysis failed for job {job_id}: {e}")
    
    finally:
        # Clean up temp files
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/analysis/{job_id}")
async def get_analysis_results(job_id: str):
    """Get analysis results for a job"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return analysis_results[job_id]

@app.get("/progress/{job_id}")
async def get_analysis_progress(job_id: str):
    """Get real-time analysis progress"""
    if job_id not in analysis_progress:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return analysis_progress[job_id]

@app.get("/video/{job_id}")
async def stream_annotated_video(job_id: str):
    """Stream the annotated video with bounding boxes"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    output_dir = analysis_results[job_id].get("output_dir", "temp_analysis")
    video_path = f"{output_dir}/annotated_video.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Annotated video not found")
    
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(iterfile(), media_type="video/mp4")

@app.get("/events/{job_id}")
async def get_events(job_id: str):
    """Get events with timestamps for notifications"""
    if job_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Job not found")
    
    results = analysis_results[job_id]["results"]
    events = results.get("events", [])
    
    # Filter for goal events
    goal_events = [event for event in events if event.get("type") == "goal"]
    
    return {
        "all_events": events,
        "goal_events": goal_events,
        "total_events": len(events)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_available": os.path.exists("models/yolov8_improved_referee.pt")}

if __name__ == "__main__":
    print("🚀 Starting Godseye AI Inference API...")
    print("📡 API will be available at: http://localhost:8001")
    print("🌐 Frontend should connect to: http://localhost:8001")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
