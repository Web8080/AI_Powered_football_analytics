# Godseye AI Testing Guide

## Quick Start Testing

### 1. Test Individual Components

```bash
# Test the system components
python test_system.py

# Test with verbose output
python test_system.py --verbose
```

### 2. Test with Your Football Video

```bash
# Test with your video file
python test_system.py --video your_football_video.mp4

# Test with verbose output
python test_system.py --video your_football_video.mp4 --verbose
```

### 3. Skip Component Tests (if you just want to test with video)

```bash
python test_system.py --video your_football_video.mp4 --skip-tests
```

## What the Test Does

The test script will:

1. **Test Class Mapping**: Verify the 8-class football classification system
2. **Test Detection Model**: Initialize YOLO detection model
3. **Test Pose Estimation**: Initialize MediaPipe pose estimator
4. **Test Event Detection**: Initialize event detection system
5. **Test Complete Pipeline**: Create the full inference pipeline
6. **Process Your Video**: Run the complete analysis on your video
7. **Export Results**: Save results to `test_results.json`

## Expected Output

```
2025-01-27 10:30:00 - INFO - Starting Godseye AI system tests...
==================================================
2025-01-27 10:30:01 - INFO - Testing class mapping system...
2025-01-27 10:30:01 - INFO - Number of classes: 8
2025-01-27 10:30:01 - INFO - Classes: ['team_a_player', 'team_a_goalkeeper', 'team_b_player', 'team_b_goalkeeper', 'referee', 'ball', 'other', 'staff']
2025-01-27 10:30:01 - INFO - Class mapping test completed ✓

2025-01-27 10:30:02 - INFO - Testing detection model...
2025-01-27 10:30:02 - INFO - Detection model created: YOLODetector
2025-01-27 10:30:02 - INFO - Detection model test completed ✓

2025-01-27 10:30:03 - INFO - Testing pose estimation...
2025-01-27 10:30:03 - INFO - Pose estimator created: MediaPipePoseEstimator
2025-01-27 10:30:03 - INFO - Pose estimation test completed ✓

2025-01-27 10:30:04 - INFO - Testing event detection...
2025-01-27 10:30:04 - INFO - Event detector created: EventDetector
2025-01-27 10:30:04 - INFO - Event detection test completed ✓

2025-01-27 10:30:05 - INFO - Testing inference pipeline...
2025-01-27 10:30:05 - INFO - Inference pipeline created successfully
2025-01-27 10:30:05 - INFO - Inference pipeline test completed ✓

2025-01-27 10:30:06 - INFO - Testing pipeline with video: your_football_video.mp4
2025-01-27 10:30:10 - INFO - Video processing completed. Processed 750 frames.
2025-01-27 10:30:10 - INFO - Summary:
2025-01-27 10:30:10 - INFO -   Total detections: 15,000
2025-01-27 10:30:10 - INFO -   Total tracks: 12,500
2025-01-27 10:30:10 - INFO -   Total poses: 8,750
2025-01-27 10:30:10 - INFO -   Total events: 125
2025-01-27 10:30:10 - INFO - Results exported to test_results.json
2025-01-27 10:30:10 - INFO - Performance metrics:
2025-01-27 10:30:10 - INFO -   Average FPS: 15.2
2025-01-27 10:30:10 - INFO -   Total frames: 750
2025-01-27 10:30:10 - INFO - Video processing test completed ✓
==================================================
2025-01-27 10:30:10 - INFO - All tests completed successfully! ✓
2025-01-27 10:30:10 - INFO - Your Godseye AI system is ready for testing.
```

## Output Files

After running the test, you'll get:

- **`test_results.json`**: Complete analysis results with detections, tracks, poses, and events
- **Console output**: Real-time processing information and statistics

## Understanding the Results

The `test_results.json` file contains:

```json
{
  "pipeline_config": {
    "detection_model": "yolov8n",
    "detection_confidence": 0.5,
    "tracking_max_disappeared": 30,
    "pose_estimator_type": "mediapipe",
    "event_confidence_threshold": 0.5
  },
  "performance_metrics": {
    "total_frames": 750,
    "total_processing_time": 49.3,
    "avg_fps": 15.2
  },
  "results": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "processing_time": 0.065,
      "detections": [
        {
          "class_id": 0,
          "class_name": "team_a_player",
          "confidence": 0.85,
          "bbox": [100, 200, 50, 80],
          "team_id": 0,
          "role": "player"
        }
      ],
      "tracks": [...],
      "poses": [...],
      "events": [...],
      "statistics": {...}
    }
  ]
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory
2. **Video Not Found**: Check the video file path
3. **Memory Issues**: Try reducing the video resolution or processing fewer frames
4. **Slow Processing**: This is normal for CPU processing. GPU would be much faster.

### Performance Tips

- **CPU Processing**: Expect 10-20 FPS on modern CPUs
- **GPU Processing**: Would achieve 30+ FPS (requires CUDA setup)
- **Video Quality**: Higher resolution = slower processing
- **Frame Skipping**: The system can skip frames for faster processing

## Next Steps

After successful testing:

1. **Review Results**: Check the `test_results.json` for analysis quality
2. **Provide Feedback**: Let me know what works well and what needs improvement
3. **Iterate**: We can refine the models and add more features based on your feedback
4. **Scale Up**: Once satisfied, we can optimize for production deployment

## Advanced Testing

For more advanced testing, you can modify the `test_system.py` script to:

- Change confidence thresholds
- Test different models
- Process specific video segments
- Export different formats (CSV, XML)
- Generate visualizations

## Support

If you encounter any issues:

1. Check the console output for error messages
2. Verify your video file is in a supported format (MP4, AVI, MOV)
3. Ensure you have sufficient disk space for output files
4. Try with a shorter video clip first

The system is designed to be robust and provide detailed error messages to help with troubleshooting.
