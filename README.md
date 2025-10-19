````markdown
# Optimized Drone Localizer

High-performance drone localization system with comprehensive CPU optimizations and **real-time map visualization**.

## Features

- **Maximum CPU Performance**: Optimized for CPU-only systems
- **Timing Breakdown**: Ttotal = Tembeddings + Tmatching + Tdisplayingresults
- **Smart Caching**: Embedding and similarity caching for speed
- **Batch Processing**: CPU-optimized batch sizes
- **Memory Efficient**: Minimized memory footprint
- **🆕 Live Map Visualization**: Real-time drone position tracking on map with trajectory trails
- **🆕 Picture-in-Picture**: Camera feed shown alongside map visualization
- **🆕 Confidence Color Coding**: Visual feedback on localization quality
- **🆕 Frame Saving**: Automatically save each frame with its position on the map
- **🆕 Combined Views**: Side-by-side camera frame and position visualization

## Quick Start

1. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Run the localizer:
   ```bash
   python local_video_trajectory_drone_localizer.py
   ```

3. **NEW: Try the Live Visualization Demo**:
   ```bash
   python demo_live_visualization.py
   ```

## Live Map Visualization

Watch your drone's position update in real-time on the map as each frame is processed!

### Features
- **Real-time Position Updates**: See drone location update live
- **Trajectory Trail**: Color-coded path showing confidence levels (green=high, red=low)
- **Direction Arrows**: Visual indication of drone heading/rotation
- **Live Statistics**: FPS, distance traveled, confidence metrics
- **Picture-in-Picture**: Camera view shown in corner of map
- **Interactive Controls**: Save, reset, or quit during analysis

### Quick Example
```python
localizer = LocalVideoTrajectoryDroneLocalizer()
localizer.load_map_from_file("map.png")
localizer.process_map()

# Enable live visualization
localizer.enable_live_map_visualization = True

# Analyze video - watch drone move on map in real-time!
localizer.analyze_video_from_file("drone_video.mp4")
```

See **[LIVE_VISUALIZATION_GUIDE.md](LIVE_VISUALIZATION_GUIDE.md)** for detailed documentation.

## Frame Saving with Position Tracking

Save each processed frame along with its predicted position on the map!

### What Gets Saved
- **Original frames** from camera/video
- **Position maps** showing where drone is on the map
- **Combined views** with frame and position side-by-side
- **Summary report** with statistics and metrics

### Quick Example
```python
localizer = LocalVideoTrajectoryDroneLocalizer()
localizer.load_map_from_file()
localizer.process_map()

# Enable frame saving
localizer.save_frames_with_positions = True
localizer.frame_save_interval = 1  # Save every frame

# Analyze - frames saved to timestamped folder
localizer.analyze_video_from_file("video.mp4")
# Output: frame_outputs/video_YYYYMMDD_HHMMSS/
```

See **[FRAME_SAVING_GUIDE.md](FRAME_SAVING_GUIDE.md)** for complete documentation.

## Performance Optimizations

- **CPU Thread Management**: Optimized for 4-core systems
- **Fast Clustering**: Reduced K-means parameters for speed
- **Efficient Search**: Optimized FAISS indexing and similarity search
- **Embedding Caching**: 2000-item cache for repeated computations

## Files

- `local_video_trajectory_drone_localizer.py`: Main optimized localizer
- `demo_live_visualization.py`: 🆕 Interactive demo of live visualization
- `requirements.txt`: All required dependencies
- `drone_trajectory.csv`: Sample trajectory data
- `drone_map_windows/`: Cached preprocessing data
- `LIVE_VISUALIZATION_GUIDE.md`: 🆕 Complete guide to live visualization feature
- `run_with_fix.sh`: Wrapper script for Jetson/ARM systems (fixes libgomp issues)

## Additional Documentation

- `LIVE_VISUALIZATION_GUIDE.md`: 🆕 Complete guide to live visualization feature
- `CAMERA_LIVE_FEED_GUIDE.md`: Camera setup and live feed analysis
- `JETSON_CAMERA_FIX.md`: Fixes for Jetson/ARM systems
- `FRAME_SAVING_GUIDE.md`: 🆕 Guide to automatic frame and position saving

````
