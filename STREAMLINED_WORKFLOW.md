# Streamlined Workflow - Quick Start Guide

## Overview
The drone localizer has been simplified for faster operation with sensible defaults.

## What Changed

### ✅ **Removed Prompts**
- ❌ No more radius customization prompts
- ❌ No more live visualization enable/disable prompts
- ❌ No more frame saving enable/disable prompts
- ❌ No more camera port selection (uses port 0 by default)
- ❌ No more FPS selection (uses 10 FPS by default)

### ✅ **Enabled by Default**
- ✔️ **Live map visualization** - Always enabled
- ✔️ **Frame saving** - Always enabled (saves every frame)
- ✔️ **Camera port 0** - Default for Jetson/Raspberry Pi
- ✔️ **10 FPS processing** - Optimal balance

### ✅ **Simplified Menu**
Now only 3 options:
1. **Full Video Analysis** - Select video file and process
2. **Live Camera Feed** - Uses camera at port 0, 10 FPS
3. **Exit**

## Usage Flow

### Quick Start
```bash
# Run the streamlined version
python local_video_trajectory_drone_localizer.py

# Or with the libgomp fix wrapper
./run_with_fix.sh local_video_trajectory_drone_localizer.py
```

## Default Settings

### Search Parameters
- Search radius: **400px**
- Max search radius: **600px**
- Temporal weight: **0.3**

### Camera Settings (Option 2)
- Camera port: **0** (Jetson CSI camera)
- Target FPS: **10**
- Duration: **Unlimited** (press 'q' to stop)

### Frame Saving
- Save interval: **Every frame** (1)
- Output formats: 
  - frames/ - Original camera frames
  - positions/ - Position marked on map
  - combined/ - Side-by-side view

## Summary

**Before:** 8+ prompts, 7 menu options
**After:** 0 prompts (defaults), 3 menu options

**Time to start processing:** 
- Before: ~30 seconds of answering prompts
- After: ~5 seconds (just select mode)

Everything works exactly the same, just faster! 🚀
