# Drone Localizer

Vision-based drone localization from video or a live camera feed: match drone
footage against a satellite/aerial map to recover the drone's trajectory,
without GPS.

The map is tiled into overlapping, rotated windows across every scale from
3x3 to 15x15 cells. Each window and each video/camera frame are embedded with
a pretrained MobileNet-V3, and the frame's position is estimated via
nearest-neighbor search (FAISS) over the map embeddings, with temporal
smoothing across frames so the trajectory doesn't jump erratically.

## How it works

1. **Tile the map** into a grid of cells, group cells into overlapping
   windows at every size from 3x3 to 15x15, and generate 8 rotated copies of
   each window (0° to 315° in 45° steps).
2. **Embed every window** with a pretrained MobileNet-V3 and index the
   embeddings with FAISS for fast similarity search. Processed windows are
   cached to disk (`drone_map_windows/`, keyed by a hash of the map + config)
   so re-running against the same map skips reprocessing.
3. **Embed each frame** — from a video file or a live camera — the same way,
   and query the index for the best-matching map window (position +
   rotation).
4. **Refine and smooth**: a local search narrows the match to a precise pixel
   location, and a temporal weighting term biases each estimate toward the
   previous frame's position.

## Features

- **Full video or live camera analysis** — process a recorded flight, or a
  live feed (including Jetson CSI cameras via GStreamer, auto-detected).
- **Live map visualization** — a real-time window showing the tracked
  position on the map, trailing path color-coded by confidence, heading
  arrow, and a picture-in-picture camera view.
- **Frame saving** — optionally save every processed frame plus its map
  position and a combined side-by-side view to `frame_outputs/<run>/`.
- **GPU-aware** — uses CUDA + `torch.compile` + FP16 automatically when
  available (Volta or newer), and falls back cleanly to CPU otherwise.
- **Preprocessed window cache management** — inspect, reuse, or delete
  cached map windows from the in-app menu.

## Installation

```bash
git clone https://github.com/mrfares11/Drone-Localization.git
cd Drone-Localization
pip install -r requirements.txt
```

Requires Python 3.7+. `tkinter` ships with most Python installs; on Linux you
may need `sudo apt install python3-tk`.

### NVIDIA Jetson / ARM boards

PyTorch's OpenMP runtime can hit a libgomp TLS allocation issue on Jetson.
Run through the included wrapper instead of calling Python directly:

```bash
./run_with_fix.sh local_video_trajectory_drone_localizer.py
```

## Usage

```bash
python local_video_trajectory_drone_localizer.py
```

This loads MobileNet-V3, prompts you to select a map image, builds (or loads
a cached copy of) the window index, then offers:

1. **Full Video Trajectory Analysis** — process a video file end to end,
   plot the trajectory, and export it to `drone_trajectory.csv`.
2. **Live Camera Feed Analysis** — analyze a live camera stream (port 0,
   10 FPS by default). Press `q` to stop, `s` to save, `r` to reset the
   trajectory mid-run.
3. **Exit**

Live map visualization and frame saving are both enabled by default; toggle
them on the `LocalVideoTrajectoryDroneLocalizer` instance before running:

```python
localizer = LocalVideoTrajectoryDroneLocalizer()
localizer.enable_live_map_visualization = True   # real-time map + trail window
localizer.save_frames_with_positions = True       # write frame_outputs/<run>/...
localizer.frame_save_interval = 1                 # save every Nth frame

localizer.load_map_from_file("map.png")
localizer.process_map()
localizer.analyze_video_from_file("drone_video.mp4")
```

## Configuration

```python
localizer.adjust_movement_freedom(
    search_radius=400,       # px, local search area around the previous position
    max_search_radius=600,   # px, adaptive expansion cap
    temporal_weight=0.3,     # 0-1, how much the previous frame's position pulls the estimate
)
```

Window sizes (3x3 through 15x15) and rotation angles are fixed at
construction time via `LocalVideoTrajectoryDroneLocalizer(base_cell_size=32,
rotation_angles=[...])` if you need a narrower sweep for faster indexing.

## Input requirements

- **Map image**: PNG/JPG/JPEG/BMP/TIFF satellite or aerial view of the flight
  area. Higher resolution improves matching accuracy.
- **Video/camera feed**: MP4/AVI/MOV/MKV/WMV/FLV, or a live camera at the
  configured port, covering the mapped area.

## Output

`export_trajectory()` writes a CSV with one row per processed frame:

```csv
frame,timestamp,x,y,rotation,confidence,window_size
0,0.00,2036.3,764.0,112.6,0.756,12
3,0.10,1912.2,786.4,112.4,0.753,12
```

`x`/`y` are pixel coordinates on the supplied map image, not GPS coordinates
— convert using your map's known scale/georeference if you need real-world
units. When frame saving is enabled, each run additionally writes to
`frame_outputs/<name>_<timestamp>/frames/`, `positions/`, and `combined/`,
plus a `run_info.txt` summary.

## Notes

- MobileNet-V3 weights are downloaded automatically on first run (requires
  internet access once).
- Indexing all 13 window sizes x 8 rotations is comprehensive but slow;
  narrow `rotation_angles` or the window-size range in the constructor if you
  need faster iteration.
- `drone_map_windows/`, `frame_outputs/`, exported CSVs, and rendered videos
  are all gitignored — they're per-run artifacts, not repo content.

## License

MIT — see [LICENSE](LICENSE).
