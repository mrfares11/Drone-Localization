# Drone Localizer - Computer Vision Trajectory Tracking

> Advanced drone localization system using neural networks and computer vision to track trajectories from video footage against satellite maps.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Network-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Similarity%20Search-orange.svg)](https://github.com/facebookresearch/faiss)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/mrfares11/drone-localizer.git
cd drone-localizer

# Install dependencies
pip install -r requirements.txt

# Run the application
python local_video_trajectory_drone_localizer.py
```

## 🎯 Key Features

- **🧠 Neural Feature Matching**: MobileNet-V3 + FAISS for robust similarity search
- **📐 Multi-Scale Analysis**: 13 window sizes (3x3 to 15x15) for comprehensive coverage  
- **🔄 Multi-Rotation Support**: 8-direction analysis for any drone orientation
- **⚡ Smart Caching**: Preprocessed map windows for instant reuse
- **📊 Advanced Tracking**: Adaptive search with temporal smoothing
- **📈 Comprehensive Output**: CSV export with confidence metrics and visualization

## 📋 Requirements

- Python 3.7+
- 4GB+ RAM (8GB+ recommended)
- Windows/Linux/macOS

## � For Researchers

See **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)** for detailed technical architecture, algorithms, and research applications.

## 📁 Project Structure

```
drone-localizer/
├── local_video_trajectory_drone_localizer.py  # Main application
├── requirements.txt                           # Dependencies
├── PROJECT_DESCRIPTION.md                     # Technical details
├── README.md                                  # Quick start guide
└── .gitignore                                # Git ignore rules
```
- `opencv-python` - Computer vision operations
- `pillow` - Image processing
- `faiss-cpu` - Efficient similarity search
- `matplotlib` - Visualization and plotting
- `numpy` - Numerical operations
- `psutil` - Memory monitoring and system information
- `tkinter` - GUI file dialogs (included with Python)

## 🎮 Usage

### Quick Start
```bash
# Validate installation
python validate_installation.py

# Run main application
python local_video_trajectory_drone_localizer.py
```

### Analysis Modes

1. **Full Video Trajectory Analysis**
   - Processes entire video and creates complete trajectory
   - Exports results to CSV
   - Shows comprehensive visualizations

2. **Random Frame Testing (70 frames)**
   - Tests random frames for accuracy assessment
   - Provides detailed frame-by-frame analysis
   - Statistical confidence reporting

3. **Adjust Movement Freedom Settings**
   - Fine-tune search parameters
   - Modify temporal smoothing weights
   - Optimize for specific flight patterns

## ⚙️ Configuration

### Optimal Settings (Current Configuration)
- **Window Size**: 12x12 cells (384x384 pixels)
- **Rotations**: 4 angles (0°, 90°, 180°, 270°)
- **Base Cell Size**: 32x32 pixels
- **Search Radius**: 400px (adaptive up to 600px)
- **Temporal Weight**: 0.3 (30% smoothing)

### Performance Results
- **Average Confidence**: 76.1%
- **Processing Speed**: ~36 seconds for 55-second video
- **Memory Usage**: Optimized for standard hardware
- **Window Count**: 7,400 searchable windows

## 📁 File Structure

```
drone-localizer/
├── local_video_trajectory_drone_localizer.py  # Main application
├── validate_installation.py                   # Installation validator
├── requirements.txt                           # Python dependencies
├── README.md                                 # This documentation
├── drone_map_windows/                        # Cached preprocessed windows
├── preprocessed_windows/                     # Legacy cache directory
├── .venv/                                   # Virtual environment
└── drone_trajectory.csv                     # Output file (generated)
```

## 🎥 Input Requirements

### Map Image
- **Format**: PNG, JPG, JPEG, BMP, TIFF
- **Size**: Any size (tested with 2702x1127)
- **Type**: Satellite/aerial view of flight area
- **Quality**: High resolution recommended for better accuracy

### Video File
- **Format**: MP4, AVI, MOV, MKV, WMV, FLV
- **Content**: Drone footage matching the map area
- **Quality**: Clear, stable footage preferred
- **Duration**: Any length (processing scales automatically)

## 📊 Output Files

### CSV Export (`drone_trajectory.csv`)
```csv
frame,timestamp,x,y,rotation,confidence,window_size
0,0.00,2036.3,764.0,112.6,0.756,12
3,0.10,1912.2,786.4,112.4,0.753,12
...
```

### Visualizations
- Trajectory plot with confidence color-coding
- Start/end point markers
- Rotation direction arrows
- Statistical analysis charts

## 🔧 Troubleshooting

### Common Issues

1. **Memory Error**
   - Solution: Use smaller window sizes or reduce rotation angles
   - Current config is optimized to avoid this

2. **Low Confidence Scores**
   - Check map-video alignment
   - Ensure good lighting in video
   - Verify map resolution

3. **Slow Processing**
   - Enable GPU if available
   - Reduce frame skip rate
   - Use smaller map images

### Performance Tips

- **GPU Acceleration**: Install CUDA for faster processing
- **Memory Management**: Close other applications during processing
- **File Organization**: Keep map and video files locally for faster access

## 📈 Advanced Usage

### Custom Parameters
```python
localizer = LocalVideoTrajectoryDroneLocalizer(
    base_cell_size=32,
    window_sizes=[12],  # Can be modified
    rotation_angles=[0, 90, 180, 270]  # Can be expanded
)
```

### Movement Freedom Adjustment
```python
localizer.adjust_movement_freedom(
    search_radius=400,
    max_search_radius=600,
    temporal_weight=0.3
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for MobileNet-V3
- Facebook AI Research for FAISS
- OpenCV community
- Contributors and testers

## 📞 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the setup guide

---

**Version**: 1.0.0  
**Last Updated**: July 27, 2025  
**Tested Configurations**: Python 3.10, Windows 11, CPU/GPU
