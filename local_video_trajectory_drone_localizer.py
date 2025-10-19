# Local Video Trajectory Drone Localizer for VS Code with Window Size Visualization
# Install required packages: pip install torch torchvision opencv-python pillow faiss-cpu matplotlib numpy tkinter

# NOTE: This script requires LD_PRELOAD to fix libgomp TLS issues on Jetson/ARM
# Use the run_with_fix.sh wrapper script to run this properly
# Or run manually with: LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python local_video_trajectory_drone_localizer.py

import cv2
import numpy as np
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import faiss
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import os
import tkinter as tk
from tkinter import filedialog
import csv
import random
import pickle
import hashlib
import json
from pathlib import Path
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not available - memory monitoring disabled")
    PSUTIL_AVAILABLE = False

# Hide tkinter root window
root = tk.Tk()
root.withdraw()

@dataclass
class CellInfo:
    row: int
    col: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    center_x: float
    center_y: float
    embedding: Optional[np.ndarray] = None

@dataclass
class MultiCellWindow:
    top_left_row: int
    top_left_col: int
    window_rows: int
    window_cols: int
    cells: List[CellInfo]
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    center_x: float
    center_y: float
    scale_factor: float
    rotation_angle: float = 0.0
    embedding: Optional[np.ndarray] = None
    # Remove stored images to save memory - we'll generate them on-demand
    # composite_image: Optional[np.ndarray] = None
    # rotated_image: Optional[np.ndarray] = None

@dataclass
class TrajectoryPoint:
    frame_number: int
    timestamp: float
    x: float
    y: float
    rotation: float
    confidence: float
    window_info: Optional[MultiCellWindow] = None

@dataclass
class FrameTimings:
    """Track detailed timing for each frame processing component"""
    total_time: float = 0.0
    embedding_time: float = 0.0
    matching_time: float = 0.0
    display_time: float = 0.0
    preprocessing_time: float = 0.0

class LocalVideoTrajectoryDroneLocalizer:
    def __init__(self, base_cell_size=32, rotation_angles=[0, 45, 90, 135, 180, 225, 270, 315]):
        self.base_cell_size = base_cell_size
        
        # SIMPLIFIED: Use ALL window sizes from smallest to largest
        # Generate comprehensive range from 3x3 to 15x15
        self.window_sizes = []
        for size in range(3, 16):  # 3x3 to 15x15
            self.window_sizes.append((size, size))  # Square windows only for simplicity
        
        print(f"Using ALL window sizes from 3x3 to 15x15: {len(self.window_sizes)} sizes")
        print(f"   Window sizes: {', '.join([f'{h}x{w}' for h, w in self.window_sizes])}")
        self.rotation_angles = rotation_angles
        
        # GPU optimization setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Initialize elevation levels for drone altitude-based window sizing
        self.elevation_levels = {
            "so_low": {
                "altitude_range": "5-25m",
                "description": "Very low altitude - detailed inspection flights",
                "square_sizes": [(3, 3), (4, 4), (5, 5)],
                "rectangular_sizes": [(3, 4), (4, 5), (5, 6)]
            },
            "low": {
                "altitude_range": "25-50m", 
                "description": "Low altitude - close terrain analysis",
                "square_sizes": [(5, 5), (6, 6), (7, 7)],
                "rectangular_sizes": [(5, 7), (6, 8), (7, 9)]
            },
            "mid": {
                "altitude_range": "50-100m",
                "description": "Medium altitude - standard operations", 
                "square_sizes": [(7, 7), (8, 8), (9, 9), (10, 10)],
                "rectangular_sizes": [(7, 10), (8, 11), (9, 12)]
            },
            "high": {
                "altitude_range": "100m+",
                "description": "High altitude - area surveys",
                "square_sizes": [(10, 10), (11, 11), (12, 12), (13, 13)],
                "rectangular_sizes": [(10, 13), (11, 14), (12, 15)]
            }
        }
        
        # Drone aspect ratio (4000x2250 pixels = 1.778:1)
        self.drone_aspect_ratio = 4000 / 2250
        self.use_rectangular_windows = True  # Allow rectangular windows by default

        # Load MobileNet-V3 with GPU optimization
        print(f"Loading MobileNet-V3 on {self.device} with optimizations...")
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        
        # Optimize model for inference
        if self.device == 'cuda':
            # Enable optimized attention and memory efficient attention if available
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
            
        # Compile model for better performance (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("   Model compiled with torch.compile for faster inference")
        except Exception as e:
            print(f"   Model compilation not available: {e}")

        # Optimized transforms with GPU acceleration
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),  # Enable antialiasing
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Batch processing optimization
        self.batch_size = 16 if self.device == 'cuda' else 4
        self.max_batch_size = 32 if self.device == 'cuda' else 8
        
        # Precompute transforms for common operations
        self.pil_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.cells = []
        self.multi_cell_windows = []
        self.map_image = None
        self.index = None
        self.trajectory = []
        self.previous_location = None
        
        # Enhanced movement freedom parameters
        self.search_radius = 400
        self.min_confidence_for_temporal = 0.6
        self.max_search_radius = 600
        self.temporal_weight = 0.3
        
        # Performance optimization flags
        self.enable_visualization = True
        self.enable_gpu_faiss = False  # Will be enabled if faiss-gpu is available
        self.enable_half_precision = False  # FP16 inference
        
        # Live visualization settings
        self.enable_live_map_visualization = True  # Show live position on map
        self.live_viz_window_name = "Drone Localization - Live Map View"
        self.live_viz_trail_length = 100  # Number of recent points to show in trail
        self.live_viz_update_interval = 1  # Update map every N frames
        self.live_viz_video_writer = None  # VideoWriter for saving visualization
        self.live_viz_save_video = False  # Set to True to save visualization as video
        
        # Frame saving settings
        self.save_frames_with_positions = False  # Set to True to save each frame with position
        self.frame_save_dir = None  # Directory for saving frames (auto-generated)
        self.frame_save_interval = 1  # Save every N frames
        self.save_original_frame = True  # Save the original camera frame
        self.save_position_on_map = True  # Save frame with position marked on map
        
        # Check for FAISS GPU support
        try:
            import faiss
            if hasattr(faiss, 'StandardGpuResources') and torch.cuda.is_available():
                self.enable_gpu_faiss = True
                print("   FAISS GPU support detected")
        except:
            print("   FAISS GPU not available, using CPU version")
        
        # Enable half precision if supported
        if self.device == 'cuda' and torch.cuda.get_device_capability()[0] >= 7:  # Volta architecture or newer
            self.enable_half_precision = True
            self.model = self.model.half()
            print("   Enabled FP16 inference for faster processing")
        
        # Preprocessed windows management - ALWAYS SAVE for reuse
        self.preprocessed_windows_dir = Path("drone_map_windows")
        self.preprocessed_windows_dir.mkdir(exist_ok=True)
        self.current_map_hash = None
        self.windows_file_path = None
        self.save_windows_for_reuse = True
        
        # Cache for frequently used computations
        self._embedding_cache = {}
        self._similarity_cache = {}
        
        # Timing statistics
        self.frame_timings = []
        self.total_embedding_time = 0.0
        self.total_matching_time = 0.0
        self.total_display_time = 0.0
        
        print("Localizer initialized with maximum performance optimizations!")

    def cleanup_memory(self):
        """Clean up memory and release resources"""
        try:
            # Clear large data structures
            if hasattr(self, 'index') and self.index:
                self.index = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                print(f"Memory usage after cleanup: {memory_usage:.1f} MB")
                
        except Exception as e:
            print(f"Warning: Error during memory cleanup: {e}")

    def check_memory_usage(self):
        """Check current memory usage"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Current memory usage: {memory_usage:.1f} MB")
            return memory_usage
        return None

    def _safe_show_plot(self, title="Plot"):
        """Safely show matplotlib plot without blocking"""
        # Skip visualization if disabled (useful for headless operation)
        if not getattr(self, 'enable_visualization', True):
            plt.close('all')
            print(f"⏭ {title} visualization skipped (disabled)")
            return
            
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Brief pause to render
            plt.close('all')  # Close to free memory
            print(f" {title} displayed")
        except Exception as e:
            print(f" {title} display not available: {e}")
            plt.close('all')

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if not PSUTIL_AVAILABLE:
            return 0  # Return 0 if psutil is not available
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            return memory_mb
        except Exception:
            return 0

    def print_memory_stats(self, stage=""):
        """Print current memory usage"""
        if not PSUTIL_AVAILABLE:
            print(f" Memory monitoring not available (install psutil: pip install psutil)")
            return
        
        try:
            memory_mb = self.get_memory_usage()
            print(f" Memory usage {stage}: {memory_mb:.1f} MB ({memory_mb/1024:.2f} GB)")
        except Exception:
            print(" Memory monitoring error")

    def generate_map_hash(self, map_path, window_sizes, rotation_angles, cell_size):
        """Generate a unique hash for the map configuration"""
        # Create a string with all parameters that affect window generation
        config_str = f"{map_path}_{window_sizes}_{rotation_angles}_{cell_size}_{self.map_width}_{self.map_height}"
        
        # Add file modification time to detect map changes
        if os.path.exists(map_path):
            mtime = os.path.getmtime(map_path)
            config_str += f"_{mtime}"
        
        # Generate hash
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()

    def get_windows_file_path(self, map_hash):
        """Get the file path for storing preprocessed windows"""
        windows_file = self.preprocessed_windows_dir / f"windows_{map_hash}.pkl"
        metadata_file = self.preprocessed_windows_dir / f"metadata_{map_hash}.json"
        return windows_file, metadata_file

    def save_preprocessed_windows(self):
        """Save preprocessed windows and metadata to disk"""
        if not self.windows_file_path or not self.save_windows_for_reuse:
            return
        
        try:
            windows_file, metadata_file = self.windows_file_path
            
            # Prepare data for saving (only essential data)
            save_data = {
                'multi_cell_windows': [],
                'cells': self.cells,
                'all_info': self.all_info if hasattr(self, 'all_info') else []
            }
            
            # Save only essential window data (without large images)
            for window in self.multi_cell_windows:
                window_data = {
                    'top_left_row': window.top_left_row,
                    'top_left_col': window.top_left_col,
                    'window_rows': window.window_rows,
                    'window_cols': window.window_cols,
                    'cells': window.cells,
                    'x_start': window.x_start,
                    'y_start': window.y_start,
                    'x_end': window.x_end,
                    'y_end': window.y_end,
                    'center_x': window.center_x,
                    'center_y': window.center_y,
                    'scale_factor': window.scale_factor,
                    'rotation_angle': window.rotation_angle,
                    'embedding': window.embedding
                }
                save_data['multi_cell_windows'].append(window_data)
            
            # Save windows data with improved compatibility
            print(f" Saving preprocessed windows to {windows_file.name}...")
            
            # Use protocol 4 for better compatibility and add module info
            save_data['_module_info'] = {
                'numpy_version': np.__version__,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'saved_by': 'LocalVideoTrajectoryDroneLocalizer'
            }
            
            with open(windows_file, 'wb') as f:
                pickle.dump(save_data, f, protocol=4)  # Use protocol 4 for better compatibility
            
            # Save metadata
            metadata = {
                'window_sizes': self.window_sizes,
                'rotation_angles': self.rotation_angles,
                'base_cell_size': self.base_cell_size,
                'map_width': self.map_width,
                'map_height': self.map_height,
                'total_windows': len(self.multi_cell_windows),
                'created_at': time.time(),
                'map_hash': self.current_map_hash,
                'original_map_path': getattr(self, 'current_map_path', None)  # Save original map path
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
            print(f" Preprocessed windows saved to {windows_file.name}")
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Windows count: {len(self.multi_cell_windows):,}")
            
        except Exception as e:
            print(f" Error saving preprocessed windows: {e}")

    def _load_with_custom_unpickler(self, file_obj):
        """Custom unpickler to handle class resolution issues"""
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Redirect __main__ classes to current module
                if module == '__main__':
                    if name in ['CellInfo', 'MultiCellWindow']:
                        return globals()[name]
                # Handle numpy version changes
                if module == 'numpy._core.numeric':
                    import numpy as np
                    return getattr(np, name)
                return super().find_class(module, name)
        return CustomUnpickler(file_obj).load()
    
    def _load_with_safe_unpickler(self, file_obj):
        """Safe unpickler with comprehensive compatibility fixes"""
        import sys
        import numpy as np
        
        # Store original modules
        original_modules = {}
        
        # Add compatibility modules
        if 'numpy._core.numeric' not in sys.modules:
            sys.modules['numpy._core.numeric'] = np
        
        # Add main module classes
        if '__main__' not in sys.modules:
            import types
            main_module = types.ModuleType('__main__')
            main_module.CellInfo = CellInfo
            main_module.MultiCellWindow = MultiCellWindow
            sys.modules['__main__'] = main_module
        
        try:
            return pickle.load(file_obj)
        finally:
            # Cleanup
            for mod_name in ['numpy._core.numeric', '__main__']:
                if mod_name in original_modules:
                    sys.modules[mod_name] = original_modules[mod_name]
                elif mod_name in sys.modules:
                    del sys.modules[mod_name]

    def load_preprocessed_windows(self, windows_file, metadata_file):
        """Load preprocessed windows from disk"""
        try:
            # Load and verify metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Convert saved window sizes from lists to tuples for comparison
            saved_window_sizes = [tuple(ws) for ws in metadata['window_sizes']]
            
            # Check if we have map dimensions already (when called after map loading)
            # or if we're loading preprocessed data first (when called before map loading)
            map_width_check = getattr(self, 'map_width', metadata['map_width'])
            map_height_check = getattr(self, 'map_height', metadata['map_height'])
            
            # Verify compatibility
            if (saved_window_sizes != self.window_sizes or
                metadata['rotation_angles'] != self.rotation_angles or
                metadata['base_cell_size'] != self.base_cell_size):
                print(" Preprocessed windows are incompatible with current settings")
                print(f"   Saved sizes: {saved_window_sizes}")
                print(f"   Current sizes: {self.window_sizes}")
                print(f"   Saved rotations: {metadata['rotation_angles']}")
                print(f"   Current rotations: {self.rotation_angles}")
                print(f"   Saved cell size: {metadata['base_cell_size']}")
                print(f"   Current cell size: {self.base_cell_size}")
                return False
            
            # Set map dimensions from metadata if not already set
            if not hasattr(self, 'map_width'):
                self.map_width = metadata['map_width']
                self.map_height = metadata['map_height']
            
            # Load windows data with improved error handling and fallback options
            print(f" Loading preprocessed windows from {os.path.basename(windows_file)}...")
            
            # Try multiple loading strategies
            save_data = None
            loading_methods = [
                ("Standard pickle", lambda f: pickle.load(f)),
                ("Custom unpickler", self._load_with_custom_unpickler),
                ("Safe unpickler", self._load_with_safe_unpickler)
            ]
            
            for method_name, load_func in loading_methods:
                try:
                    print(f"   Trying {method_name}...")
                    with open(windows_file, 'rb') as f:
                        save_data = load_func(f)
                    print(f"    {method_name} successful!")
                    break
                except Exception as e:
                    print(f"    {method_name} failed: {e}")
                    continue
            
            if save_data is None:
                print(" All loading methods failed")
                print(" Preprocessed windows will be regenerated with current compatibility")
                return False
            
            # Restore cells
            self.cells = save_data['cells']
            
            # Restore windows
            self.multi_cell_windows = []
            for window_data in save_data['multi_cell_windows']:
                window = MultiCellWindow(**window_data)
                self.multi_cell_windows.append(window)
            
            # Restore all_info if available
            if 'all_info' in save_data:
                self.all_info = save_data['all_info']
            else:
                # Rebuild all_info
                self.all_info = []
                for window in self.multi_cell_windows:
                    self.all_info.append(('window', window))
            
            # Rebuild FAISS index
            print(" Rebuilding FAISS index...")
            all_embeddings = []
            for window in self.multi_cell_windows:
                if window.embedding is not None:
                    all_embeddings.append(window.embedding)
            
            if all_embeddings:
                all_embeddings = np.array(all_embeddings)
                all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
                
                dimension = all_embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(all_embeddings.astype('float32'))
            
            file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
            created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_at']))
            
            print(f" Loaded {len(self.multi_cell_windows):,} preprocessed windows")
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Created: {created_time}")
            print(f"   Window sizes: {metadata['window_sizes']}")
            print(f"   Rotations: {len(metadata['rotation_angles'])} angles")
            
            # Try to load the original map image if path is available
            original_map_path = metadata.get('original_map_path')
            map_loaded = False
            
            if original_map_path and original_map_path != "preprocessed_map_loaded":
                try:
                    if os.path.exists(original_map_path):
                        print(f" Loading original map image: {os.path.basename(original_map_path)}")
                        self.map_image = cv2.imread(original_map_path)
                        if self.map_image is not None:
                            self.current_map_path = original_map_path
                            self._is_placeholder = False
                            map_loaded = True
                            print(f" Original map image loaded successfully")
                        else:
                            print(f" Could not read map file: {original_map_path}")
                    else:
                        print(f" Original map file not found: {original_map_path}")
                except Exception as e:
                    print(f" Error loading original map: {e}")
            
            # If no original map path saved, offer to select one manually
            elif not original_map_path or original_map_path == "Not saved":
                print(" No original map path saved in preprocessed data")
                print(" You can load the original map image for better trajectory visualization")
                
                while True:
                    choice = input("   Load original map image? [Y/n]: ").strip().lower()
                    if choice in ['', 'y', 'yes']:
                        try:
                            from tkinter import filedialog
                            map_path = filedialog.askopenfilename(
                                title="Select Original Map Image",
                                filetypes=[
                                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                                    ("All files", "*.*")
                                ]
                            )
                            
                            if map_path and os.path.exists(map_path):
                                test_image = cv2.imread(map_path)
                                if (test_image is not None and 
                                    test_image.shape[:2] == (self.map_height, self.map_width)):
                                    self.map_image = test_image
                                    self.current_map_path = map_path
                                    self._is_placeholder = False
                                    map_loaded = True
                                    print(f" Original map loaded: {os.path.basename(map_path)}")
                                    break
                                elif test_image is not None:
                                    print(f" Map dimensions don't match: {test_image.shape[:2]} vs expected {(self.map_height, self.map_width)}")
                                else:
                                    print(f" Could not read image file")
                            else:
                                print(" No valid map file selected")
                                break
                        except Exception as e:
                            print(f" Error selecting map: {e}")
                            break
                    elif choice in ['n', 'no']:
                        print(" Using placeholder map for visualization")
                        break
                    else:
                        print(" Please enter 'y' for yes or 'n' for no")
            
            # Fall back to placeholder if original map couldn't be loaded
            if not map_loaded:
                if not original_map_path or original_map_path == "Not saved":
                    print(" Creating placeholder map image")
                else:
                    print(" Creating placeholder map image (original map not available)")
                self.map_image = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 128  # Gray placeholder
                self._is_placeholder = True
                self.current_map_path = "preprocessed_map_loaded"
            
            return True
            
        except Exception as e:
            print(f" Error loading preprocessed windows: {e}")
            return False

    def check_for_preprocessed_windows(self, map_path):
        """Check if preprocessed windows exist for this map configuration"""
        self.current_map_hash = self.generate_map_hash(map_path, self.window_sizes, self.rotation_angles, self.base_cell_size)
        windows_file, metadata_file = self.get_windows_file_path(self.current_map_hash)
        self.windows_file_path = (windows_file, metadata_file)
        
        if windows_file.exists() and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
                created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_at']))
                
                print(f"\n Found preprocessed windows for this map configuration:")
                print(f"   File: {windows_file.name}")
                print(f"   Size: {file_size_mb:.1f} MB")
                print(f"   Windows: {metadata['total_windows']:,}")
                print(f"   Created: {created_time}")
                print(f"   Window sizes: {metadata['window_sizes']}")
                print(f"   Rotations: {len(metadata['rotation_angles'])} angles")
                
                return True, windows_file, metadata_file
            except:
                return False, None, None
        
        return False, None, None

    def cleanup_preprocessed_windows(self):
        """Clean up preprocessed windows if user chose not to save them"""
        if not self.save_windows_for_reuse and self.windows_file_path:
            windows_file, metadata_file = self.windows_file_path
            
            try:
                if windows_file.exists():
                    file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
                    os.remove(windows_file)
                    print(f" Cleaned up windows file ({file_size_mb:.1f} MB)")
                
                if metadata_file.exists():
                    os.remove(metadata_file)
                    print(f" Cleaned up metadata file")
                    
            except Exception as e:
                print(f" Error cleaning up preprocessed windows: {e}")

    def ask_about_preprocessed_windows(self, map_path):
        """Ask user about loading/saving preprocessed windows"""
        
        # If we already loaded preprocessed windows, skip this step
        if map_path == "preprocessed_map_loaded":
            print(" Using already loaded preprocessed windows")
            return True
        
        found, windows_file, metadata_file = self.check_for_preprocessed_windows(map_path)
        
        if found:
            print(f"\n Do you want to use the existing preprocessed windows?")
            print("   This will skip the time-consuming window generation process.")
            
            while True:
                choice = input("   Use preprocessed windows? [Y/n]: ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    if self.load_preprocessed_windows(windows_file, metadata_file):
                        return True  # Successfully loaded
                    else:
                        print(" Failed to load preprocessed windows, will generate new ones")
                        break
                elif choice in ['n', 'no']:
                    print(" Will generate new windows from scratch")
                    break
                else:
                    print(" Please enter 'y' for yes or 'n' for no")
        
        # Ask about saving for future use
        print(f"\n Do you want to save the processed windows for future reuse?")
        print("   This will speed up future analyses of the same map with same settings.")
        print("   If you choose 'no', windows will be cleaned up after analysis to save storage.")
        
        while True:
            choice = input("   Save windows for reuse? [Y/n]: ").strip().lower()
            if choice in ['', 'y', 'yes']:
                self.save_windows_for_reuse = True
                print(" Will save windows for future reuse")
                break
            elif choice in ['n', 'no']:
                self.save_windows_for_reuse = False
                print(" Will clean up windows after analysis to save storage")
                break
            else:
                print(" Please enter 'y' for yes or 'n' for no")
        
        return False  # Need to generate new windows

    def load_map_from_file(self, map_path=None):
        """Load map image from local file with preprocessed windows support"""
        
        # First, ask if user wants to use preprocessed map windows
        print("\n DRONE LOCALIZER - MAP LOADING")
        print("=" * 50)
        
        # Check if there are any preprocessed windows available
        existing_windows = list(self.preprocessed_windows_dir.glob("windows_*.pkl"))
        
        if existing_windows:
            print(f" Found {len(existing_windows)} preprocessed map(s) in 'drone_map_windows' folder:")
            
            # Show available preprocessed maps
            valid_options = []
            for i, windows_file in enumerate(existing_windows):
                hash_id = windows_file.stem.replace("windows_", "")
                metadata_file = self.preprocessed_windows_dir / f"metadata_{hash_id}.json"
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
                        created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_at']))
                        
                        print(f"\n{i+1}. Preprocessed Map")
                        print(f"   Size: {file_size_mb:.1f} MB")
                        print(f"   Windows: {metadata['total_windows']:,}")
                        print(f"   Created: {created_time}")
                        print(f"   Map size: {metadata['map_width']}x{metadata['map_height']}")
                        
                        valid_options.append((windows_file, metadata_file, metadata))
                        
                    except Exception as e:
                        print(f" Error reading metadata for {windows_file.name}: {e}")
            
            if valid_options:
                print(f"\n Choose an option:")
                for i, _ in enumerate(valid_options):
                    print(f"{i+1}. Use preprocessed map #{i+1}")
                print(f"{len(valid_options)+1}. Load new map image")
                
                while True:
                    try:
                        choice = int(input(f"\nEnter your choice (1-{len(valid_options)+1}): ").strip())
                        if 1 <= choice <= len(valid_options):
                            # Load selected preprocessed map
                            windows_file, metadata_file, metadata = valid_options[choice-1]
                            print(f"\n Loading preprocessed map #{choice}...")
                            
                            if self.load_preprocessed_windows(windows_file, metadata_file):
                                return "preprocessed_map_loaded"
                            else:
                                print(" Failed to load preprocessed map, continuing to new map...")
                                break
                                
                        elif choice == len(valid_options)+1:
                            print(" Loading new map image...")
                            break
                        else:
                            print(f" Invalid choice. Please enter 1-{len(valid_options)+1}")
                    except ValueError:
                        print(" Please enter a valid number")
        
        # Load new map image
        if map_path is None:
            print(" Please select your map image file...")
            map_path = filedialog.askopenfilename(
                title="Select Map Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                    ("All files", "*.*")
                ]
            )
        
        if not map_path or not os.path.exists(map_path):
            raise ValueError("No map file selected or file doesn't exist")

        self.map_image = cv2.imread(map_path)
        if self.map_image is None:
            raise ValueError(f"Could not load image from: {map_path}")

        self.map_height, self.map_width = self.map_image.shape[:2]
        filename = os.path.basename(map_path)
        print(f" Loaded map: {filename} ({self.map_width}x{self.map_height} pixels)")

        # Show map
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Map: {filename} (Drone Localizer - Processing ALL Window Sizes)')
        plt.axis('off')
        self._safe_show_plot("Map visualization")

        # Store the map path for preprocessed windows management
        self.current_map_path = map_path
        
        # Generate hash for this configuration
        self.current_map_hash = self.generate_map_hash(map_path, self.window_sizes, self.rotation_angles, self.base_cell_size)
        self.windows_file_path = self.get_windows_file_path(self.current_map_hash)
        
        return map_path


    def configure_commercial_settings(self):
        """Commercial-ready configuration wizard for optimal drone localization"""
        print("\n COMMERCIAL DRONE LOCALIZER - CONFIGURATION WIZARD")
        print("=" * 65)
        print("This wizard will help you configure optimal settings for your specific use case.")
        print()
        
        # Step 1: Map characteristics
        print(" STEP 1: Map Characteristics")
        print("-" * 30)
        print("Your map's viewing characteristics significantly impact matching accuracy.")
        print()
        
        map_type = self._ask_map_type()
        map_resolution = self._ask_map_resolution() 
        
        # Step 2: Window shape strategy
        print("\n STEP 2: Window Shape Strategy")
        print("-" * 35)
        print("Based on testing, SQUARE windows typically provide better accuracy (0.7 confidence)")
        print("while rectangular windows may reduce accuracy (0.3 confidence).")
        print()
        
        window_strategy = self._ask_window_strategy()
        
        # Step 3: Elevation-based optimization
        print("\n STEP 3: Flight Altitude Optimization") 
        print("-" * 40)
        print("Choose elevation levels that match your typical drone operations.")
        print()
        
        elevation_selection = self._ask_elevation_selection()
        
        # Apply configuration
        self._apply_commercial_config(map_type, map_resolution, window_strategy, elevation_selection)
        
        print("\n Configuration complete! Your system is optimized for commercial use.")
        return True
    
    def _ask_map_type(self):
        """Ask user about their map source and characteristics"""
        print("What type of map are you using?")
        print("1. Google Earth satellite imagery (high altitude, orthographic)")
        print("2. Drone-captured map (similar altitude to operational flights)")
        print("3. Mixed/Custom imagery")
        
        while True:
            try:
                choice = input("Select map type (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    return int(choice)
                else:
                    print(" Please enter 1, 2, or 3")
            except ValueError:
                print(" Please enter a valid number")
    
    def _ask_map_resolution(self):
        """Ask about map resolution and scale"""
        print("\nWhat is your map resolution/detail level?")
        print("1. Very High Detail (close-up satellite, <1m per pixel)")
        print("2. High Detail (standard satellite, 1-3m per pixel)")
        print("3. Medium Detail (regional view, 3-10m per pixel)")
        print("4. Low Detail (wide area view, >10m per pixel)")
        
        while True:
            try:
                choice = input("Select resolution (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print(" Please enter 1, 2, 3, or 4")
            except ValueError:
                print(" Please enter a valid number")
    
    def _ask_window_strategy(self):
        """Ask about window shape preference"""
        print("Choose window shape strategy:")
        print("1.  SQUARE ONLY (Recommended) - Proven 0.7+ confidence")
        print("2.  MIXED (Square + Rectangular) - Experimental, may reduce accuracy")
        print("3.  RECTANGULAR ONLY - Not recommended based on your results")
        print("4.  CUSTOM - Let me choose specific sizes")
        
        while True:
            try:
                choice = input("Select strategy (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print(" Please enter 1, 2, 3, or 4")
            except ValueError:
                print(" Please enter a valid number")
                
    def _ask_elevation_selection(self):
        """Ask about elevation/altitude preferences"""
        print("Which drone altitudes will you typically use?")
        print("1.  LOW altitude flights (5-50m) - Detailed inspection")
        print("2.  MEDIUM altitude flights (50-100m) - Standard operations") 
        print("3.  HIGH altitude flights (100m+) - Area surveys")
        print("4.  ALL altitudes - Complete coverage")
        
        while True:
            try:
                choice = input("Select altitude range (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print(" Please enter 1, 2, 3, or 4")
            except ValueError:
                print(" Please enter a valid number")
    
    def _apply_commercial_config(self, map_type, map_resolution, window_strategy, elevation_selection):
        """Apply the commercial configuration based on user choices"""
        print(f"\n APPLYING CONFIGURATION...")
        print("-" * 30)
        
        # Configure window strategy
        if window_strategy == 1:  # Square only
            self.use_rectangular_windows = False
            print(" Window strategy: SQUARE ONLY (optimal for accuracy)")
            
        elif window_strategy == 2:  # Mixed
            self.use_rectangular_windows = True
            print(" Window strategy: MIXED (may reduce accuracy)")
            print("   Consider testing with Square Only if accuracy is insufficient")
            
        elif window_strategy == 3:  # Rectangular only
            self.use_rectangular_windows = True
            print(" Window strategy: RECTANGULAR ONLY (not recommended)")
            print("   Strongly consider switching to Square Only for better results")
            
        elif window_strategy == 4:  # Custom
            print(" Window strategy: CUSTOM - you'll select specific sizes")
        
        # Configure elevation-based sizes
        if elevation_selection == 1:  # Low altitude
            selected_levels = ["so_low", "low"]
        elif elevation_selection == 2:  # Medium altitude
            selected_levels = ["low", "mid"]
        elif elevation_selection == 3:  # High altitude
            selected_levels = ["mid", "high"]
        else:  # All altitudes
            selected_levels = ["so_low", "low", "mid", "high"]
        
        # Build window sizes based on configuration
        window_sizes = []
        for level in selected_levels:
            info = self.elevation_levels[level]
            
            # Always include square sizes (proven effective)
            square_sizes = info.get("square_sizes", [])
            window_sizes.extend(square_sizes)
            
            # Include rectangular only if enabled
            if self.use_rectangular_windows and window_strategy != 1:
                rect_sizes = info.get("rectangular_sizes", [])
                window_sizes.extend(rect_sizes)
        
        # Remove duplicates and limit size
        seen = set()
        unique_sizes = []
        for size in window_sizes:
            if size not in seen:
                seen.add(size)
                unique_sizes.append(size)
        
        # Optimize based on map characteristics
        if map_resolution == 1:  # Very high detail
            # Use larger windows for high detail maps
            unique_sizes = [s for s in unique_sizes if max(s) >= 5]
        elif map_resolution == 4:  # Low detail
            # Use smaller windows for low detail maps
            unique_sizes = [s for s in unique_sizes if max(s) <= 7]
        
        self.window_sizes = unique_sizes[:10]  # Limit to 10 sizes max
        
        # Set appropriate cell size based on map resolution
        if map_resolution == 1:  # Very high detail
            suggested_cell_size = 16  # Smaller cells for high detail
        elif map_resolution == 4:  # Low detail
            suggested_cell_size = 64  # Larger cells for low detail
        else:
            suggested_cell_size = 32  # Default
        
        if suggested_cell_size != self.base_cell_size:
            print(f" RECOMMENDATION: Consider using cell size {suggested_cell_size}x{suggested_cell_size} pixels")
            print(f"   (Current: {self.base_cell_size}x{self.base_cell_size})")
            print(f"   Restart with base_cell_size={suggested_cell_size} for optimal results")
        
        print(f" Selected {len(self.window_sizes)} window sizes: {self.window_sizes}")
        print(f" Elevation levels: {', '.join(selected_levels)}")
        
        # Store configuration for future reference
        self.commercial_config = {
            'map_type': map_type,
            'map_resolution': map_resolution, 
            'window_strategy': window_strategy,
            'elevation_selection': elevation_selection,
            'use_rectangular': self.use_rectangular_windows,
            'selected_levels': selected_levels,
            'window_sizes': self.window_sizes
        }

    def choose_elevation_level(self):
        """Let user choose elevation level to get appropriate window sizes"""
        print("\n DRONE ELEVATION-BASED WINDOW SELECTION")
        print("=" * 50)
        print("Choose your drone's flight altitude to get optimized window sizes:")
        print(" NOTE: Square windows typically provide better accuracy than rectangular ones!")
        print()
        
        levels = ["so_low", "low", "mid", "high"]
        for i, level in enumerate(levels):
            info = self.elevation_levels[level]
            altitude_range = info["altitude_range"]
            description = info["description"]
            square_sizes = info.get("square_sizes", [])
            rect_sizes = info.get("rectangular_sizes", [])
            
            print(f"{i+1}. {level.upper().replace('_', ' ')} ({altitude_range})")
            print(f"   {description}")
            print(f"   Square sizes (recommended): {square_sizes}")
            if self.use_rectangular_windows:
                print(f"   Rectangular sizes (optional): {rect_sizes}")
            print()
        
        print("5. COMMERCIAL CONFIG - Use configuration wizard (recommended)")
        print("6. CUSTOM - Mix sizes from different levels")
        print("7. ALL LEVELS - Use sizes from all elevation levels")
        
        while True:
            try:
                choice = input("\nSelect elevation level (1-7): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= 4:
                    level = levels[choice_num - 1]
                    return self._select_level_sizes(level)
                elif choice_num == 5:
                    if self.configure_commercial_settings():
                        return self.window_sizes
                elif choice_num == 6:
                    return self.choose_custom_elevation_mix()
                elif choice_num == 7:
                    return self.choose_all_elevation_levels()
                else:
                    print(" Please enter 1-7")
            except ValueError:
                print(" Please enter a valid number")
    
    def _select_level_sizes(self, level):
        """Select sizes for a specific elevation level"""
        info = self.elevation_levels[level]
        selected_sizes = []
        
        # Always include square sizes (proven effective)
        square_sizes = info.get("square_sizes", [])
        selected_sizes.extend(square_sizes)
        
        # Optionally include rectangular sizes
        if self.use_rectangular_windows:
            rect_sizes = info.get("rectangular_sizes", [])
            
            print(f"\n Include rectangular windows for {level.upper().replace('_', ' ')}?")
            print(" Warning: Rectangular windows may reduce accuracy")
            include_rect = input("Include rectangular sizes? [y/N]: ").strip().lower()
            
            if include_rect in ['y', 'yes']:
                selected_sizes.extend(rect_sizes)
                print(" Added rectangular sizes - monitor accuracy carefully")
            else:
                print(" Using square sizes only (recommended)")
        
        self.window_sizes = selected_sizes
        self.current_elevation_level = level
        
        sizes_str = ", ".join([f"{h}×{w}" for h, w in selected_sizes])
        level_name = level.upper().replace('_', ' ')
        print(f"\n Selected {level_name} elevation level")
        print(f"   Window sizes: {sizes_str}")
        return self.window_sizes
    
    def choose_custom_elevation_mix(self):
        """Let user mix and match sizes from different elevation levels"""
        print("\n CUSTOM ELEVATION MIX")
        print("Select individual window sizes from any elevation level:")
        print()
        
        # Show all available sizes organized by level
        all_sizes = []
        level_names = []
        
        for level, info in self.elevation_levels.items():
            print(f" {level.upper().replace('_', ' ')} - {info['description']}")
            
            # Add square sizes
            square_sizes = info.get("square_sizes", [])
            for h, w in square_sizes:
                all_sizes.append((h, w))
                level_names.append(f"{level.upper().replace('_', ' ')} (Square)")
                aspect = w / h
                print(f"  {len(all_sizes)}. {h}×{w} (aspect {aspect:.3f}:1) - Square (recommended)")
            
            # Add rectangular sizes if they exist
            if self.use_rectangular_windows:
                rect_sizes = info.get("rectangular_sizes", [])
                for h, w in rect_sizes:
                    all_sizes.append((h, w))
                    level_names.append(f"{level.upper().replace('_', ' ')} (Rectangular)")
                    aspect = w / h
                    print(f"  {len(all_sizes)}. {h}×{w} (aspect {aspect:.3f}:1) - Rectangular")
            print()
        
        chosen_sizes = []
        
        # Get number of sizes to choose
        while True:
            try:
                num_sizes = input(f"How many window sizes do you want? (1-{len(all_sizes)}): ").strip()
                num_sizes = int(num_sizes)
                if 1 <= num_sizes <= len(all_sizes):
                    break
                else:
                    print(f" Please enter a number between 1 and {len(all_sizes)}")
            except ValueError:
                print(" Please enter a valid number")
        
        # Choose individual sizes
        while len(chosen_sizes) < num_sizes:
            try:
                choice = input(f"\nSelect window size #{len(chosen_sizes)+1} (1-{len(all_sizes)}): ").strip()
                idx = int(choice) - 1
                
                if 0 <= idx < len(all_sizes):
                    selected_size = all_sizes[idx]
                    if selected_size not in chosen_sizes:
                        chosen_sizes.append(selected_size)
                        level_name = level_names[idx]
                        h, w = selected_size
                        print(f" Added {h}×{w} from {level_name}")
                    else:
                        print(" Size already selected")
                else:
                    print(f" Please enter a number between 1 and {len(all_sizes)}")
                    
            except ValueError:
                print(" Please enter a valid number")
        
        self.window_sizes = chosen_sizes
        self.current_elevation_level = "custom_mix"
        
        sizes_str = ", ".join([f"{h}×{w}" for h, w in chosen_sizes])
        print(f"\n Custom elevation mix selected: {sizes_str}")
        return self.window_sizes
    
    def choose_all_elevation_levels(self):
        """Use window sizes from all elevation levels"""
        all_sizes = []
        for level, info in self.elevation_levels.items():
            # Add square sizes (always included)
            square_sizes = info.get("square_sizes", [])
            all_sizes.extend(square_sizes)
            
            # Add rectangular sizes if enabled
            if self.use_rectangular_windows:
                rect_sizes = info.get("rectangular_sizes", [])
                all_sizes.extend(rect_sizes)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sizes = []
        for size in all_sizes:
            if size not in seen:
                seen.add(size)
                unique_sizes.append(size)
        
        self.window_sizes = unique_sizes
        self.current_elevation_level = "all_levels"
        
        sizes_str = ", ".join([f"{h}×{w}" for h, w in unique_sizes])
        print(f"\n Using all elevation levels: {sizes_str}")
        print(f"   Total: {len(unique_sizes)} different window sizes")
        print(f"   Covers all drone altitudes from very low to high!")
        return self.window_sizes

    def visualize_cell_grid(self):
        """Visualize the base cell grid on the map"""
        print(f" Visualizing base cell grid ({self.base_cell_size}x{self.base_cell_size} pixels)...")
        
        # Safety check for map image
        if self.map_image is None or self.map_image.size == 0:
            print(" No map image available for grid visualization")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Handle different image formats
        if len(self.map_image.shape) == 3 and self.map_image.shape[2] == 3:
            # RGB or BGR image
            if hasattr(self, '_is_placeholder') and self._is_placeholder:
                # Placeholder is already in RGB format
                ax.imshow(self.map_image)
            else:
                # Real map image is in BGR format
                ax.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale or other format
            ax.imshow(self.map_image, cmap='gray')
        
        # Draw grid lines
        grid_cols = (self.map_width + self.base_cell_size - 1) // self.base_cell_size
        grid_rows = (self.map_height + self.base_cell_size - 1) // self.base_cell_size
        
        # Vertical lines
        for col in range(grid_cols + 1):
            x = col * self.base_cell_size
            if x <= self.map_width:
                ax.axvline(x=x, color='cyan', alpha=0.5, linewidth=0.5)
        
        # Horizontal lines
        for row in range(grid_rows + 1):
            y = row * self.base_cell_size
            if y <= self.map_height:
                ax.axhline(y=y, color='cyan', alpha=0.5, linewidth=0.5)
        
        # Add some cell labels for reference
        for row in range(0, min(10, grid_rows), 2):
            for col in range(0, min(10, grid_cols), 2):
                x = col * self.base_cell_size + self.base_cell_size // 2
                y = row * self.base_cell_size + self.base_cell_size // 2
                ax.text(x, y, f'{row},{col}', ha='center', va='center', 
                       color='yellow', fontsize=8, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        ax.set_title(f'Base Cell Grid Visualization\n'
                    f'Grid: {grid_rows} rows × {grid_cols} columns\n'
                    f'Cell size: {self.base_cell_size}×{self.base_cell_size} pixels', fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        
        # Non-blocking show with error handling
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Brief pause to render
            plt.close('all')  # Close to free memory
        except Exception as e:
            print(f" Visualization display not available: {e}")
            plt.close('all')
        
        print(f" Grid contains {grid_rows * grid_cols:,} base cells")

    def visualize_window_sizes(self):
        """Visualize different window sizes to help user choose optimal sizes"""
        print(" Visualizing different window sizes...")
        
        # Define available window sizes with drone-friendly aspect ratios (height, width)
        available_sizes = [
            # Standard drone aspect ratios
            (3, 4),   # 4:3 landscape aspect ratio
            (6, 8),   # 4:3 landscape aspect ratio (larger)
            (9, 12),  # 4:3 landscape aspect ratio (even larger)
            (9, 16),  # 16:9 landscape aspect ratio (most common drone format)
            (5, 8),   # 8:5 landscape aspect ratio
            (12, 20), # 5:3 landscape aspect ratio
            # Square options for comparison
            (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (12, 12),
            # Portrait rectangles (for different orientations)
            (8, 6), (12, 9), (16, 9)
        ]
        
        grid_cols = (self.map_width + self.base_cell_size - 1) // self.base_cell_size
        grid_rows = (self.map_height + self.base_cell_size - 1) // self.base_cell_size
        
        # Create subplots - 3 windows per row
        n_cols = 3
        n_rows = (len(available_sizes) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'lime']
        
        for idx, (window_rows, window_cols) in enumerate(available_sizes):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            
            if row_idx < len(axes) and col_idx < len(axes[0]):
                ax = axes[row_idx, col_idx]
                ax.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
                
                color = colors[idx % len(colors)]
                
                # Show multiple examples of this window size
                examples_shown = 0
                max_examples = 8
                
                for top_row in range(0, grid_rows - window_rows + 1, max(1, (grid_rows - window_rows) // 4)):
                    for top_col in range(0, grid_cols - window_cols + 1, max(1, (grid_cols - window_cols) // 4)):
                        if examples_shown >= max_examples:
                            break
                        
                        # Calculate window boundaries in pixels
                        x_start = top_col * self.base_cell_size
                        y_start = top_row * self.base_cell_size
                        x_end = min(x_start + window_cols * self.base_cell_size, self.map_width)
                        y_end = min(y_start + window_rows * self.base_cell_size, self.map_height)
                        
                        # Draw rectangle
                        width = x_end - x_start
                        height = y_end - y_start
                        
                        rect = plt.Rectangle((x_start, y_start), width, height,
                                           linewidth=2, edgecolor=color, facecolor=color, alpha=0.2)
                        ax.add_patch(rect)
                        
                        # Add label on first example
                        if examples_shown == 0:
                            center_x = x_start + width // 2
                            center_y = y_start + height // 2
                            ax.text(center_x, center_y, f'{window_rows}×{window_cols}', 
                                   ha='center', va='center', color='white', fontsize=12, weight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
                        
                        examples_shown += 1
                    
                    if examples_shown >= max_examples:
                        break
                
                # Calculate total windows that would be created
                total_windows = 0
                for tr in range(grid_rows - window_rows + 1):
                    for tc in range(grid_cols - window_cols + 1):
                        total_windows += 1
                
                total_with_rotations = total_windows * len(self.rotation_angles)
                
                ax.set_title(f'Window Size: {window_rows}×{window_cols}\n'
                           f'~{total_windows:,} positions × {len(self.rotation_angles)} rotations = {total_with_rotations:,} total', 
                           fontsize=11)
                ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(available_sizes), n_rows * n_cols):
            row_idx = idx // n_cols
            col_idx = idx % n_cols
            if row_idx < len(axes) and col_idx < len(axes[0]):
                axes[row_idx, col_idx].axis('off')
        
        plt.suptitle('Window Size Visualization - Choose Multiple Rectangular Sizes for Drone Video Analysis', fontsize=16)
        plt.tight_layout()
        
        # Non-blocking show with error handling
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Brief pause to render
            plt.close('all')  # Close to free memory
        except Exception as e:
            print(f" Visualization display not available: {e}")
            plt.close('all')
        
        return available_sizes

    def choose_window_sizes(self):
        """Interactive method to choose window sizes with commercial-ready options"""
        print("\n DRONE WINDOW SIZE SELECTION")
        print("=" * 50)
        print(" PERFORMANCE NOTE: Square windows typically achieve 0.7+ confidence")
        print(" WARNING: Rectangular windows may reduce confidence to 0.3")
        print()
        
        print("Choose selection method:")
        print("1.  COMMERCIAL CONFIG - Guided setup (Recommended for commercial use)")
        print("2.  Elevation-based selection")
        print("3.  Manual selection from all available sizes")
        print("4.  Add custom size")
        
        while True:
            try:
                method = input("\nSelect method (1-4): ").strip()
                
                if method == "1":
                    if self.configure_commercial_settings():
                        return self.window_sizes
                    else:
                        continue  # Let user try again
                        
                elif method == "2":
                    return self.choose_elevation_level()
                    
                elif method == "3":
                    return self.choose_manual_sizes()
                    
                elif method == "4":
                    return self.add_custom_drone_ratio_size()
                    
                else:
                    print(" Please enter 1, 2, 3, or 4")
                    
            except ValueError:
                print(" Please enter a valid number")
    
    def choose_manual_sizes(self):
        """Manual selection from all available sizes"""
        print("\n MANUAL SIZE SELECTION")
        print("Available window sizes from all elevation levels:")
        print()
        
        # Collect all sizes from all levels
        all_sizes = []
        size_sources = []
        
        for level, info in self.elevation_levels.items():
            level_name = level.upper().replace('_', ' ')
            
            # Add square sizes
            square_sizes = info.get("square_sizes", [])
            for h, w in square_sizes:
                if (h, w) not in all_sizes:  # Avoid duplicates
                    all_sizes.append((h, w))
                    size_sources.append(f"{level_name} (Square)")
            
            # Add rectangular sizes if enabled
            if self.use_rectangular_windows:
                rect_sizes = info.get("rectangular_sizes", [])
                for h, w in rect_sizes:
                    if (h, w) not in all_sizes:  # Avoid duplicates
                        all_sizes.append((h, w))
                        size_sources.append(f"{level_name} (Rectangular)")
        
        # Display all sizes
        for i, ((h, w), source) in enumerate(zip(all_sizes, size_sources)):
            aspect = w / h
            pixel_area = h * w * (self.base_cell_size ** 2)
            print(f"{i+1:2d}. {h}×{w} (aspect {aspect:.3f}:1) - {source} altitude - {pixel_area:,} pixels")
        
        chosen_sizes = []
        
        # Get number of sizes
        while True:
            try:
                num_sizes = input(f"\nHow many sizes do you want? (1-{len(all_sizes)}): ").strip()
                num_sizes = int(num_sizes)
                if 1 <= num_sizes <= len(all_sizes):
                    break
                else:
                    print(f" Please enter between 1 and {len(all_sizes)}")
            except ValueError:
                print(" Please enter a valid number")
        
        # Choose individual sizes
        while len(chosen_sizes) < num_sizes:
            try:
                choice = input(f"\nSelect size #{len(chosen_sizes)+1} (1-{len(all_sizes)}): ").strip()
                idx = int(choice) - 1
                
                if 0 <= idx < len(all_sizes):
                    selected_size = all_sizes[idx]
                    if selected_size not in chosen_sizes:
                        chosen_sizes.append(selected_size)
                        h, w = selected_size
                        source = size_sources[idx]
                        print(f" Added {h}×{w} ({source})")
                    else:
                        print(" Size already selected")
                else:
                    print(f" Please enter between 1 and {len(all_sizes)}")
            except ValueError:
                print(" Please enter a valid number")
        
        self.window_sizes = chosen_sizes
        self.current_elevation_level = "manual_selection"
        return self.window_sizes
    
    def add_custom_drone_ratio_size(self):
        """Add a custom size that maintains the drone aspect ratio"""
        print("\n ADD CUSTOM DRONE-RATIO SIZE")
        print(f"Creating a window that maintains your drone's {self.drone_aspect_ratio:.3f}:1 aspect ratio")
        print()
        
        while True:
            try:
                height = input("Enter window height in cells (2-30): ").strip()
                height = int(height)
                if 2 <= height <= 30:
                    break
                else:
                    print(" Please enter between 2 and 30")
            except ValueError:
                print(" Please enter a valid number")
        
        # Calculate width to maintain drone aspect ratio
        width = round(height * self.drone_aspect_ratio)
        actual_ratio = width / height
        
        print(f"\n Calculated dimensions:")
        print(f"   Height: {height} cells ({height * self.base_cell_size} pixels)")
        print(f"   Width: {width} cells ({width * self.base_cell_size} pixels)")
        print(f"   Aspect ratio: {actual_ratio:.3f}:1")
        print(f"   Difference from drone ratio: {abs(actual_ratio - self.drone_aspect_ratio):.3f}")
        
        if abs(actual_ratio - self.drone_aspect_ratio) < 0.1:
            print("    Excellent match with drone aspect ratio!")
        else:
            print("    Slight deviation due to rounding to whole cells")
        
        confirm = input(f"\nUse this size ({height}×{width})? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            custom_size = (height, width)
            
            # Add to current sizes or create new list
            if hasattr(self, 'window_sizes') and self.window_sizes:
                if custom_size not in self.window_sizes:
                    self.window_sizes.append(custom_size)
                    print(f" Added {height}×{width} to existing window sizes")
                else:
                    print(" This size already exists")
            else:
                self.window_sizes = [custom_size]
                print(f" Set {height}×{width} as window size")
            
            self.current_elevation_level = "custom_ratio"
            return self.window_sizes
        else:
            print(" Custom size cancelled")
            return self.choose_window_sizes()  # Return to main menu

    def display_elevation_system_info(self):
        """Display comprehensive information about the elevation-based window system"""
        print("\n ELEVATION-BASED WINDOW SIZING SYSTEM")
        print("=" * 55)
        print(" Designed specifically for drone video analysis")
        print(f" All windows maintain your drone's {self.drone_aspect_ratio:.3f}:1 aspect ratio (4000×2250 pixels)")
        print()
        
        print(" ALTITUDE LEVELS:")
        for level, info in self.elevation_levels.items():
            level_name = level.upper().replace('_', ' ')
            altitude_range = info["altitude_range"]
            description = info["description"]
            square_sizes = info.get("square_sizes", [])
            rect_sizes = info.get("rectangular_sizes", [])
            
            print(f"\n {level_name} ALTITUDE ({altitude_range})")
            print(f"   Purpose: {description}")
            print(f"   Square window sizes: {len(square_sizes)} options (recommended)")
            
            for h, w in square_sizes:
                pixel_area = h * w * (self.base_cell_size ** 2)
                aspect = w / h
                print(f"   • {h}×{w} cells → {h * self.base_cell_size}×{w * self.base_cell_size} pixels ({pixel_area:,} total pixels)")
            
            if rect_sizes:
                print(f"   Rectangular window sizes: {len(rect_sizes)} options (optional)")
                for h, w in rect_sizes:
                    pixel_area = h * w * (self.base_cell_size ** 2)
                    aspect = w / h
                    print(f"   • {h}×{w} cells → {h * self.base_cell_size}×{w * self.base_cell_size} pixels ({pixel_area:,} total pixels)")
        
        print(f"\n USAGE RECOMMENDATIONS:")
        print(f"   • Lower altitude = Larger windows (more detail, closer terrain)")
        print(f"   • Higher altitude = Smaller windows (broader coverage, distant terrain)")
        print(f"   • All windows maintain perfect drone aspect ratio for optimal matching")
        print(f"   • Base cell size: {self.base_cell_size}×{self.base_cell_size} pixels")
        
        input("\nPress Enter to continue...")



    def show_selected_sizes_preview(self):
        """Show a preview of the selected window sizes"""
        if not self.window_sizes:
            return
        
        print(f" Previewing selected window sizes: {self.window_sizes}")
        
        # Calculate subplot layout
        num_sizes = len(self.window_sizes)
        if num_sizes <= 2:
            rows, cols = 1, num_sizes
            figsize = (10 * num_sizes, 10)
        elif num_sizes <= 4:
            rows, cols = 2, 2
            figsize = (20, 20)
        elif num_sizes <= 6:
            rows, cols = 2, 3
            figsize = (30, 20)
        else:
            rows = (num_sizes + 2) // 3  # Round up division by 3
            cols = 3
            figsize = (30, 10 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle case where axes is not a list (single subplot)
        if num_sizes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        grid_cols = (self.map_width + self.base_cell_size - 1) // self.base_cell_size
        grid_rows = (self.map_height + self.base_cell_size - 1) // self.base_cell_size
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for idx, window_size in enumerate(self.window_sizes):
            window_rows, window_cols = window_size  # Unpack rectangular dimensions
            ax = axes[idx]
            ax.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
            
            color = colors[idx % len(colors)]
            
            # Show multiple examples
            examples_shown = 0
            max_examples = 12
            
            for top_row in range(0, grid_rows - window_rows + 1, max(1, (grid_rows - window_rows) // 4)):
                for top_col in range(0, grid_cols - window_cols + 1, max(1, (grid_cols - window_cols) // 4)):
                    if examples_shown >= max_examples:
                        break
                    
                    x_start = top_col * self.base_cell_size
                    y_start = top_row * self.base_cell_size
                    x_end = min(x_start + window_cols * self.base_cell_size, self.map_width)
                    y_end = min(y_start + window_rows * self.base_cell_size, self.map_height)
                    
                    width = x_end - x_start
                    height = y_end - y_start
                    
                    rect = plt.Rectangle((x_start, y_start), width, height,
                                       linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
                    ax.add_patch(rect)
                    
                    if examples_shown == 0:
                        center_x = x_start + width // 2
                        center_y = y_start + height // 2
                        aspect_ratio = window_cols / window_rows
                        ax.text(center_x, center_y, f'{window_rows}×{window_cols}\n({aspect_ratio:.2f}:1)', 
                               ha='center', va='center', color='white', fontsize=12, weight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9))
                    
                    examples_shown += 1
                
                if examples_shown >= max_examples:
                    break
            
            # Calculate statistics
            total_positions = (grid_rows - window_rows + 1) * (grid_cols - window_cols + 1)
            total_with_rotations = total_positions * len(self.rotation_angles)
            aspect_ratio = window_cols / window_rows
            
            ax.set_title(f'Selected Size: {window_rows}×{window_cols} (aspect {aspect_ratio:.2f}:1)\n'
                        f'{total_positions:,} positions × {len(self.rotation_angles)} rotations = {total_with_rotations:,} windows', 
                        fontsize=11)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_sizes, len(axes)):
            axes[idx].axis('off')
        
        # Create title showing all selected sizes
        size_descriptions = []
        for rows, cols in self.window_sizes:
            aspect_ratio = cols / rows
            size_descriptions.append(f"{rows}×{cols} ({aspect_ratio:.2f}:1)")
        plt.suptitle(f'Final Selected Window Sizes: {", ".join(size_descriptions)}', fontsize=16)
        plt.tight_layout()
        self._safe_show_plot()

    def extract_embedding_optimized(self, image):
        """Optimized embedding extraction with GPU acceleration and caching"""
        embedding_start_time = time.time()
        
        # Convert image to tensor format efficiently
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Move to device and convert to half precision if enabled
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        if self.enable_half_precision:
            input_tensor = input_tensor.half()

        with torch.no_grad():
            # Disable gradient computation and use optimized inference
            with torch.cuda.amp.autocast(enabled=self.enable_half_precision):
                embedding = self.model(input_tensor)
                
            if len(embedding.shape) > 2:
                embedding = embedding.view(embedding.size(0), -1)
            
            # Convert back to float32 for compatibility
            if self.enable_half_precision:
                embedding = embedding.float()
                
            result = embedding.cpu().numpy().flatten()
            
        # Clear GPU memory immediately
        del input_tensor, embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        embedding_time = time.time() - embedding_start_time
        return result, embedding_time

    def extract_embedding_batch(self, images):
        """Extract embeddings for multiple images in batch for efficiency"""
        if not images:
            return [], 0.0
            
        embedding_start_time = time.time()
        
        # Prepare batch
        batch_tensors = []
        for image in images:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            tensor = self.transform(image)
            batch_tensors.append(tensor)
        
        # Create batch tensor
        batch_tensor = torch.stack(batch_tensors)
        batch_tensor = batch_tensor.to(self.device, non_blocking=True)
        
        if self.enable_half_precision:
            batch_tensor = batch_tensor.half()
        
        # Process batch
        embeddings = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_half_precision):
                batch_embeddings = self.model(batch_tensor)
                
            if len(batch_embeddings.shape) > 2:
                batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
            
            if self.enable_half_precision:
                batch_embeddings = batch_embeddings.float()
                
            embeddings = batch_embeddings.cpu().numpy()
        
        # Cleanup
        del batch_tensor, batch_tensors, batch_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        embedding_time = time.time() - embedding_start_time
        return embeddings, embedding_time

    # Keep the legacy method for compatibility
    def extract_embedding(self, image):
        """Legacy method - redirects to optimized version"""
        result, _ = self.extract_embedding_optimized(image)
        return result

    def create_composite_image(self, cell_images):
        if len(cell_images) == 1:
            return cell_images[0]

        num_cells = len(cell_images)
        grid_size = int(np.sqrt(num_cells))

        if grid_size * grid_size == num_cells:
            rows, cols = grid_size, grid_size
        else:
            for r in range(1, num_cells + 1):
                if num_cells % r == 0:
                    rows, cols = r, num_cells // r
                    break

        cell_h, cell_w = cell_images[0].shape[:2]
        composite = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for i, img in enumerate(cell_images):
            row = i // cols
            col = i % cols
            y_start = row * cell_h
            x_start = col * cell_w

            if img.shape[:2] != (cell_h, cell_w):
                img = cv2.resize(img, (cell_w, cell_h))

            composite[y_start:y_start+cell_h, x_start:x_start+cell_w] = img

        return composite

    def generate_window_image(self, window):
        """Generate composite and rotated image for a window on-demand (memory efficient)"""
        cell_images = []
        for cell in window.cells:
            cell_img = self.map_image[cell.y_start:cell.y_end, cell.x_start:cell.x_end]
            cell_images.append(cell_img)

        composite_image = self.create_composite_image(cell_images)
        rotated_image = self.rotate_image(composite_image, window.rotation_angle)
        return composite_image, rotated_image

    def rotate_image(self, image, angle):
        """Rotate image by given angle (in degrees) around its center"""
        if angle == 0:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))

        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        return rotated

    def process_map(self):
        print(f" Processing map with ALL window sizes 3x3 to 15x15 ({len(self.window_sizes)} sizes)...")
        print(f"   Window sizes: {', '.join([f'{h}x{w}' for h, w in self.window_sizes])}")
        self.print_memory_stats("at start")

        # Check if we already have preprocessed windows loaded
        if hasattr(self, 'multi_cell_windows') and len(self.multi_cell_windows) > 0 and hasattr(self, 'index') and self.index is not None:
            print(" Using already loaded preprocessed windows")
            print(f" Total searchable windows: {len(self.multi_cell_windows):,}")
            return

        # Create grid
        grid_cols = (self.map_width + self.base_cell_size - 1) // self.base_cell_size
        grid_rows = (self.map_height + self.base_cell_size - 1) // self.base_cell_size

        # Create base cells
        self.cells = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                x_start = col * self.base_cell_size
                y_start = row * self.base_cell_size
                x_end = min(x_start + self.base_cell_size, self.map_width)
                y_end = min(y_start + self.base_cell_size, self.map_height)

                center_x = (x_start + x_end) / 2
                center_y = (y_start + y_end) / 2

                cell = CellInfo(
                    row=row, col=col,
                    x_start=x_start, y_start=y_start,
                    x_end=x_end, y_end=y_end,
                    center_x=center_x, center_y=center_y
                )
                self.cells.append(cell)

        # Create multi-cell windows with rotations
        self.multi_cell_windows = []
        window_counts = {}

        for window_size in self.window_sizes:
            window_rows, window_cols = window_size  # Now expects (rows, cols) tuple
            for angle in self.rotation_angles:
                key = f"{window_rows}x{window_cols}@{angle}°"
                window_counts[key] = 0

            for top_row in range(grid_rows - window_rows + 1):
                for top_col in range(grid_cols - window_cols + 1):
                    window_cells = []
                    for r in range(window_rows):
                        for c in range(window_cols):
                            cell_idx = (top_row + r) * grid_cols + (top_col + c)
                            if cell_idx < len(self.cells):
                                window_cells.append(self.cells[cell_idx])

                    if len(window_cells) == window_rows * window_cols:
                        x_starts = [cell.x_start for cell in window_cells]
                        y_starts = [cell.y_start for cell in window_cells]
                        x_ends = [cell.x_end for cell in window_cells]
                        y_ends = [cell.y_end for cell in window_cells]

                        x_start = min(x_starts)
                        y_start = min(y_starts)
                        x_end = max(x_ends)
                        y_end = max(y_ends)

                        center_x = (x_start + x_end) / 2
                        center_y = (y_start + y_end) / 2
                        scale_factor = max(window_rows, window_cols)  # Use max dimension for scale

                        for angle in self.rotation_angles:
                            window = MultiCellWindow(
                                top_left_row=top_row, top_left_col=top_col,
                                window_rows=window_rows, window_cols=window_cols,
                                cells=window_cells,
                                x_start=x_start, y_start=y_start,
                                x_end=x_end, y_end=y_end,
                                center_x=center_x, center_y=center_y,
                                scale_factor=scale_factor,
                                rotation_angle=angle
                            )
                            self.multi_cell_windows.append(window)
                            window_counts[f"{window_rows}x{window_cols}@{angle}°"] += 1

        print(f" Created {len(self.cells):,} base cells")
        for window_rows, window_cols in self.window_sizes:
            size_total = sum(count for key, count in window_counts.items() if key.startswith(f"{window_rows}x{window_cols}"))
            print(f" Created {size_total:,} windows of size {window_rows}x{window_cols} (all rotations)")
        print(f" Total large windows: {len(self.multi_cell_windows):,}")
        self.print_memory_stats("after creating windows")

        # Optimized embedding computation with batch processing
        start_time = time.time()
        print("Computing embeddings with batch processing...")

        # Process windows in batches for better GPU utilization
        batch_size = self.batch_size
        total_batches = (len(self.multi_cell_windows) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.multi_cell_windows))
            batch_windows = self.multi_cell_windows[start_idx:end_idx]
            
            # Generate images for batch
            batch_images = []
            for window in batch_windows:
                _, rotated_image = self.generate_window_image(window)
                batch_images.append(rotated_image)
            
            # Extract embeddings in batch
            embeddings, embedding_time = self.extract_embedding_batch(batch_images)
            
            # Assign embeddings to windows
            for i, embedding in enumerate(embeddings):
                batch_windows[i].embedding = embedding
            
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"   Processing batches: {batch_idx + 1}/{total_batches} ({progress:.1f}%)")

        # Build optimized FAISS index
        print("Building optimized FAISS index...")
        self.build_optimized_faiss_index()
        
        # Build all_info for compatibility - ONLY include windows with embeddings
        # to match FAISS index structure
        self.all_info = []
        for window in self.multi_cell_windows:
            if window.embedding is not None:
                self.all_info.append(('window', window))

        elapsed = time.time() - start_time
        print(f"Map processing completed in {elapsed/60:.1f} minutes!")
        print(f"Total searchable windows: {len(self.all_info):,}")
        
        # Save preprocessed windows
        self.save_preprocessed_windows()
        
        # Force garbage collection to free memory
        gc.collect()
        print(f"Memory cleanup completed")

    def build_optimized_faiss_index(self):
        """Build optimized FAISS index with GPU support if available"""
        print("Building optimized FAISS index...")
        
        all_embeddings = []
        for window in self.multi_cell_windows:
            if window.embedding is not None:
                all_embeddings.append(window.embedding)
        
        if not all_embeddings:
            print("No embeddings available for FAISS index")
            return
        
        all_embeddings = np.array(all_embeddings)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        
        dimension = all_embeddings.shape[1]
        
        if self.enable_gpu_faiss and torch.cuda.is_available():
            try:
                # Use GPU FAISS for faster search
                res = faiss.StandardGpuResources()  # Use default GPU
                
                # Create CPU index first
                cpu_index = faiss.IndexFlatIP(dimension)
                
                # Move to GPU
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self.index.add(all_embeddings.astype('float32'))
                
                print(f"   GPU FAISS index created with {len(all_embeddings):,} vectors")
                
            except Exception as e:
                print(f"   GPU FAISS failed, using CPU: {e}")
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(all_embeddings.astype('float32'))
        else:
            # Use CPU FAISS
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(all_embeddings.astype('float32'))
            print(f"   CPU FAISS index created with {len(all_embeddings):,} vectors")

    def cleanup_memory(self):
        """Clean up memory by removing stored data"""
        
        # Clear large data structures
        if hasattr(self, 'multi_cell_windows'):
            for window in self.multi_cell_windows:
                window.embedding = None
        
        if hasattr(self, 'cells'):
            for cell in self.cells:
                cell.embedding = None
        
        # Clear trajectory data
        self.trajectory = []
        self.previous_location = None
        
        # Clear model from GPU memory if using CUDA
        if hasattr(self, 'model') and torch.cuda.is_available():
            del self.model
            torch.cuda.empty_cache()
        
        # Clean up preprocessed windows if not saving for reuse
        self.cleanup_preprocessed_windows()
        
        # Force garbage collection
        gc.collect()
        print("Memory cleanup completed - freed several GBs!")

    def print_timing_statistics(self):
        """Print detailed timing statistics for frame processing"""
        if not self.frame_timings:
            print("No timing statistics available")
            return
        
        print("\n" + "="*50)
        print("PERFORMANCE TIMING ANALYSIS")
        print("="*50)
        
        # Calculate averages
        avg_total = np.mean([t.total_time for t in self.frame_timings])
        avg_embedding = np.mean([t.embedding_time for t in self.frame_timings])
        avg_matching = np.mean([t.matching_time for t in self.frame_timings])
        avg_display = np.mean([t.display_time for t in self.frame_timings])
        
        print(f"Frames processed: {len(self.frame_timings)}")
        print(f"\nTIMING BREAKDOWN (per frame):")
        print(f"  Total time:      {avg_total:.4f}s")
        print(f"  - Embedding:     {avg_embedding:.4f}s ({avg_embedding/avg_total*100:.1f}%)")
        print(f"  - Matching:      {avg_matching:.4f}s ({avg_matching/avg_total*100:.1f}%)")
        print(f"  - Display:       {avg_display:.4f}s ({avg_display/avg_total*100:.1f}%)")
        
        print(f"\nPROCESSING RATE:")
        print(f"  Frames per second: {1/avg_total:.2f} FPS")
        print(f"  Embedding FPS:     {1/avg_embedding:.2f} FPS")
        print(f"  Matching FPS:      {1/avg_matching:.2f} FPS")
        
        # Find slowest operations
        slowest_total = max(self.frame_timings, key=lambda x: x.total_time)
        slowest_embedding = max(self.frame_timings, key=lambda x: x.embedding_time)
        slowest_matching = max(self.frame_timings, key=lambda x: x.matching_time)
        
        print(f"\nSLOWEST OPERATIONS:")
        print(f"  Slowest total:     {slowest_total.total_time:.4f}s")
        print(f"  Slowest embedding: {slowest_embedding.embedding_time:.4f}s")
        print(f"  Slowest matching:  {slowest_matching.matching_time:.4f}s")
        
        print("="*50)

    def get_nearby_windows(self, center_x, center_y, radius):
        """Get windows within radius of given center point"""
        nearby_indices = []
        for i, (item_type, window) in enumerate(self.all_info):
            distance = np.sqrt((window.center_x - center_x)**2 + (window.center_y - center_y)**2)
            if distance <= radius:
                nearby_indices.append(i)
        return nearby_indices

    def localize_frame_optimized(self, frame_image, frame_number, timestamp, k=25):
        """Optimized frame localization with detailed timing"""
        
        # Initialize timing
        timing = FrameTimings()
        frame_start_time = time.time()
        
        # Safety check
        if self.index is None:
            print(f"  Error: FAISS index not initialized")
            return None, [], timing
        
        # 1. EMBEDDING EXTRACTION (Tembeddings)
        embedding_start_time = time.time()
        frame_embedding, embedding_time = self.extract_embedding_optimized(frame_image)
        frame_embedding = frame_embedding / np.linalg.norm(frame_embedding)
        timing.embedding_time = time.time() - embedding_start_time
        
        # 2. MATCHING PROCESS (Tmatching)
        matching_start_time = time.time()
        
        use_global_search = False
        
        # Determine search strategy
        if self.previous_location is not None:
            if self.previous_location.confidence < self.min_confidence_for_temporal:
                use_global_search = True
            else:
                confidence_factor = 1.0 - self.previous_location.confidence
                adaptive_radius = self.search_radius + (confidence_factor * self.max_search_radius)
                adaptive_radius = min(adaptive_radius, self.max_search_radius)
        else:
            use_global_search = True
            adaptive_radius = self.search_radius
        
        # Execute search strategy
        if use_global_search:
            # Global search using FAISS
            similarities, indices = self.index.search(
                frame_embedding.reshape(1, -1).astype('float32'), k
            )
            top_similarities = similarities[0]
            top_indices_global = indices[0]
            
            # Debug: Check if indices are valid
            if frame_number == 0:  # Only print for first frame to avoid spam
                print(f"   DEBUG: FAISS returned {len(top_indices_global)} indices")
                print(f"   DEBUG: all_info size: {len(self.all_info)}")
                print(f"   DEBUG: Sample indices: {top_indices_global[:5]}")
                print(f"   DEBUG: Index ntotal: {self.index.ntotal}")
            
        else:
            # Local search around previous location
            nearby_indices = self.get_nearby_windows(
                self.previous_location.x, 
                self.previous_location.y, 
                adaptive_radius
            )
            
            if len(nearby_indices) < 15:
                # Fallback to global search
                similarities, indices = self.index.search(
                    frame_embedding.reshape(1, -1).astype('float32'), k
                )
                top_similarities = similarities[0]
                top_indices_global = indices[0]
            else:
                # Local search
                nearby_embeddings = np.array([self.all_info[i][1].embedding for i in nearby_indices])
                nearby_embeddings = nearby_embeddings / np.linalg.norm(nearby_embeddings, axis=1, keepdims=True)
                
                similarities = np.dot(nearby_embeddings, frame_embedding)
                top_k_local = min(k, len(similarities))
                top_indices_local = np.argsort(similarities)[::-1][:top_k_local]
                
                top_similarities = similarities[top_indices_local]
                top_indices_global = [nearby_indices[i] for i in top_indices_local]
        
        timing.matching_time = time.time() - matching_start_time
        
        # 3. RESULT PROCESSING (minimal - Tdisplayingresults)
        display_start_time = time.time()
        
        # Create matches - filter out invalid indices (-1)
        matches = []
        for sim, idx in zip(top_similarities, top_indices_global):
            # Skip invalid indices from FAISS
            if idx < 0 or idx >= len(self.all_info):
                continue
            item_type, info = self.all_info[idx]
            matches.append({
                'similarity': sim,
                'x': info.center_x,
                'y': info.center_y,
                'type': item_type,
                'scale': info.scale_factor,
                'rotation': info.rotation_angle,
                'window': info
            })

        # Compute final position using weighted average
        if matches:
            top_matches = matches[:min(5, len(matches))]  # Use top 5 for speed
            
            weights = [m['similarity'] for m in top_matches]
            total_weight = sum(weights)
            
            if total_weight > 0:
                estimated_x = sum(w * m['x'] for w, m in zip(weights, top_matches)) / total_weight
                estimated_y = sum(w * m['y'] for w, m in zip(weights, top_matches)) / total_weight
                estimated_rotation = sum(w * m['rotation'] for w, m in zip(weights, top_matches)) / total_weight
                
                # Apply temporal smoothing if applicable (lightweight)
                if (self.previous_location is not None and 
                    self.previous_location.confidence > self.min_confidence_for_temporal and
                    not use_global_search):
                    
                    estimated_x = (1 - self.temporal_weight) * estimated_x + self.temporal_weight * self.previous_location.x
                    estimated_y = (1 - self.temporal_weight) * estimated_y + self.temporal_weight * self.previous_location.y
                    estimated_rotation = (1 - self.temporal_weight) * estimated_rotation + self.temporal_weight * self.previous_location.rotation
                
                confidence = matches[0]['similarity']
            else:
                estimated_x = matches[0]['x']
                estimated_y = matches[0]['y']
                estimated_rotation = matches[0]['rotation']
                confidence = matches[0]['similarity']
        else:
            estimated_x, estimated_y = 0, 0
            estimated_rotation = 0
            confidence = 0

        timing.display_time = time.time() - display_start_time
        timing.total_time = time.time() - frame_start_time
        
        # Store timing statistics
        self.frame_timings.append(timing)
        self.total_embedding_time += timing.embedding_time
        self.total_matching_time += timing.matching_time
        self.total_display_time += timing.display_time

        trajectory_point = TrajectoryPoint(
            frame_number=frame_number,
            timestamp=timestamp,
            x=estimated_x,
            y=estimated_y,
            rotation=estimated_rotation,
            confidence=confidence,
            window_info=matches[0]['window'] if matches else None
        )

        self.previous_location = trajectory_point
        return trajectory_point, matches, timing

    # Keep the legacy method for compatibility
    def localize_frame(self, frame_image, frame_number, timestamp, k=25):
        """Legacy method - redirects to optimized version"""
        result, matches, timing = self.localize_frame_optimized(frame_image, frame_number, timestamp, k)
        return result, matches

    def analyze_video_from_file(self, video_path=None):
        """Analyze video from local file"""
        if video_path is None:
            print(" Please select your drone video file...")
            video_path = filedialog.askopenfilename(
                title="Select Drone Video",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                    ("All files", "*.*")
                ]
            )

        if not video_path or not os.path.exists(video_path):
            raise ValueError("No video file selected or file doesn't exist")

        filename = os.path.basename(video_path)
        print(f" Processing video: {filename}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f" Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

        self.trajectory = []
        self.previous_location = None

        frame_number = 0
        start_time = time.time()
        frame_processing_times = []  # Track processing time for each frame

        frame_skip = max(1, int(fps // 10))
        print(f" Processing every {frame_skip} frame(s) for efficiency")

        # Live visualization and frame saving are enabled by default from main()
        if self.enable_live_map_visualization:
            print("\n Live map visualization ENABLED - watch the drone move on the map!")
            print("   Press 'q' on the map window to stop")
        
        if self.save_frames_with_positions:
            # Create output directory
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            self.frame_save_dir = self.create_frame_save_directory(video_basename)
            print(f"\n Frame saving ENABLED - saving every {self.frame_save_interval} frame(s)")
            print(f"   Output directory: {self.frame_save_dir}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip == 0:
                timestamp = frame_number / fps
                
                print(f" Processing frame {frame_number}/{total_frames} (t={timestamp:.2f}s)")
                
                try:
                    # Start timing for this frame
                    frame_start_time = time.time()
                    
                    trajectory_point, matches = self.localize_frame(frame, frame_number, timestamp)
                    
                    # End timing for this frame
                    frame_end_time = time.time()
                    frame_processing_time = frame_end_time - frame_start_time
                    frame_processing_times.append(frame_processing_time)
                    
                    self.trajectory.append(trajectory_point)
                    
                    print(f"   Position: ({trajectory_point.x:.1f}, {trajectory_point.y:.1f})")
                    print(f"   Rotation: {trajectory_point.rotation:.1f}°, Confidence: {trajectory_point.confidence:.3f}")
                    print(f"   ⏱ Frame processing time: {frame_processing_time:.3f}s")
                    
                    # Save frame with position if enabled
                    if self.save_frames_with_positions and (len(self.trajectory) % self.frame_save_interval == 0):
                        self.save_frame_with_position(frame, trajectory_point, frame_number, self.frame_save_dir)
                    
                    # Update live visualization
                    if self.enable_live_map_visualization and len(self.trajectory) % self.live_viz_update_interval == 0:
                        self.update_live_map_visualization(
                            current_frame_image=frame,
                            trajectory_point=trajectory_point,
                            processing_time=frame_processing_time,
                            frame_number=frame_number
                        )
                        
                        # Check for quit key on map window
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n User requested stop (pressed 'q' on map window)")
                            break
                    
                except Exception as e:
                    print(f"    Error processing frame {frame_number}: {e}")

            frame_number += 1

        cap.release()
        cv2.destroyAllWindows()  # Close all windows including live map visualization
        self.finalize_live_visualization()  # Clean up video writer and visualization resources
        
        # Generate summary report if frames were saved
        if self.save_frames_with_positions and self.frame_save_dir:
            print("\n Generating summary report...")
            self.generate_summary_report(self.frame_save_dir)
            
            print(f"\n " + "=" * 70)
            print(f" FRAMES SAVED TO: {self.frame_save_dir}")
            print(f" " + "=" * 70)
            print(f"   📁 frames/     - {len(list((self.frame_save_dir / 'frames').glob('*.jpg')))} original camera frames")
            print(f"   📁 positions/  - {len(list((self.frame_save_dir / 'positions').glob('*.jpg')))} position maps")
            print(f"   📁 combined/   - {len(list((self.frame_save_dir / 'combined').glob('*.jpg')))} combined views")
            print(f"   📄 SUMMARY_REPORT.txt - Detailed statistics and information")
            print(f" " + "=" * 70)
        
        processing_time = time.time() - start_time
        
        # Calculate timing statistics
        if frame_processing_times:
            avg_frame_time = np.mean(frame_processing_times)
            min_frame_time = np.min(frame_processing_times)
            max_frame_time = np.max(frame_processing_times)
            total_frame_time = np.sum(frame_processing_times)
            
            print(f" Video analysis completed in {processing_time:.1f} seconds!")
            print(f" Processed {len(self.trajectory)} trajectory points")
            print(f"⏱ Frame Processing Time Statistics:")
            print(f"    Average time per frame: {avg_frame_time:.3f}s")
            print(f"    Fastest frame: {min_frame_time:.3f}s")
            print(f"    Slowest frame: {max_frame_time:.3f}s")
            print(f"    Total frame processing time: {total_frame_time:.1f}s")
            print(f"    Overhead time (I/O, visualization, etc.): {processing_time - total_frame_time:.1f}s")
            print(f"    Processing speed: {len(self.trajectory) / total_frame_time:.2f} frames/second")
        else:
            print(f" Video analysis completed in {processing_time:.1f} seconds!")
            print(f" Processed {len(self.trajectory)} trajectory points")

        return self.trajectory

    def analyze_camera_live_feed(self, camera_port=0, target_fps=10, max_duration=None):
        """Analyze live camera feed from Raspberry Pi camera or other camera
        
        Args:
            camera_port: Camera port number (default 0 for Raspberry Pi camera on Jetson)
            target_fps: Target frames per second to process (default 10)
            max_duration: Maximum duration in seconds (None for unlimited)
        """
        print("\n LIVE CAMERA FEED ANALYSIS")
        print("=" * 60)
        print(f"Attempting to open camera at port {camera_port}...")
        
        # Detect if running on NVIDIA Jetson
        is_jetson = os.path.exists('/etc/nv_tegra_release')
        
        if is_jetson:
            print(" Detected NVIDIA Jetson board - using GStreamer pipeline...")
            
            # GStreamer pipeline for Jetson with CSI camera
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={camera_port} ! "
                f"video/x-raw(memory:NVMM), "
                f"width=(int)1280, height=(int)720, "
                f"format=(string)NV12, framerate=(fraction){target_fps}/1 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, width=(int)1280, height=(int)720, "
                f"format=(string)BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=(string)BGR ! "
                f"appsink drop=1 max-buffers=1"
            )
            
            print(" GStreamer pipeline configured for IMX219 camera")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                print(" WARNING: GStreamer pipeline failed, trying direct V4L2...")
                cap = cv2.VideoCapture(camera_port, cv2.CAP_V4L2)
        else:
            # Open camera with V4L2 backend for better Raspberry Pi camera support
            cap = cv2.VideoCapture(camera_port, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print(f" WARNING: Could not open camera with V4L2 backend, trying default...")
            cap = cv2.VideoCapture(camera_port)
        
        if not cap.isOpened():
            print(f" ERROR: Could not open camera at port {camera_port}")
            print("Please check:")
            print("  - Camera is properly connected")
            print("  - Camera permissions are set correctly")
            print("  - No other application is using the camera")
            if is_jetson:
                print("  - Try: ls /dev/video*")
                print("  - Check camera with: gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! nvoverlaysink")
            else:
                print("  - Try running: sudo chmod 666 /dev/video0")
            return None
        
        # Configure camera for better performance (skip for GStreamer)
        if not is_jetson or not gst_pipeline:
            print(" Configuring camera settings...")
            
            # Reduce resolution for faster processing (adjust as needed)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Set buffer size to 1 to get latest frame
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set FPS if possible
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            
            # Disable auto-focus and auto-exposure for consistent frames
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Read and discard first few frames to allow camera to stabilize
            print(" Warming up camera (reading initial frames)...")
            for i in range(10):
                ret, _ = cap.read()
                if not ret:
                    print(f"   Warning: Failed to read warmup frame {i+1}/10")
                time.sleep(0.1)
            
            # Clear any buffered frames
            for i in range(5):
                cap.grab()
        
        # Clear any buffered frames
        for i in range(5):
            cap.grab()
        
        # Get actual camera properties after configuration
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f" Camera opened and configured successfully!")
        print(f"   Resolution: {frame_width}x{frame_height}")
        print(f"   Camera FPS: {camera_fps:.1f}" if camera_fps > 0 else "   Camera FPS: Unknown")
        print(f"   Processing target: {target_fps} FPS")
        
        # Calculate frame skip for target FPS
        if camera_fps > 0 and camera_fps > target_fps:
            frame_skip = max(1, int(camera_fps / target_fps))
        else:
            frame_skip = 1
        
        print(f"   Processing every {frame_skip} frame(s)")
        
        # Initialize tracking variables
        self.trajectory = []
        self.previous_location = None
        
        frame_number = 0
        processed_frames = 0
        failed_reads = 0
        max_failed_reads = 30  # Stop after 30 consecutive failures
        start_time = time.time()
        frame_processing_times = []
        
        # Live visualization and frame saving are enabled by default from main()
        if self.enable_live_map_visualization:
            print("\n Live map visualization ENABLED - watch the drone move on the map!")
            print("   Press 'q' on either window to stop")
        
        if self.save_frames_with_positions:
            # Create output directory
            self.frame_save_dir = self.create_frame_save_directory("camera_live")
            print(f"\n Frame saving ENABLED - saving every {self.frame_save_interval} frame(s)")
            print(f"   Output directory: {self.frame_save_dir}")
        
        print("\n LIVE LOCALIZATION STARTED")
        print("Press 'q' to stop, 's' to save current trajectory, 'r' to reset trajectory")
        print("-" * 60)
        
        try:
            while True:
                # Check duration limit
                if max_duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed > max_duration:
                        print(f"\n Maximum duration ({max_duration}s) reached. Stopping...")
                        break
                
                # Read frame with timeout handling
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        print(f"\n ERROR: Failed to read {max_failed_reads} consecutive frames. Camera may be disconnected.")
                        break
                    elif failed_reads % 5 == 0:
                        print(f" Warning: Failed to read frame (attempt {failed_reads}/{max_failed_reads})")
                    
                    # Try to clear buffer and continue
                    time.sleep(0.05)
                    cap.grab()
                    continue
                
                # Reset failed read counter on successful read
                failed_reads = 0
                
                # Process frame at target rate
                if frame_number % frame_skip == 0:
                    timestamp = time.time() - start_time
                    
                    try:
                        # Start timing for this frame
                        frame_start_time = time.time()
                        
                        # Localize the frame
                        trajectory_point, matches = self.localize_frame(frame, processed_frames, timestamp)
                        
                        # End timing for this frame
                        frame_end_time = time.time()
                        frame_processing_time = frame_end_time - frame_start_time
                        frame_processing_times.append(frame_processing_time)
                        
                        self.trajectory.append(trajectory_point)
                        
                        # Print status every 10 processed frames or if confidence is low
                        if processed_frames % 10 == 0 or trajectory_point.confidence < 0.5:
                            print(f"[Frame {processed_frames:4d}] t={timestamp:6.1f}s | "
                                  f"Pos: ({trajectory_point.x:6.1f}, {trajectory_point.y:6.1f}) | "
                                  f"Rot: {trajectory_point.rotation:5.1f}° | "
                                  f"Conf: {trajectory_point.confidence:.3f} | "
                                  f"Time: {frame_processing_time:.3f}s")
                        
                        # Save frame with position if enabled
                        if self.save_frames_with_positions and (processed_frames % self.frame_save_interval == 0):
                            self.save_frame_with_position(frame, trajectory_point, processed_frames, self.frame_save_dir)
                        
                        # Update live map visualization
                        if self.enable_live_map_visualization and processed_frames % self.live_viz_update_interval == 0:
                            self.update_live_map_visualization(
                                current_frame_image=frame,
                                trajectory_point=trajectory_point,
                                processing_time=frame_processing_time,
                                frame_number=processed_frames
                            )
                        
                        processed_frames += 1
                        
                    except Exception as e:
                        print(f"\n ERROR processing frame {processed_frames}: {e}")
                
                frame_number += 1
                
                # Optional: Display live feed with overlay (can be disabled for headless mode)
                # This requires X server or display output
                try:
                    # Create display frame with localization info
                    display_frame = frame.copy()
                    
                    if self.previous_location:
                        # Draw position text
                        info_text = f"Pos: ({self.previous_location.x:.0f}, {self.previous_location.y:.0f}) | Conf: {self.previous_location.confidence:.3f}"
                        cv2.putText(display_frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw FPS
                        if frame_processing_times:
                            current_fps = 1.0 / frame_processing_times[-1] if frame_processing_times[-1] > 0 else 0
                            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Live Camera Feed - Drone Localization', display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n User requested stop (pressed 'q')")
                        break
                    elif key == ord('s'):
                        # Save current trajectory
                        temp_filename = f"live_trajectory_{int(time.time())}.csv"
                        self.export_trajectory(temp_filename)
                        print(f"\n Trajectory saved to {temp_filename}")
                    elif key == ord('r'):
                        # Reset trajectory
                        self.trajectory = []
                        self.previous_location = None
                        processed_frames = 0
                        print("\n Trajectory reset")
                        
                except Exception as e:
                    # Headless mode - display not available
                    pass
                
        except KeyboardInterrupt:
            print("\n\n Interrupted by user (Ctrl+C)")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.finalize_live_visualization()  # Clean up video writer and visualization resources
            
            # Generate summary report if frames were saved
            if self.save_frames_with_positions and self.frame_save_dir:
                print("\n Generating summary report...")
                self.generate_summary_report(self.frame_save_dir)
                
                print(f"\n " + "=" * 70)
                print(f" FRAMES SAVED TO: {self.frame_save_dir}")
                print(f" " + "=" * 70)
                print(f"   📁 frames/     - {len(list((self.frame_save_dir / 'frames').glob('*.jpg')))} original camera frames")
                print(f"   📁 positions/  - {len(list((self.frame_save_dir / 'positions').glob('*.jpg')))} position maps")
                print(f"   📁 combined/   - {len(list((self.frame_save_dir / 'combined').glob('*.jpg')))} combined views")
                print(f"   📄 SUMMARY_REPORT.txt - Detailed statistics and information")
                print(f" " + "=" * 70)
            
            # Calculate and display statistics
            processing_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print(" LIVE CAMERA ANALYSIS COMPLETED")
            print("=" * 60)
            
            if frame_processing_times:
                avg_frame_time = np.mean(frame_processing_times)
                min_frame_time = np.min(frame_processing_times)
                max_frame_time = np.max(frame_processing_times)
                total_frame_time = np.sum(frame_processing_times)
                
                print(f" Total duration: {processing_time:.1f} seconds")
                print(f" Frames captured: {frame_number}")
                print(f" Frames processed: {processed_frames}")
                print(f" Trajectory points: {len(self.trajectory)}")
                print(f"\n Frame Processing Time Statistics:")
                print(f"    Average time per frame: {avg_frame_time:.3f}s ({1/avg_frame_time:.2f} FPS)")
                print(f"    Fastest frame: {min_frame_time:.3f}s")
                print(f"    Slowest frame: {max_frame_time:.3f}s")
                print(f"    Total processing time: {total_frame_time:.1f}s")
                print(f"    Real-time factor: {processing_time/total_frame_time:.2f}x")
                
                if self.trajectory:
                    confidences = [p.confidence for p in self.trajectory]
                    print(f"\n Confidence Statistics:")
                    print(f"    Average: {np.mean(confidences):.3f}")
                    print(f"    Min: {np.min(confidences):.3f}")
                    print(f"    Max: {np.max(confidences):.3f}")
            else:
                print(f" No frames were processed")
            
            print("=" * 60)
        
        return self.trajectory

    def test_random_frames_from_file(self, video_path=None, num_frames=70):
        """Test random frames from local video file"""
        if video_path is None:
            print(" Please select your drone video file for random frame testing...")
            video_path = filedialog.askopenfilename(
                title="Select Drone Video for Testing",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                    ("All files", "*.*")
                ]
            )

        if not video_path or not os.path.exists(video_path):
            raise ValueError("No video file selected or file doesn't exist")

        filename = os.path.basename(video_path)
        print(f" Processing video for random testing: {filename}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f" Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

        random_indices = np.random.choice(total_frames, min(num_frames, total_frames), replace=False)
        random_indices = sorted(random_indices)
        
        print(f" Selected {len(random_indices)} random frames for testing")

        test_results = []
        
        for i, frame_idx in enumerate(random_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f" Could not read frame {frame_idx}")
                continue
                
            timestamp = frame_idx / fps
            print(f" Testing frame {i+1}/{len(random_indices)} (frame #{frame_idx}, t={timestamp:.2f}s)")
            
            try:
                temp_prev_location = self.previous_location
                self.previous_location = None
                
                trajectory_point, matches = self.localize_frame(frame, frame_idx, timestamp)
                
                self.previous_location = temp_prev_location
                
                test_results.append({
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'frame_image': frame.copy(),
                    'trajectory_point': trajectory_point,
                    'matches': matches[:5]
                })
                
                print(f"   Position: ({trajectory_point.x:.1f}, {trajectory_point.y:.1f})")
                print(f"   Confidence: {trajectory_point.confidence:.3f}")
                
            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")

        cap.release()
        print(f" Random frame testing completed! Processed {len(test_results)} frames")
        
        return test_results

    def visualize_random_frame_results(self, test_results, frames_per_figure=10):
        """Create visualizations showing frames alongside their localization results"""
        if not test_results:
            print(" No test results to visualize")
            return

        num_figures = (len(test_results) + frames_per_figure - 1) // frames_per_figure
        
        for fig_idx in range(num_figures):
            start_idx = fig_idx * frames_per_figure
            end_idx = min(start_idx + frames_per_figure, len(test_results))
            current_results = test_results[start_idx:end_idx]
            
            fig = plt.figure(figsize=(20, 4 * len(current_results)))
            
            for i, result in enumerate(current_results):
                frame_image = result['frame_image']
                trajectory_point = result['trajectory_point']
                matches = result['matches']
                frame_idx = result['frame_idx']
                timestamp = result['timestamp']
                
                # Left subplot: Original drone frame
                ax1 = plt.subplot(len(current_results), 2, 2*i + 1)
                frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
                plt.imshow(frame_rgb)
                plt.title(f'Frame #{frame_idx} (t={timestamp:.2f}s)\nDrone View')
                plt.axis('off')
                
                # Right subplot: Map with localization
                ax2 = plt.subplot(len(current_results), 2, 2*i + 2)
                plt.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
                
                # Plot estimated position
                plt.plot(trajectory_point.x, trajectory_point.y, 'ro', markersize=12, 
                        label=f'Estimated Position')
                plt.plot(trajectory_point.x, trajectory_point.y, 'r+', markersize=15, markeredgewidth=3)
                
                # Draw rotation arrow
                arrow_length = 50
                arrow_x = trajectory_point.x + arrow_length * np.cos(np.radians(trajectory_point.rotation))
                arrow_y = trajectory_point.y + arrow_length * np.sin(np.radians(trajectory_point.rotation))
                plt.arrow(trajectory_point.x, trajectory_point.y, 
                         arrow_x - trajectory_point.x, arrow_y - trajectory_point.y,
                         head_width=20, head_length=15, fc='red', ec='red', alpha=0.8)
                
                # Plot top matches
                colors = ['green', 'orange', 'purple', 'blue', 'cyan']
                for j, match in enumerate(matches):
                    color = colors[j % len(colors)]
                    alpha = 0.7 - j * 0.1
                    size = 8 - j
                    plt.plot(match['x'], match['y'], 'o', color=color, markersize=size, 
                            alpha=alpha, label=f'Match {j+1} ({match["similarity"]:.3f})')
                
                plt.title(f'Localization Result\nPos: ({trajectory_point.x:.0f}, {trajectory_point.y:.0f}), '
                         f'Rot: {trajectory_point.rotation:.0f}°, Conf: {trajectory_point.confidence:.3f}')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plt.axis('off')
            
            plt.suptitle(f'Random Frame Testing Results - Set {fig_idx + 1}/{num_figures}', 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            self._safe_show_plot()

    def create_summary_visualization(self, test_results):
        """Create a summary visualization showing all tested frames on the map"""
        if not test_results:
            return
            
        plt.figure(figsize=(16, 12))
        plt.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
        
        x_coords = [r['trajectory_point'].x for r in test_results]
        y_coords = [r['trajectory_point'].y for r in test_results]
        confidences = [r['trajectory_point'].confidence for r in test_results]
        timestamps = [r['timestamp'] for r in test_results]
        
        scatter = plt.scatter(x_coords, y_coords, c=confidences, cmap='viridis', 
                            s=100, alpha=0.8, edgecolors='white', linewidth=2)
        plt.colorbar(scatter, label='Confidence', shrink=0.8)
        
        for i, (x, y, t) in enumerate(zip(x_coords, y_coords, timestamps)):
            plt.annotate(f'F{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.title(f'Random Frame Testing Summary\n'
                 f'{len(test_results)} frames tested, Avg confidence: {np.mean(confidences):.3f}')
        plt.axis('off')
        plt.tight_layout()
        self._safe_show_plot()
        
        print(f"\n Random Frame Testing Statistics:")
        print(f"   Frames tested: {len(test_results)}")
        print(f"   Average confidence: {np.mean(confidences):.3f}")
        print(f"   Min confidence: {np.min(confidences):.3f}")
        print(f"   Max confidence: {np.max(confidences):.3f}")
        print(f"   Std deviation: {np.std(confidences):.3f}")
        
        high_conf = sum(1 for c in confidences if c > 0.8)
        med_conf = sum(1 for c in confidences if 0.6 <= c <= 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        print(f"   High confidence (>0.8): {high_conf} frames ({high_conf/len(confidences)*100:.1f}%)")
        print(f"   Medium confidence (0.6-0.8): {med_conf} frames ({med_conf/len(confidences)*100:.1f}%)")
        print(f"   Low confidence (<0.6): {low_conf} frames ({low_conf/len(confidences)*100:.1f}%)")

    def visualize_trajectory(self, show_confidence=True, show_arrows=True):
        """Visualize the complete drone trajectory on the map"""
        if not self.trajectory:
            print(" No trajectory data to visualize")
            return

        plt.figure(figsize=(16, 12))
        plt.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))

        x_coords = [point.x for point in self.trajectory]
        y_coords = [point.y for point in self.trajectory]
        confidences = [point.confidence for point in self.trajectory]
        timestamps = [point.timestamp for point in self.trajectory]

        if show_confidence:
            scatter = plt.scatter(x_coords, y_coords, c=confidences, cmap='viridis', 
                                s=30, alpha=0.8, edgecolors='white', linewidth=1)
            plt.colorbar(scatter, label='Confidence')
        else:
            plt.plot(x_coords, y_coords, 'r-', linewidth=2, alpha=0.8)
            plt.scatter(x_coords, y_coords, c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1)

        if len(self.trajectory) > 0:
            start_point = self.trajectory[0]
            end_point = self.trajectory[-1]
            
            plt.plot(start_point.x, start_point.y, 'go', markersize=15, label=f'Start (t={start_point.timestamp:.1f}s)')
            plt.plot(end_point.x, end_point.y, 'ro', markersize=15, label=f'End (t={end_point.timestamp:.1f}s)')

        if show_arrows and len(self.trajectory) > 10:
            arrow_interval = max(1, len(self.trajectory) // 10)
            for i in range(0, len(self.trajectory), arrow_interval):
                point = self.trajectory[i]
                arrow_length = 40
                arrow_x = point.x + arrow_length * np.cos(np.radians(point.rotation))
                arrow_y = point.y + arrow_length * np.sin(np.radians(point.rotation))
                
                plt.arrow(point.x, point.y, arrow_x - point.x, arrow_y - point.y,
                         head_width=15, head_length=10, fc='blue', ec='blue', alpha=0.6)

        plt.title(f'Local Drone Video Trajectory Analysis\n'
                 f'Total points: {len(self.trajectory)}, Duration: {timestamps[-1]:.1f}s\n'
                 f'Window sizes: {self.window_sizes}, Search radius: {self.search_radius}px')
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        self._safe_show_plot()

        self._show_trajectory_stats()

    def _show_trajectory_stats(self):
        """Show detailed trajectory statistics"""
        if len(self.trajectory) < 2:
            return

        print(f"\n Trajectory Analysis:")
        print(f"   Total points: {len(self.trajectory)}")
        print(f"   Duration: {self.trajectory[-1].timestamp:.1f} seconds")
        
        distances = []
        for i in range(1, len(self.trajectory)):
            prev = self.trajectory[i-1]
            curr = self.trajectory[i]
            dist = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
            distances.append(dist)

        if distances:
            print(f"   Average speed: {np.mean(distances):.1f} pixels/frame")
            print(f"   Max speed: {np.max(distances):.1f} pixels/frame")
            print(f"   Total distance: {np.sum(distances):.1f} pixels")

        confidences = [p.confidence for p in self.trajectory]
        print(f"   Average confidence: {np.mean(confidences):.3f}")
        print(f"   Min confidence: {np.min(confidences):.3f}")
        print(f"   Max confidence: {np.max(confidences):.3f}")

        window_sizes_used = [p.window_info.scale_factor for p in self.trajectory if p.window_info]
        if window_sizes_used:
            from collections import Counter
            size_counts = Counter(window_sizes_used)
            print(f"   Window sizes used: {dict(size_counts)}")

    def update_live_map_visualization(self, current_frame_image=None, trajectory_point=None, 
                                     processing_time=0, frame_number=0):
        """
        Update live visualization showing drone position on map with real-time trajectory
        
        Args:
            current_frame_image: Current camera frame (optional, shown in corner)
            trajectory_point: Latest trajectory point
            processing_time: Time taken to process this frame
            frame_number: Current frame number
        """
        if not self.enable_live_map_visualization:
            return
        
        try:
            # Create a copy of the map for visualization
            map_viz = self.map_image.copy()
            
            # Draw trajectory trail (recent points)
            if len(self.trajectory) > 1:
                trail_points = self.trajectory[-self.live_viz_trail_length:]
                
                # Draw trajectory line
                for i in range(1, len(trail_points)):
                    prev = trail_points[i-1]
                    curr = trail_points[i]
                    
                    # Color based on confidence (green = high, red = low)
                    confidence_color = self._get_confidence_color(curr.confidence)
                    
                    # Draw line segment
                    cv2.line(map_viz, 
                            (int(prev.x), int(prev.y)),
                            (int(curr.x), int(curr.y)),
                            confidence_color, 2)
                    
                    # Draw small circle at each point
                    cv2.circle(map_viz, (int(curr.x), int(curr.y)), 3, confidence_color, -1)
            
            # Draw current position with larger marker
            if trajectory_point:
                x, y = int(trajectory_point.x), int(trajectory_point.y)
                
                # Draw pulsing circle effect for current position
                cv2.circle(map_viz, (x, y), 20, (0, 255, 255), 2)  # Yellow outer circle
                cv2.circle(map_viz, (x, y), 10, (0, 0, 255), -1)    # Red filled circle
                cv2.circle(map_viz, (x, y), 5, (255, 255, 255), -1) # White center
                
                # Draw rotation arrow
                arrow_length = 50
                arrow_x = int(x + arrow_length * np.cos(np.radians(trajectory_point.rotation)))
                arrow_y = int(y + arrow_length * np.sin(np.radians(trajectory_point.rotation)))
                cv2.arrowedLine(map_viz, (x, y), (arrow_x, arrow_y), (255, 0, 255), 3, tipLength=0.3)
                
                # Draw info box near current position
                info_y = max(50, y - 80)
                info_x = max(10, x - 100)
                
                # Background for text
                cv2.rectangle(map_viz, (info_x - 5, info_y - 35), (info_x + 250, info_y + 45), (0, 0, 0), -1)
                cv2.rectangle(map_viz, (info_x - 5, info_y - 35), (info_x + 250, info_y + 45), (255, 255, 255), 2)
                
                # Position text
                cv2.putText(map_viz, f"Pos: ({x}, {y})", (info_x, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(map_viz, f"Rot: {trajectory_point.rotation:.1f}deg", (info_x, info_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(map_viz, f"Conf: {trajectory_point.confidence:.3f}", (info_x, info_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._get_confidence_color(trajectory_point.confidence), 2)
            
            # Draw start position marker
            if len(self.trajectory) > 0:
                start = self.trajectory[0]
                cv2.circle(map_viz, (int(start.x), int(start.y)), 15, (0, 255, 0), 3)  # Green circle
                cv2.putText(map_viz, "START", (int(start.x) - 25, int(start.y) - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw statistics overlay (top-left corner)
            stats_y = 30
            stats_x = 10
            
            # Background for stats
            cv2.rectangle(map_viz, (stats_x - 5, stats_y - 25), (stats_x + 350, stats_y + 120), (0, 0, 0), -1)
            cv2.rectangle(map_viz, (stats_x - 5, stats_y - 25), (stats_x + 350, stats_y + 120), (0, 255, 255), 2)
            
            # Stats text
            cv2.putText(map_viz, f"LIVE DRONE LOCALIZATION", (stats_x, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(map_viz, f"Frame: {frame_number} | Points: {len(self.trajectory)}", (stats_x, stats_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if trajectory_point:
                cv2.putText(map_viz, f"Time: {trajectory_point.timestamp:.1f}s | FPS: {1/processing_time:.1f}", 
                           (stats_x, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Distance traveled
                if len(self.trajectory) > 1:
                    total_dist = sum([
                        np.sqrt((self.trajectory[i].x - self.trajectory[i-1].x)**2 + 
                               (self.trajectory[i].y - self.trajectory[i-1].y)**2)
                        for i in range(1, len(self.trajectory))
                    ])
                    cv2.putText(map_viz, f"Distance: {total_dist:.1f}px", (stats_x, stats_y + 75), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Average confidence
                avg_conf = np.mean([p.confidence for p in self.trajectory])
                cv2.putText(map_viz, f"Avg Conf: {avg_conf:.3f}", (stats_x, stats_y + 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._get_confidence_color(avg_conf), 2)
            
            # Optionally show current camera frame in corner (picture-in-picture)
            if current_frame_image is not None:
                pip_height = 150
                pip_width = int(pip_height * current_frame_image.shape[1] / current_frame_image.shape[0])
                pip_frame = cv2.resize(current_frame_image, (pip_width, pip_height))
                
                # Position in top-right corner
                pip_y = 10
                pip_x = map_viz.shape[1] - pip_width - 10
                
                # Draw PIP frame
                map_viz[pip_y:pip_y+pip_height, pip_x:pip_x+pip_width] = pip_frame
                
                # Border around PIP
                cv2.rectangle(map_viz, (pip_x-2, pip_y-2), (pip_x+pip_width+2, pip_y+pip_height+2), 
                             (255, 255, 255), 2)
                cv2.putText(map_viz, "Camera View", (pip_x, pip_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw legend (bottom-left)
            legend_y = map_viz.shape[0] - 100
            legend_x = 10
            
            cv2.rectangle(map_viz, (legend_x - 5, legend_y - 5), (legend_x + 200, legend_y + 95), (0, 0, 0), -1)
            cv2.rectangle(map_viz, (legend_x - 5, legend_y - 5), (legend_x + 200, legend_y + 95), (255, 255, 255), 1)
            
            cv2.putText(map_viz, "Legend:", (legend_x, legend_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(map_viz, (legend_x + 10, legend_y + 35), 5, (0, 255, 0), -1)
            cv2.putText(map_viz, "Start", (legend_x + 20, legend_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(map_viz, (legend_x + 10, legend_y + 55), 5, (0, 0, 255), -1)
            cv2.putText(map_viz, "Current", (legend_x + 20, legend_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.line(map_viz, (legend_x + 5, legend_y + 75), (legend_x + 15, legend_y + 75), (0, 255, 255), 2)
            cv2.putText(map_viz, "Trail", (legend_x + 20, legend_y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display the visualization
            cv2.imshow(self.live_viz_window_name, map_viz)
            cv2.waitKey(1)  # Brief pause to update display
            
            # Optionally save visualization to video file
            if self.live_viz_save_video:
                if self.live_viz_video_writer is None:
                    # Initialize video writer on first frame
                    output_path = "live_localization_visualization.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 10  # Output video FPS
                    frame_size = (map_viz.shape[1], map_viz.shape[0])
                    self.live_viz_video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                    print(f" Started recording visualization to: {output_path}")
                
                # Write frame to video
                self.live_viz_video_writer.write(map_viz)
            
        except Exception as e:
            # Silently fail if visualization not available (headless mode)
            if frame_number == 0:  # Only print once
                print(f" Live map visualization not available: {e}")
                self.enable_live_map_visualization = False

    def _get_confidence_color(self, confidence):
        """Get BGR color based on confidence level (green=high, red=low)"""
        if confidence > 0.7:
            return (0, 255, 0)  # Green
        elif confidence > 0.5:
            return (0, 255, 255)  # Yellow
        elif confidence > 0.3:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red

    def finalize_live_visualization(self):
        """Clean up live visualization resources"""
        try:
            # Close video writer if active
            if self.live_viz_video_writer is not None:
                self.live_viz_video_writer.release()
                print(" Visualization video saved successfully")
                self.live_viz_video_writer = None
            
            # Close visualization window
            cv2.destroyWindow(self.live_viz_window_name)
            
        except Exception as e:
            pass  # Silently ignore errors during cleanup

    def create_frame_save_directory(self, video_name="run"):
        """Create a unique directory for saving frames from this run
        
        Args:
            video_name: Base name for the directory
            
        Returns:
            Path to the created directory
        """
        from datetime import datetime
        
        # Create base directory for all runs
        base_dir = Path("frame_outputs")
        base_dir.mkdir(exist_ok=True)
        
        # Create unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / f"{video_name}_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (run_dir / "frames").mkdir(exist_ok=True)  # Original frames
        (run_dir / "positions").mkdir(exist_ok=True)  # Position on map
        (run_dir / "combined").mkdir(exist_ok=True)  # Combined view
        
        # Create info file
        info_file = run_dir / "run_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Drone Localization Run\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Video/Source: {video_name}\n")
            f.write(f"Map size: {self.map_width}x{self.map_height}\n")
            f.write(f"Window sizes: {self.window_sizes}\n")
            f.write(f"Frame save interval: {self.frame_save_interval}\n")
            f.write(f"\n")
        
        print(f" Created output directory: {run_dir}")
        print(f"   - frames/: Original camera frames")
        print(f"   - positions/: Position marked on map")
        print(f"   - combined/: Side-by-side view")
        
        return run_dir

    def save_frame_with_position(self, frame_image, trajectory_point, frame_number, run_dir):
        """Save the processed frame along with its position on the map
        
        Args:
            frame_image: The original camera frame
            trajectory_point: TrajectoryPoint with position info
            frame_number: Frame number for naming
            run_dir: Directory to save files
        """
        try:
            # 1. Save original frame
            if self.save_original_frame:
                frame_path = run_dir / "frames" / f"frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(frame_path), frame_image)
            
            # 2. Create and save position visualization on map
            if self.save_position_on_map:
                map_viz = self.map_image.copy()
                
                # Draw all trajectory points so far (mini trail)
                if len(self.trajectory) > 1:
                    recent_points = self.trajectory[-20:]  # Last 20 points
                    for i in range(1, len(recent_points)):
                        prev = recent_points[i-1]
                        curr = recent_points[i]
                        color = self._get_confidence_color(curr.confidence)
                        cv2.line(map_viz, 
                                (int(prev.x), int(prev.y)),
                                (int(curr.x), int(curr.y)),
                                color, 2)
                
                # Draw start position
                if len(self.trajectory) > 0:
                    start = self.trajectory[0]
                    cv2.circle(map_viz, (int(start.x), int(start.y)), 10, (0, 255, 0), -1)
                    cv2.putText(map_viz, "START", (int(start.x) - 25, int(start.y) - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw current position
                x, y = int(trajectory_point.x), int(trajectory_point.y)
                cv2.circle(map_viz, (x, y), 15, (0, 0, 255), 3)  # Red outer circle
                cv2.circle(map_viz, (x, y), 8, (0, 0, 255), -1)  # Red filled
                cv2.circle(map_viz, (x, y), 3, (255, 255, 255), -1)  # White center
                
                # Draw direction arrow
                arrow_length = 40
                arrow_x = int(x + arrow_length * np.cos(np.radians(trajectory_point.rotation)))
                arrow_y = int(y + arrow_length * np.sin(np.radians(trajectory_point.rotation)))
                cv2.arrowedLine(map_viz, (x, y), (arrow_x, arrow_y), (255, 0, 255), 2, tipLength=0.3)
                
                # Add info text overlay
                info_bg_height = 120
                overlay = map_viz.copy()
                cv2.rectangle(overlay, (10, 10), (400, info_bg_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, map_viz, 0.3, 0, map_viz)
                
                # Frame info
                cv2.putText(map_viz, f"Frame: {frame_number}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(map_viz, f"Position: ({x}, {y})", (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(map_viz, f"Rotation: {trajectory_point.rotation:.1f}deg", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                conf_color = self._get_confidence_color(trajectory_point.confidence)
                cv2.putText(map_viz, f"Confidence: {trajectory_point.confidence:.3f}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
                
                # Save position map
                position_path = run_dir / "positions" / f"position_{frame_number:06d}.jpg"
                cv2.imwrite(str(position_path), map_viz)
            
            # 3. Create and save combined view (frame + position side by side)
            if self.save_original_frame and self.save_position_on_map:
                # Resize frame to match map height for side-by-side
                target_height = min(800, self.map_height)  # Limit height for reasonable file size
                
                # Resize map
                map_aspect = self.map_width / self.map_height
                map_display_width = int(target_height * map_aspect)
                map_resized = cv2.resize(map_viz, (map_display_width, target_height))
                
                # Resize frame
                frame_aspect = frame_image.shape[1] / frame_image.shape[0]
                frame_display_width = int(target_height * frame_aspect)
                frame_resized = cv2.resize(frame_image, (frame_display_width, target_height))
                
                # Add labels
                cv2.putText(frame_resized, "Camera Frame", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(map_resized, "Position on Map", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # Combine side by side
                combined = np.hstack([frame_resized, map_resized])
                
                # Save combined view
                combined_path = run_dir / "combined" / f"combined_{frame_number:06d}.jpg"
                cv2.imwrite(str(combined_path), combined)
            
        except Exception as e:
            print(f"   Warning: Could not save frame {frame_number}: {e}")

    def generate_summary_report(self, run_dir):
        """Generate a summary report with statistics and sample images
        
        Args:
            run_dir: Directory containing the run data
        """
        try:
            from datetime import datetime
            
            report_path = run_dir / "SUMMARY_REPORT.txt"
            
            with open(report_path, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("DRONE LOCALIZATION RUN - SUMMARY REPORT\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Run Directory: {run_dir.name}\n\n")
                
                # Trajectory statistics
                f.write("TRAJECTORY STATISTICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total frames processed: {len(self.trajectory)}\n")
                
                if self.trajectory:
                    f.write(f"Duration: {self.trajectory[-1].timestamp:.2f} seconds\n")
                    
                    # Confidence stats
                    confidences = [p.confidence for p in self.trajectory]
                    f.write(f"Average confidence: {np.mean(confidences):.3f}\n")
                    f.write(f"Min confidence: {np.min(confidences):.3f}\n")
                    f.write(f"Max confidence: {np.max(confidences):.3f}\n\n")
                    
                    # Distance stats
                    distances = []
                    for i in range(1, len(self.trajectory)):
                        prev = self.trajectory[i-1]
                        curr = self.trajectory[i]
                        dist = np.sqrt((curr.x - prev.x)**2 + (curr.y - prev.y)**2)
                        distances.append(dist)
                    
                    if distances:
                        f.write(f"Total distance traveled: {np.sum(distances):.1f} pixels\n")
                        f.write(f"Average speed: {np.mean(distances):.2f} pixels/frame\n")
                        f.write(f"Max speed: {np.max(distances):.2f} pixels/frame\n\n")
                    
                    # Position range
                    xs = [p.x for p in self.trajectory]
                    ys = [p.y for p in self.trajectory]
                    f.write(f"Position range:\n")
                    f.write(f"  X: [{np.min(xs):.1f}, {np.max(xs):.1f}]\n")
                    f.write(f"  Y: [{np.min(ys):.1f}, {np.max(ys):.1f}]\n\n")
                
                # File counts
                f.write("OUTPUT FILES\n")
                f.write("-" * 70 + "\n")
                
                frames_dir = run_dir / "frames"
                positions_dir = run_dir / "positions"
                combined_dir = run_dir / "combined"
                
                if frames_dir.exists():
                    frame_count = len(list(frames_dir.glob("*.jpg")))
                    f.write(f"Original frames: {frame_count} files in frames/\n")
                
                if positions_dir.exists():
                    pos_count = len(list(positions_dir.glob("*.jpg")))
                    f.write(f"Position maps: {pos_count} files in positions/\n")
                
                if combined_dir.exists():
                    comb_count = len(list(combined_dir.glob("*.jpg")))
                    f.write(f"Combined views: {comb_count} files in combined/\n")
                
                f.write("\n")
                f.write("=" * 70 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 70 + "\n")
            
            print(f" Generated summary report: {report_path}")
            
        except Exception as e:
            print(f"   Warning: Could not generate summary report: {e}")


    def export_trajectory(self, filename="drone_trajectory.csv"):
        """Export trajectory to CSV file"""
        if not self.trajectory:
            print(" No trajectory data to export")
            return

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'timestamp', 'x', 'y', 'rotation', 'confidence', 'window_size']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for point in self.trajectory:
                writer.writerow({
                    'frame': point.frame_number,
                    'timestamp': point.timestamp,
                    'x': point.x,
                    'y': point.y,
                    'rotation': point.rotation,
                    'confidence': point.confidence,
                    'window_size': point.window_info.scale_factor if point.window_info else 'unknown'
                })
        
        print(f" Trajectory exported to {filename}")

    def adjust_movement_freedom(self, search_radius=None, max_search_radius=None, 
                              min_confidence_for_temporal=None, temporal_weight=None):
        """Adjust movement freedom parameters during runtime"""
        if search_radius is not None:
            self.search_radius = search_radius
            print(f" Search radius updated to: {search_radius}px")
        
        if max_search_radius is not None:
            self.max_search_radius = max_search_radius
            print(f" Max search radius updated to: {max_search_radius}px")
        
        if min_confidence_for_temporal is not None:
            self.min_confidence_for_temporal = min_confidence_for_temporal
            print(f" Min confidence for temporal coherence updated to: {min_confidence_for_temporal}")
        
        if temporal_weight is not None:
            self.temporal_weight = temporal_weight
            print(f" Temporal smoothing weight updated to: {temporal_weight}")
        
        print(f"\n Current Movement Freedom Settings:")
        print(f"   Search radius: {self.search_radius}px")
        print(f"   Max search radius: {self.max_search_radius}px")
        print(f"   Min confidence for temporal: {self.min_confidence_for_temporal}")
        print(f"   Temporal weight: {self.temporal_weight}")

    def run_local_analysis(self):
        """Main function to run local analysis with streamlined choices"""
        print("\n Choose Analysis Mode:")
        print("1. Full Video Trajectory Analysis")
        print("2. Live Camera Feed Analysis (Camera on port 0, 10 FPS)")
        print("3. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    print("\n Running Full Video Trajectory Analysis...")
                    trajectory = self.analyze_video_from_file()
                    if trajectory:
                        print("\n Creating trajectory visualization...")
                        self.visualize_trajectory()
                        print("\n Exporting trajectory data...")
                        self.export_trajectory()
                    break
                    
                elif choice == "2":
                    print("\n LIVE CAMERA FEED ANALYSIS")
                    print("=" * 50)
                    print("Using camera port 0 at 10 FPS (default settings)")
                    
                    print("\n Starting live camera analysis...")
                    print("NOTE: Press 'q' to stop, 's' to save, 'r' to reset trajectory")
                    
                    trajectory = self.analyze_camera_live_feed(
                        camera_port=0,
                        target_fps=10,
                        max_duration=None
                    )
                    
                    if trajectory:
                        print("\n Creating trajectory visualization...")
                        self.visualize_trajectory()
                        print("\n Exporting trajectory data...")
                        export_name = f"camera_trajectory_{int(time.time())}.csv"
                        self.export_trajectory(export_name)
                    break
                    
                elif choice == "3":
                    print(" Goodbye!")
                    break
                    
                else:
                    print(" Invalid choice. Please enter 1-3.")
                    
            except KeyboardInterrupt:
                print("\n Analysis cancelled")
                break
            except Exception as e:
                print(f" Error: {e}")
                break

    def manage_preprocessed_windows(self):
        """Manage existing preprocessed windows"""
        print("\n PREPROCESSED WINDOWS MANAGEMENT")
        print("=" * 50)
        
        # List existing preprocessed windows
        windows_files = list(self.preprocessed_windows_dir.glob("windows_*.pkl"))
        metadata_files = list(self.preprocessed_windows_dir.glob("metadata_*.json"))
        
        if not windows_files:
            print(" No preprocessed windows found")
            return
        
        print(f" Found {len(windows_files)} preprocessed window file(s):")
        
        total_size_mb = 0
        valid_files = []
        
        for windows_file in windows_files:
            hash_id = windows_file.stem.replace("windows_", "")
            metadata_file = self.preprocessed_windows_dir / f"metadata_{hash_id}.json"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
                    total_size_mb += file_size_mb
                    created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_at']))
                    
                    print(f"\n File: {windows_file.name}")
                    print(f"   Size: {file_size_mb:.1f} MB")
                    print(f"   Windows: {metadata['total_windows']:,}")
                    print(f"   Created: {created_time}")
                    print(f"   Window sizes: {metadata['window_sizes']}")
                    print(f"   Map size: {metadata['map_width']}x{metadata['map_height']}")
                    
                    valid_files.append((windows_file, metadata_file, metadata))
                    
                except Exception as e:
                    print(f"\n Error reading {windows_file.name}: {e}")
        
        print(f"\n Total storage used: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        
        if not valid_files:
            return
        
        print(f"\n What would you like to do?")
        print("1. Delete all preprocessed windows")
        print("2. Delete specific files")
        print("3. Keep all files")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            confirm = input(f" Delete ALL {len(valid_files)} files ({total_size_mb:.1f} MB)? [y/N]: ").strip().lower()
            if confirm == 'y':
                deleted_size = 0
                for windows_file, metadata_file, _ in valid_files:
                    try:
                        deleted_size += os.path.getsize(windows_file) / (1024 * 1024)
                        os.remove(windows_file)
                        os.remove(metadata_file)
                    except Exception as e:
                        print(f" Error deleting {windows_file.name}: {e}")
                
                print(f" Deleted all preprocessed windows ({deleted_size:.1f} MB freed)")
            else:
                print(" Cancelled")
        
        elif choice == "2":
            print(f"\n Select files to delete (enter numbers separated by spaces):")
            for i, (windows_file, _, metadata) in enumerate(valid_files):
                file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
                print(f"{i+1}. {windows_file.name} ({file_size_mb:.1f} MB)")
            
            try:
                selections = input("\nEnter file numbers: ").strip().split()
                deleted_size = 0
                
                for sel in selections:
                    idx = int(sel) - 1
                    if 0 <= idx < len(valid_files):
                        windows_file, metadata_file, _ = valid_files[idx]
                        try:
                            deleted_size += os.path.getsize(windows_file) / (1024 * 1024)
                            os.remove(windows_file)
                            os.remove(metadata_file)
                            print(f" Deleted {windows_file.name}")
                        except Exception as e:
                            print(f" Error deleting {windows_file.name}: {e}")
                    else:
                        print(f" Invalid selection: {sel}")
                
                if deleted_size > 0:
                    print(f" Freed {deleted_size:.1f} MB of storage")
                
            except (ValueError, IndexError):
                print(" Invalid input")
        
        elif choice == "3":
            print(" Keeping all preprocessed windows")
        
        else:
            print(" Invalid choice")

def main():
    """Main function to run the local drone localizer with streamlined workflow"""
    print("STREAMLINED Drone Video Trajectory Localizer")
    print("=" * 60)
    print("Required packages: torch, torchvision, opencv-python, pillow, faiss-cpu, matplotlib, numpy, tkinter")
    print("Running locally in VS Code")
    print("AUTO-MODE: Using ALL window sizes 3x3 to 15x15")
    print("Live visualization and frame saving ENABLED by default")
    print()

    # Initialize the localizer (automatically uses ALL window sizes 3x3 to 15x15)
    localizer = LocalVideoTrajectoryDroneLocalizer(
        base_cell_size=32,
        rotation_angles=[0, 45, 90, 135, 180, 225, 270, 315]
    )
    
    # Enable live visualization and frame saving by default
    localizer.enable_live_map_visualization = True
    localizer.save_frames_with_positions = True
    localizer.frame_save_interval = 1

    print("\nStep 1: Using default search radius settings")
    print(f"   Search radius: {localizer.search_radius}px")
    print(f"   Max search radius: {localizer.max_search_radius}px")
    print(f"   Temporal weight: {localizer.temporal_weight}")

    print("\nStep 2: Loading map image...")
    preprocessed_loaded = False
    try:
        map_result = localizer.load_map_from_file()
        if map_result == "preprocessed_map_loaded":
            preprocessed_loaded = True
            print(" Preprocessed map loaded successfully - skipping window generation")
    except Exception as e:
        print(f"Error loading map: {e}")
        return

    print("\nStep 3: Visualizing cell grid...")
    try:
        localizer.visualize_cell_grid()
    except Exception as e:
        print(f"Error visualizing grid: {e}")
        return

    print(f"\nStep 4: AUTO-SELECTED window sizes:")
    print(f"   Using ALL {len(localizer.window_sizes)} window sizes: {', '.join([f'{h}x{w}' for h, w in localizer.window_sizes])}")
    print(f"   No user selection needed - processing comprehensive range!")

    # Only check for preprocessed windows if we didn't already load them in Step 2
    if not preprocessed_loaded:
        print("\nStep 5: Checking for preprocessed windows...")
        try:
            # Check if we can use preprocessed windows
            windows_loaded = localizer.ask_about_preprocessed_windows(localizer.current_map_path)
            
            if not windows_loaded:
                print("\nStep 6: Processing map with ALL window sizes (3x3 to 15x15)...")
                print("   This may take a few minutes for comprehensive coverage...")
                localizer.process_map()
            else:
                print("\nStep 6: Using preprocessed windows (skipped processing)")
        except Exception as e:
            print(f"Error with preprocessed windows: {e}")
            return
    else:
        print("\nStep 5: Skipped (preprocessed map already loaded in Step 2)")
        print("\nStep 6: Skipped (using preprocessed windows from Step 2)")

    print("\nStep 7: Choose analysis mode...")
    try:
        localizer.run_local_analysis()
    finally:
        # Cleanup if user chose not to save windows
        if not localizer.save_windows_for_reuse:
            print("\nFinal cleanup (user chose not to save windows)...")
            localizer.cleanup_preprocessed_windows()

if __name__ == "__main__":
    main()