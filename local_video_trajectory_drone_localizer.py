# Local Video Trajectory Drone Localizer for VS Code with Window Size Visualization
# Install required packages: pip install torch torchvision opencv-python pillow faiss-cpu matplotlib numpy tkinter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import faiss
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

class LocalVideoTrajectoryDroneLocalizer:
    def __init__(self, base_cell_size=32, rotation_angles=[0, 45, 90, 135, 180, 225, 270, 315]):
        self.base_cell_size = base_cell_size
        
        # Drone aspect ratio for elevation-based window sizing
        self.drone_aspect_ratio = 4000 / 2250  # Standard drone video aspect ratio
        
        # Elevation levels for altitude-based window sizing
        self.elevation_levels = {
            'so_low': {
                'name': 'So Low',
                'altitude_range': '5-15m',
                'description': 'Ultra-close details, small features',
                'window_sizes': [(3, 3), (4, 4), (5, 5)],
                'emoji': '🔍'
            },
            'low': {
                'name': 'Low',
                'altitude_range': '15-50m',
                'description': 'Buildings, roads, close structures',
                'window_sizes': [(6, 6), (7, 7), (8, 8)],
                'emoji': '🏢'
            },
            'mid': {
                'name': 'Mid',
                'altitude_range': '50-150m',
                'description': 'Neighborhoods, large structures',
                'window_sizes': [(9, 9), (10, 10), (11, 11)],
                'emoji': '🏘️'
            },
            'high': {
                'name': 'High',
                'altitude_range': '150m+',
                'description': 'Wide areas, landscapes',
                'window_sizes': [(12, 12), (13, 13), (14, 14), (15, 15)],
                'emoji': '🌄'
            }
        }
        
        # SIMPLIFIED: Use ALL window sizes from smallest to largest
        # Generate comprehensive range from 3x3 to 15x15
        self.window_sizes = []
        for size in range(3, 16):  # 3x3 to 15x15
            self.window_sizes.append((size, size))  # Square windows only for simplicity
        
        print(f"🎯 Using ALL window sizes from 3x3 to 15x15: {len(self.window_sizes)} sizes")
        print(f"   Window sizes: {', '.join([f'{h}x{w}' for h, w in self.window_sizes])}")
        self.rotation_angles = rotation_angles
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load MobileNet-V3
        print(f"Loading MobileNet-V3 on {self.device}...")
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Identity()
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        
        # Preprocessed windows management - ALWAYS SAVE for reuse
        self.preprocessed_windows_dir = Path("drone_map_windows")  # Descriptive folder name
        self.preprocessed_windows_dir.mkdir(exist_ok=True)
        self.current_map_hash = None
        self.windows_file_path = None
        self.save_windows_for_reuse = True  # Always save for efficiency

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if not PSUTIL_AVAILABLE:
            return 0  # Return 0 if psutil is not available
        
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb

    def print_memory_stats(self, stage=""):
        """Print current memory usage"""
        if not PSUTIL_AVAILABLE:
            print(f"💾 Memory monitoring not available (install psutil: pip install psutil)")
            return
        
        try:
            memory_mb = self.get_memory_usage()
            print(f"💾 Memory usage {stage}: {memory_mb:.1f} MB ({memory_mb/1024:.2f} GB)")
        except Exception:
            print("💾 Memory monitoring error")

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
            
            # Save windows data
            with open(windows_file, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata = {
                'window_sizes': self.window_sizes,
                'rotation_angles': self.rotation_angles,
                'base_cell_size': self.base_cell_size,
                'map_width': self.map_width,
                'map_height': self.map_height,
                'total_windows': len(self.multi_cell_windows),
                'created_at': time.time(),
                'map_hash': self.current_map_hash
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            file_size_mb = os.path.getsize(windows_file) / (1024 * 1024)
            print(f"✅ Preprocessed windows saved to {windows_file.name}")
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Windows count: {len(self.multi_cell_windows):,}")
            
        except Exception as e:
            print(f"⚠️ Error saving preprocessed windows: {e}")

    def load_preprocessed_windows(self, windows_file, metadata_file):
        """Load preprocessed windows from disk"""
        try:
            # Load and verify metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Set map dimensions from metadata for compatibility check
            expected_map_width = metadata['map_width']
            expected_map_height = metadata['map_height']
            
            # Verify compatibility - Convert both to same format for comparison
            expected_window_sizes = [tuple(size) for size in metadata['window_sizes']]
            current_window_sizes = [tuple(size) for size in self.window_sizes]
            
            if (expected_window_sizes != current_window_sizes or
                metadata['rotation_angles'] != self.rotation_angles or
                metadata['base_cell_size'] != self.base_cell_size):
                print("⚠️ Preprocessed windows are incompatible with current settings")
                print(f"   Expected window_sizes: {metadata['window_sizes']}")
                print(f"   Current window_sizes: {self.window_sizes}")
                print(f"   Expected rotation_angles: {metadata['rotation_angles']}")
                print(f"   Current rotation_angles: {self.rotation_angles}")
                print(f"   Expected base_cell_size: {metadata['base_cell_size']}")
                print(f"   Current base_cell_size: {self.base_cell_size}")
                return False
            
            # Load windows data
            print(f"📂 Loading preprocessed windows from {windows_file.name}...")
            with open(windows_file, 'rb') as f:
                save_data = pickle.load(f)
            
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
            print("🔄 Rebuilding FAISS index...")
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
            
            # Set map dimensions from metadata
            self.map_width = expected_map_width
            self.map_height = expected_map_height
            
            # Set map hash and file path for future use
            self.current_map_hash = metadata['map_hash']
            self.windows_file_path = (windows_file, metadata_file)
            
            print(f"✅ Loaded {len(self.multi_cell_windows):,} preprocessed windows")
            print(f"   File size: {file_size_mb:.1f} MB")
            print(f"   Created: {created_time}")
            print(f"   Map size: {self.map_width}x{self.map_height}")
            print(f"   Window sizes: {metadata['window_sizes']}")
            print(f"   Rotations: {len(metadata['rotation_angles'])} angles")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading preprocessed windows: {e}")
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
                
                print(f"\n🔍 Found preprocessed windows for this map configuration:")
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
                    print(f"🧹 Cleaned up windows file ({file_size_mb:.1f} MB)")
                
                if metadata_file.exists():
                    os.remove(metadata_file)
                    print(f"🧹 Cleaned up metadata file")
                    
            except Exception as e:
                print(f"⚠️ Error cleaning up preprocessed windows: {e}")

    def ask_about_preprocessed_windows(self, map_path):
        """Ask user about loading/saving preprocessed windows"""
        found, windows_file, metadata_file = self.check_for_preprocessed_windows(map_path)
        
        if found:
            print(f"\n🤔 Do you want to use the existing preprocessed windows?")
            print("   This will skip the time-consuming window generation process.")
            
            while True:
                choice = input("   Use preprocessed windows? [Y/n]: ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    if self.load_preprocessed_windows(windows_file, metadata_file):
                        return True  # Successfully loaded
                    else:
                        print("❌ Failed to load preprocessed windows, will generate new ones")
                        break
                elif choice in ['n', 'no']:
                    print("📝 Will generate new windows from scratch")
                    break
                else:
                    print("❌ Please enter 'y' for yes or 'n' for no")
        
        # Ask about saving for future use
        print(f"\n💾 Do you want to save the processed windows for future reuse?")
        print("   This will speed up future analyses of the same map with same settings.")
        print("   If you choose 'no', windows will be cleaned up after analysis to save storage.")
        
        while True:
            choice = input("   Save windows for reuse? [Y/n]: ").strip().lower()
            if choice in ['', 'y', 'yes']:
                self.save_windows_for_reuse = True
                print("✅ Will save windows for future reuse")
                break
            elif choice in ['n', 'no']:
                self.save_windows_for_reuse = False
                print("✅ Will clean up windows after analysis to save storage")
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no")
        
        return False  # Need to generate new windows

    def load_map_from_file(self, map_path=None):
        """Load map image from local file with preprocessed windows support"""
        
        # First, ask if user wants to use preprocessed map windows
        print("\n🚀 DRONE LOCALIZER - MAP LOADING")
        print("=" * 50)
        
        # Check if there are any preprocessed windows available
        existing_windows = list(self.preprocessed_windows_dir.glob("windows_*.pkl"))
        
        if existing_windows:
            print(f"📁 Found {len(existing_windows)} preprocessed map(s) in 'drone_map_windows' folder:")
            
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
                        print(f"⚠️ Error reading metadata for {windows_file.name}: {e}")
            
            if valid_options:
                print(f"\n🎯 Choose an option:")
                for i, _ in enumerate(valid_options):
                    print(f"{i+1}. Use preprocessed map #{i+1}")
                print(f"{len(valid_options)+1}. Load new map image")
                
                while True:
                    try:
                        choice = int(input(f"\nEnter your choice (1-{len(valid_options)+1}): ").strip())
                        if 1 <= choice <= len(valid_options):
                            # Load selected preprocessed map
                            windows_file, metadata_file, metadata = valid_options[choice-1]
                            print(f"\n🔄 Loading preprocessed map #{choice}...")
                            
                            if self.load_preprocessed_windows(windows_file, metadata_file):
                                return "preprocessed_map_loaded"
                            else:
                                print("❌ Failed to load preprocessed map, continuing to new map...")
                                break
                                
                        elif choice == len(valid_options)+1:
                            print("📤 Loading new map image...")
                            break
                        else:
                            print(f"❌ Invalid choice. Please enter 1-{len(valid_options)+1}")
                    except ValueError:
                        print("❌ Please enter a valid number")
        
        # Load new map image
        if map_path is None:
            print("📤 Please select your map image file...")
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
        print(f"✅ Loaded map: {filename} ({self.map_width}x{self.map_height} pixels)")

        # Show map
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Map: {filename} (Drone Localizer - Processing ALL Window Sizes)')
        plt.axis('off')
        plt.show()

        # Store the map path for preprocessed windows management
        self.current_map_path = map_path
        
        # Generate hash for this configuration
        self.current_map_hash = self.generate_map_hash(map_path, self.window_sizes, self.rotation_angles, self.base_cell_size)
        self.windows_file_path = self.get_windows_file_path(self.current_map_hash)
        
        return map_path


    def configure_commercial_settings(self):
        """Commercial-ready configuration wizard for optimal drone localization"""
        print("\n🚁 COMMERCIAL DRONE LOCALIZER - CONFIGURATION WIZARD")
        print("=" * 65)
        print("This wizard will help you configure optimal settings for your specific use case.")
        print()
        
        # Step 1: Map characteristics
        print("📍 STEP 1: Map Characteristics")
        print("-" * 30)
        print("Your map's viewing characteristics significantly impact matching accuracy.")
        print()
        
        map_type = self._ask_map_type()
        map_resolution = self._ask_map_resolution() 
        
        # Step 2: Window shape strategy
        print("\n🔲 STEP 2: Window Shape Strategy")
        print("-" * 35)
        print("Based on testing, SQUARE windows typically provide better accuracy (0.7 confidence)")
        print("while rectangular windows may reduce accuracy (0.3 confidence).")
        print()
        
        window_strategy = self._ask_window_strategy()
        
        # Step 3: Elevation-based optimization
        print("\n🛩️ STEP 3: Flight Altitude Optimization") 
        print("-" * 40)
        print("Choose elevation levels that match your typical drone operations.")
        print()
        
        elevation_selection = self._ask_elevation_selection()
        
        # Apply configuration
        self._apply_commercial_config(map_type, map_resolution, window_strategy, elevation_selection)
        
        print("\n✅ Configuration complete! Your system is optimized for commercial use.")
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
                    print("❌ Please enter 1, 2, or 3")
            except ValueError:
                print("❌ Please enter a valid number")
    
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
                    print("❌ Please enter 1, 2, 3, or 4")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _ask_window_strategy(self):
        """Ask about window shape preference"""
        print("Choose window shape strategy:")
        print("1. 🟩 SQUARE ONLY (Recommended) - Proven 0.7+ confidence")
        print("2. 🟨 MIXED (Square + Rectangular) - Experimental, may reduce accuracy")
        print("3. 🟧 RECTANGULAR ONLY - Not recommended based on your results")
        print("4. 🔧 CUSTOM - Let me choose specific sizes")
        
        while True:
            try:
                choice = input("Select strategy (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print("❌ Please enter 1, 2, 3, or 4")
            except ValueError:
                print("❌ Please enter a valid number")
                
    def _ask_elevation_selection(self):
        """Ask about elevation/altitude preferences"""
        print("Which drone altitudes will you typically use?")
        print("1. 🛩️ LOW altitude flights (5-50m) - Detailed inspection")
        print("2. 🚁 MEDIUM altitude flights (50-100m) - Standard operations") 
        print("3. ✈️ HIGH altitude flights (100m+) - Area surveys")
        print("4. 🌐 ALL altitudes - Complete coverage")
        
        while True:
            try:
                choice = input("Select altitude range (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print("❌ Please enter 1, 2, 3, or 4")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _apply_commercial_config(self, map_type, map_resolution, window_strategy, elevation_selection):
        """Apply the commercial configuration based on user choices"""
        print(f"\n🔧 APPLYING CONFIGURATION...")
        print("-" * 30)
        
        # Configure window strategy
        if window_strategy == 1:  # Square only
            self.use_rectangular_windows = False
            print("✅ Window strategy: SQUARE ONLY (optimal for accuracy)")
            
        elif window_strategy == 2:  # Mixed
            self.use_rectangular_windows = True
            print("⚠️ Window strategy: MIXED (may reduce accuracy)")
            print("   Consider testing with Square Only if accuracy is insufficient")
            
        elif window_strategy == 3:  # Rectangular only
            self.use_rectangular_windows = True
            print("🚨 Window strategy: RECTANGULAR ONLY (not recommended)")
            print("   Strongly consider switching to Square Only for better results")
            
        elif window_strategy == 4:  # Custom
            print("🔧 Window strategy: CUSTOM - you'll select specific sizes")
        
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
            print(f"💡 RECOMMENDATION: Consider using cell size {suggested_cell_size}x{suggested_cell_size} pixels")
            print(f"   (Current: {self.base_cell_size}x{self.base_cell_size})")
            print(f"   Restart with base_cell_size={suggested_cell_size} for optimal results")
        
        print(f"✅ Selected {len(self.window_sizes)} window sizes: {self.window_sizes}")
        print(f"✅ Elevation levels: {', '.join(selected_levels)}")
        
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
        print("\n🚁 DRONE ELEVATION-BASED WINDOW SELECTION")
        print("=" * 50)
        print("Choose your drone's flight altitude to get optimized window sizes:")
        print("📊 NOTE: Square windows typically provide better accuracy than rectangular ones!")
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
                    print("❌ Please enter 1-7")
            except ValueError:
                print("❌ Please enter a valid number")
    
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
            
            print(f"\n❓ Include rectangular windows for {level.upper().replace('_', ' ')}?")
            print("⚠️ Warning: Rectangular windows may reduce accuracy")
            include_rect = input("Include rectangular sizes? [y/N]: ").strip().lower()
            
            if include_rect in ['y', 'yes']:
                selected_sizes.extend(rect_sizes)
                print("⚠️ Added rectangular sizes - monitor accuracy carefully")
            else:
                print("✅ Using square sizes only (recommended)")
        
        self.window_sizes = selected_sizes
        self.current_elevation_level = level
        
        sizes_str = ", ".join([f"{h}×{w}" for h, w in selected_sizes])
        level_name = level.upper().replace('_', ' ')
        print(f"\n✅ Selected {level_name} elevation level")
        print(f"   Window sizes: {sizes_str}")
        return self.window_sizes
    
    def choose_custom_elevation_mix(self):
        """Let user mix and match sizes from different elevation levels"""
        print("\n🎯 CUSTOM ELEVATION MIX")
        print("Select individual window sizes from any elevation level:")
        print()
        
        # Show all available sizes organized by level
        all_sizes = []
        level_names = []
        
        for level, info in self.elevation_levels.items():
            print(f"📊 {level.upper().replace('_', ' ')} - {info['description']}")
            
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
                    print(f"❌ Please enter a number between 1 and {len(all_sizes)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
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
                        print(f"✅ Added {h}×{w} from {level_name}")
                    else:
                        print("❌ Size already selected")
                else:
                    print(f"❌ Please enter a number between 1 and {len(all_sizes)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
        
        self.window_sizes = chosen_sizes
        self.current_elevation_level = "custom_mix"
        
        sizes_str = ", ".join([f"{h}×{w}" for h, w in chosen_sizes])
        print(f"\n✅ Custom elevation mix selected: {sizes_str}")
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
        print(f"\n✅ Using all elevation levels: {sizes_str}")
        print(f"   Total: {len(unique_sizes)} different window sizes")
        print(f"   Covers all drone altitudes from very low to high!")
        return self.window_sizes

    def visualize_cell_grid(self):
        """Visualize the base cell grid on the map"""
        print(f"🔍 Visualizing base cell grid ({self.base_cell_size}x{self.base_cell_size} pixels)...")
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB))
        
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
        plt.show()
        
        print(f"✓ Grid contains {grid_rows * grid_cols:,} base cells")

    def visualize_window_sizes(self):
        """Visualize different window sizes to help user choose optimal sizes"""
        print("🔍 Visualizing different window sizes...")
        
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
        plt.show()
        
        return available_sizes

    def choose_window_sizes(self):
        """Interactive method to choose window sizes with commercial-ready options"""
        print("\n🎯 DRONE WINDOW SIZE SELECTION")
        print("=" * 50)
        print("� PERFORMANCE NOTE: Square windows typically achieve 0.7+ confidence")
        print("� WARNING: Rectangular windows may reduce confidence to 0.3")
        print()
        
        print("Choose selection method:")
        print("1. 🏢 COMMERCIAL CONFIG - Guided setup (Recommended for commercial use)")
        print("2. 🚁 Elevation-based selection")
        print("3. 📋 Manual selection from all available sizes")
        print("4. ➕ Add custom size")
        
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
                    print("❌ Please enter 1, 2, 3, or 4")
                    
            except ValueError:
                print("❌ Please enter a valid number")
    
    def choose_manual_sizes(self):
        """Manual selection from all available sizes"""
        print("\n📋 MANUAL SIZE SELECTION")
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
                    print(f"❌ Please enter between 1 and {len(all_sizes)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
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
                        print(f"✅ Added {h}×{w} ({source})")
                    else:
                        print("❌ Size already selected")
                else:
                    print(f"❌ Please enter between 1 and {len(all_sizes)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        self.window_sizes = chosen_sizes
        self.current_elevation_level = "manual_selection"
        return self.window_sizes
    
    def add_custom_drone_ratio_size(self):
        """Add a custom size that maintains the drone aspect ratio"""
        print("\n➕ ADD CUSTOM DRONE-RATIO SIZE")
        print(f"Creating a window that maintains your drone's {self.drone_aspect_ratio:.3f}:1 aspect ratio")
        print()
        
        while True:
            try:
                height = input("Enter window height in cells (2-30): ").strip()
                height = int(height)
                if 2 <= height <= 30:
                    break
                else:
                    print("❌ Please enter between 2 and 30")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Calculate width to maintain drone aspect ratio
        width = round(height * self.drone_aspect_ratio)
        actual_ratio = width / height
        
        print(f"\n📐 Calculated dimensions:")
        print(f"   Height: {height} cells ({height * self.base_cell_size} pixels)")
        print(f"   Width: {width} cells ({width * self.base_cell_size} pixels)")
        print(f"   Aspect ratio: {actual_ratio:.3f}:1")
        print(f"   Difference from drone ratio: {abs(actual_ratio - self.drone_aspect_ratio):.3f}")
        
        if abs(actual_ratio - self.drone_aspect_ratio) < 0.1:
            print("   ✅ Excellent match with drone aspect ratio!")
        else:
            print("   ⚠️ Slight deviation due to rounding to whole cells")
        
        confirm = input(f"\nUse this size ({height}×{width})? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            custom_size = (height, width)
            
            # Add to current sizes or create new list
            if hasattr(self, 'window_sizes') and self.window_sizes:
                if custom_size not in self.window_sizes:
                    self.window_sizes.append(custom_size)
                    print(f"✅ Added {height}×{width} to existing window sizes")
                else:
                    print("❌ This size already exists")
            else:
                self.window_sizes = [custom_size]
                print(f"✅ Set {height}×{width} as window size")
            
            self.current_elevation_level = "custom_ratio"
            return self.window_sizes
        else:
            print("❌ Custom size cancelled")
            return self.choose_window_sizes()  # Return to main menu

    def display_elevation_system_info(self):
        """Display comprehensive information about the elevation-based window system"""
        print("\n🏔️ ELEVATION-BASED WINDOW SIZING SYSTEM")
        print("=" * 55)
        print("🚁 Designed specifically for drone video analysis")
        print(f"📐 All windows maintain your drone's {self.drone_aspect_ratio:.3f}:1 aspect ratio (4000×2250 pixels)")
        print()
        
        print("🎯 ALTITUDE LEVELS:")
        for level, info in self.elevation_levels.items():
            level_name = level.upper().replace('_', ' ')
            altitude_range = info["altitude_range"]
            description = info["description"]
            square_sizes = info.get("square_sizes", [])
            rect_sizes = info.get("rectangular_sizes", [])
            
            print(f"\n📏 {level_name} ALTITUDE ({altitude_range})")
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
        
        print(f"\n💡 USAGE RECOMMENDATIONS:")
        print(f"   • Lower altitude = Larger windows (more detail, closer terrain)")
        print(f"   • Higher altitude = Smaller windows (broader coverage, distant terrain)")
        print(f"   • All windows maintain perfect drone aspect ratio for optimal matching")
        print(f"   • Base cell size: {self.base_cell_size}×{self.base_cell_size} pixels")
        
        input("\nPress Enter to continue...")



    def show_selected_sizes_preview(self):
        """Show a preview of the selected window sizes"""
        if not self.window_sizes:
            return
        
        print(f"🔍 Previewing selected window sizes: {self.window_sizes}")
        
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
        plt.show()

    def extract_embedding(self, image):
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_tensor)
            if len(embedding.shape) > 2:
                embedding = embedding.view(embedding.size(0), -1)
            return embedding.cpu().numpy().flatten()

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
        print(f"🔄 Processing map with ALL window sizes 3x3 to 15x15 ({len(self.window_sizes)} sizes)...")
        print(f"   Window sizes: {', '.join([f'{h}x{w}' for h, w in self.window_sizes])}")
        self.print_memory_stats("at start")

        # Check if we already have preprocessed windows loaded
        if hasattr(self, 'multi_cell_windows') and len(self.multi_cell_windows) > 0 and hasattr(self, 'index') and self.index is not None:
            print("✅ Using already loaded preprocessed windows")
            print(f"📊 Total searchable windows: {len(self.multi_cell_windows):,}")
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

        print(f"✓ Created {len(self.cells):,} base cells")
        for window_rows, window_cols in self.window_sizes:
            size_total = sum(count for key, count in window_counts.items() if key.startswith(f"{window_rows}x{window_cols}"))
            print(f"✓ Created {size_total:,} windows of size {window_rows}x{window_cols} (all rotations)")
        print(f"✓ Total large windows: {len(self.multi_cell_windows):,}")
        self.print_memory_stats("after creating windows")

        # Compute embeddings (memory efficient - don't store images)
        start_time = time.time()

        for i, window in enumerate(self.multi_cell_windows):
            if i % 500 == 0:
                print(f"  Processing windows: {i:,}/{len(self.multi_cell_windows):,}")
                if i % 2000 == 0 and i > 0:  # Print memory every 2000 windows
                    self.print_memory_stats(f"after {i} windows")

            # Generate images on-demand, extract embedding, then discard images
            composite_image, rotated_image = self.generate_window_image(window)
            window.embedding = self.extract_embedding(rotated_image)
            
            # Images are automatically garbage collected after this scope
            # This saves GBs of memory!

        # Build FAISS index
        all_embeddings = []
        self.all_info = []

        for window in self.multi_cell_windows:
            all_embeddings.append(window.embedding)
            self.all_info.append(('window', window))

        all_embeddings = np.array(all_embeddings)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(all_embeddings.astype('float32'))

        elapsed = time.time() - start_time
        print(f"✅ Map processing completed in {elapsed/60:.1f} minutes!")
        print(f"📊 Total searchable windows: {len(self.all_info):,}")
        
        # Always save preprocessed windows for reuse (efficiency)
        print(f"💾 Saving preprocessed windows to 'drone_map_windows' folder...")
        self.save_preprocessed_windows()
        
        # Force garbage collection to free memory
        gc.collect()
        print(f"🧹 Memory cleanup completed")

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
        print("🧹 Memory cleanup completed - freed several GBs!")

    def get_nearby_windows(self, center_x, center_y, radius):
        """Get windows within radius of given center point"""
        nearby_indices = []
        for i, (item_type, window) in enumerate(self.all_info):
            distance = np.sqrt((window.center_x - center_x)**2 + (window.center_y - center_y)**2)
            if distance <= radius:
                nearby_indices.append(i)
        return nearby_indices

    def localize_frame(self, frame_image, frame_number, timestamp, k=25):
        """Locate drone in a single frame with enhanced movement freedom and precise cell-level matching"""
        
        frame_embedding = self.extract_embedding(frame_image)
        frame_embedding = frame_embedding / np.linalg.norm(frame_embedding)

        use_global_search = False
        
        if self.previous_location is not None:
            if self.previous_location.confidence < self.min_confidence_for_temporal:
                use_global_search = True
                print(f"  Frame {frame_number}: Low previous confidence ({self.previous_location.confidence:.3f}) - using global search")
            else:
                confidence_factor = 1.0 - self.previous_location.confidence
                adaptive_radius = self.search_radius + (confidence_factor * self.max_search_radius)
                adaptive_radius = min(adaptive_radius, self.max_search_radius)
                
                print(f"  Frame {frame_number}: Adaptive search radius: {adaptive_radius:.0f}px (confidence: {self.previous_location.confidence:.3f})")
        else:
            use_global_search = True
            adaptive_radius = self.search_radius
        
        if use_global_search or self.previous_location is None:
            print(f"  Frame {frame_number}: Global search")
            similarities, indices = self.index.search(
                frame_embedding.reshape(1, -1).astype('float32'), k
            )
            top_similarities = similarities[0]
            top_indices_global = indices[0]
        else:
            nearby_indices = self.get_nearby_windows(
                self.previous_location.x, 
                self.previous_location.y, 
                adaptive_radius
            )
            
            if len(nearby_indices) < 10:
                print(f"  Frame {frame_number}: Too few nearby windows ({len(nearby_indices)}), expanding search...")
                nearby_indices = self.get_nearby_windows(
                    self.previous_location.x, 
                    self.previous_location.y, 
                    self.max_search_radius
                )
                
                if len(nearby_indices) < 15:
                    print(f"  Frame {frame_number}: Still insufficient windows, switching to global search")
                    similarities, indices = self.index.search(
                        frame_embedding.reshape(1, -1).astype('float32'), k
                    )
                    top_similarities = similarities[0]
                    top_indices_global = indices[0]
                else:
                    nearby_embeddings = np.array([self.all_info[i][1].embedding for i in nearby_indices])
                    nearby_embeddings = nearby_embeddings / np.linalg.norm(nearby_embeddings, axis=1, keepdims=True)
                    
                    similarities = np.dot(nearby_embeddings, frame_embedding)
                    top_k_local = min(k, len(similarities))
                    top_indices_local = np.argsort(similarities)[::-1][:top_k_local]
                    
                    top_similarities = similarities[top_indices_local]
                    top_indices_global = [nearby_indices[i] for i in top_indices_local]
            else:
                print(f"  Frame {frame_number}: Local search with {len(nearby_indices)} windows")
                nearby_embeddings = np.array([self.all_info[i][1].embedding for i in nearby_indices])
                nearby_embeddings = nearby_embeddings / np.linalg.norm(nearby_embeddings, axis=1, keepdims=True)
                
                similarities = np.dot(nearby_embeddings, frame_embedding)
                top_k_local = min(k, len(similarities))
                top_indices_local = np.argsort(similarities)[::-1][:top_k_local]
                
                top_similarities = similarities[top_indices_local]
                top_indices_global = [nearby_indices[i] for i in top_indices_local]

        # Create matches
        matches = []
        for sim, idx in zip(top_similarities, top_indices_global):
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

        # Use the best matching windows directly (simplified approach)
        if matches:
            best_window = matches[0]['window']
            print(f"  Frame {frame_number}: Using best window match at ({matches[0]['x']:.1f}, {matches[0]['y']:.1f})")
            
            # Use top matches for weighted average (more robust than single window)
            top_matches = matches[:min(8, len(matches))]
            
            weights = [m['similarity'] for m in top_matches]
            total_weight = sum(weights)
            
            if total_weight > 0:
                similarity_x = sum(w * m['x'] for w, m in zip(weights, top_matches)) / total_weight
                similarity_y = sum(w * m['y'] for w, m in zip(weights, top_matches)) / total_weight
                similarity_rotation = sum(w * m['rotation'] for w, m in zip(weights, top_matches)) / total_weight
                
                print(f"  Frame {frame_number}: Weighted average from {len(top_matches)} top matches")
                
                # Apply temporal smoothing if applicable
                if (self.previous_location is not None and 
                    self.previous_location.confidence > self.min_confidence_for_temporal and
                    not use_global_search):
                    
                    estimated_x = (1 - self.temporal_weight) * similarity_x + self.temporal_weight * self.previous_location.x
                    estimated_y = (1 - self.temporal_weight) * similarity_y + self.temporal_weight * self.previous_location.y
                    estimated_rotation = (1 - self.temporal_weight) * similarity_rotation + self.temporal_weight * self.previous_location.rotation
                    
                    print(f"  Frame {frame_number}: Applied temporal smoothing (weight: {self.temporal_weight})")
                else:
                    estimated_x = similarity_x
                    estimated_y = similarity_y
                    estimated_rotation = similarity_rotation
                    print(f"  Frame {frame_number}: No temporal smoothing applied")
                
                # Use the best window's similarity as confidence
                confidence = matches[0]['similarity']
                
            else:
                # Fallback to single best match
                estimated_x = matches[0]['x']
                estimated_y = matches[0]['y']
                estimated_rotation = matches[0]['rotation']
                confidence = matches[0]['similarity']
        else:
            # No matches found
            estimated_x, estimated_y = 0, 0
            estimated_rotation = 0
            confidence = 0
            best_window = None

        trajectory_point = TrajectoryPoint(
            frame_number=frame_number,
            timestamp=timestamp,
            x=estimated_x,
            y=estimated_y,
            rotation=estimated_rotation,
            confidence=confidence,
            window_info=best_window
        )

        self.previous_location = trajectory_point
        return trajectory_point, matches

    def analyze_video_from_file(self, video_path=None):
        """Analyze video from local file"""
        if video_path is None:
            print("📤 Please select your drone video file...")
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
        print(f"🎬 Processing video: {filename}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"📹 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

        self.trajectory = []
        self.previous_location = None

        frame_number = 0
        start_time = time.time()

        frame_skip = max(1, int(fps // 10))
        print(f"🔄 Processing every {frame_skip} frame(s) for efficiency")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_skip == 0:
                timestamp = frame_number / fps
                
                print(f"📍 Processing frame {frame_number}/{total_frames} (t={timestamp:.2f}s)")
                
                try:
                    trajectory_point, matches = self.localize_frame(frame, frame_number, timestamp)
                    self.trajectory.append(trajectory_point)
                    
                    print(f"   Position: ({trajectory_point.x:.1f}, {trajectory_point.y:.1f})")
                    print(f"   Rotation: {trajectory_point.rotation:.1f}°, Confidence: {trajectory_point.confidence:.3f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing frame {frame_number}: {e}")

            frame_number += 1

        cap.release()
        
        processing_time = time.time() - start_time
        print(f"✅ Video analysis completed in {processing_time:.1f} seconds!")
        print(f"📊 Processed {len(self.trajectory)} trajectory points")

        return self.trajectory

    def test_random_frames_from_file(self, video_path=None, num_frames=70):
        """Test random frames from local video file"""
        if video_path is None:
            print("📤 Please select your drone video file for random frame testing...")
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
        print(f"🎬 Processing video for random testing: {filename}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"📹 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")

        random_indices = np.random.choice(total_frames, min(num_frames, total_frames), replace=False)
        random_indices = sorted(random_indices)
        
        print(f"🎲 Selected {len(random_indices)} random frames for testing")

        test_results = []
        
        for i, frame_idx in enumerate(random_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ Could not read frame {frame_idx}")
                continue
                
            timestamp = frame_idx / fps
            print(f"🔍 Testing frame {i+1}/{len(random_indices)} (frame #{frame_idx}, t={timestamp:.2f}s)")
            
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
                print(f"   ⚠️ Error processing frame {frame_idx}: {e}")

        cap.release()
        print(f"✅ Random frame testing completed! Processed {len(test_results)} frames")
        
        return test_results

    def visualize_random_frame_results(self, test_results, frames_per_figure=10):
        """Create visualizations showing frames alongside their localization results"""
        if not test_results:
            print("❌ No test results to visualize")
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
            plt.show()

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
        plt.show()
        
        print(f"\n📊 Random Frame Testing Statistics:")
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
            print("❌ No trajectory data to visualize")
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
        plt.show()

        self._show_trajectory_stats()

    def _show_trajectory_stats(self):
        """Show detailed trajectory statistics"""
        if len(self.trajectory) < 2:
            return

        print(f"\n📊 Trajectory Analysis:")
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


    def export_trajectory(self, filename="drone_trajectory.csv"):
        """Export trajectory to CSV file"""
        if not self.trajectory:
            print("❌ No trajectory data to export")
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
        
        print(f"✅ Trajectory exported to {filename}")

    def adjust_movement_freedom(self, search_radius=None, max_search_radius=None, 
                              min_confidence_for_temporal=None, temporal_weight=None):
        """Adjust movement freedom parameters during runtime"""
        if search_radius is not None:
            self.search_radius = search_radius
            print(f"✓ Search radius updated to: {search_radius}px")
        
        if max_search_radius is not None:
            self.max_search_radius = max_search_radius
            print(f"✓ Max search radius updated to: {max_search_radius}px")
        
        if min_confidence_for_temporal is not None:
            self.min_confidence_for_temporal = min_confidence_for_temporal
            print(f"✓ Min confidence for temporal coherence updated to: {min_confidence_for_temporal}")
        
        if temporal_weight is not None:
            self.temporal_weight = temporal_weight
            print(f"✓ Temporal smoothing weight updated to: {temporal_weight}")
        
        print(f"\n📊 Current Movement Freedom Settings:")
        print(f"   Search radius: {self.search_radius}px")
        print(f"   Max search radius: {self.max_search_radius}px")
        print(f"   Min confidence for temporal: {self.min_confidence_for_temporal}")
        print(f"   Temporal weight: {self.temporal_weight}")

    def run_local_analysis(self):
        """Main function to run local analysis with user choices"""
        print("\n🎯 Choose Analysis Mode:")
        print("1. Full Video Trajectory Analysis")
        print("2. Random Frame Testing (70 frames)")
        print("3. Adjust Movement Freedom Settings")
        print("4. Cleanup Memory (Free GBs)")
        print("5. Manage Preprocessed Windows")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == "1":
                    print("\n🎬 Running Full Video Trajectory Analysis...")
                    trajectory = self.analyze_video_from_file()
                    if trajectory:
                        print("\n🎨 Creating trajectory visualization...")
                        self.visualize_trajectory()
                        print("\n💾 Exporting trajectory data...")
                        self.export_trajectory()
                    break
                    
                elif choice == "2":
                    print("\n🎲 Running Random Frame Testing...")
                    test_results = self.test_random_frames_from_file(num_frames=70)
                    if test_results:
                        print("\n🎨 Creating detailed visualizations...")
                        self.visualize_random_frame_results(test_results, frames_per_figure=10)
                        print("\n📊 Creating summary visualization...")
                        self.create_summary_visualization(test_results)
                    break
                    
                elif choice == "3":
                    print("\n⚙️ Current Settings:")
                    print(f"   Search radius: {self.search_radius}px")
                    print(f"   Max search radius: {self.max_search_radius}px")
                    print(f"   Min confidence for temporal: {self.min_confidence_for_temporal}")
                    print(f"   Temporal weight: {self.temporal_weight}")
                    
                    try:
                        new_radius = input(f"\nNew search radius (current: {self.search_radius}, Enter to skip): ").strip()
                        if new_radius:
                            self.search_radius = int(new_radius)
                        
                        new_max_radius = input(f"New max search radius (current: {self.max_search_radius}, Enter to skip): ").strip()
                        if new_max_radius:
                            self.max_search_radius = int(new_max_radius)
                        
                        new_temporal = input(f"New temporal weight (current: {self.temporal_weight}, Enter to skip): ").strip()
                        if new_temporal:
                            self.temporal_weight = float(new_temporal)
                        
                        print("✅ Settings updated!")
                        
                    except ValueError:
                        print("❌ Invalid input, settings unchanged")
                
                elif choice == "4":
                    print("\n🧹 Cleaning up memory...")
                    self.print_memory_stats("before cleanup")
                    self.cleanup_memory()
                    self.print_memory_stats("after cleanup")
                
                elif choice == "5":
                    self.manage_preprocessed_windows()
                    
                elif choice == "6":
                    print("👋 Goodbye!")
                    break
                    
                else:
                    print("❌ Invalid choice. Please enter 1-6.")
                    
            except KeyboardInterrupt:
                print("\n❌ Analysis cancelled")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break

    def manage_preprocessed_windows(self):
        """Manage existing preprocessed windows"""
        print("\n📁 PREPROCESSED WINDOWS MANAGEMENT")
        print("=" * 50)
        
        # List existing preprocessed windows
        windows_files = list(self.preprocessed_windows_dir.glob("windows_*.pkl"))
        metadata_files = list(self.preprocessed_windows_dir.glob("metadata_*.json"))
        
        if not windows_files:
            print("📂 No preprocessed windows found")
            return
        
        print(f"📂 Found {len(windows_files)} preprocessed window file(s):")
        
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
                    
                    print(f"\n🔹 File: {windows_file.name}")
                    print(f"   Size: {file_size_mb:.1f} MB")
                    print(f"   Windows: {metadata['total_windows']:,}")
                    print(f"   Created: {created_time}")
                    print(f"   Window sizes: {metadata['window_sizes']}")
                    print(f"   Map size: {metadata['map_width']}x{metadata['map_height']}")
                    
                    valid_files.append((windows_file, metadata_file, metadata))
                    
                except Exception as e:
                    print(f"\n⚠️ Error reading {windows_file.name}: {e}")
        
        print(f"\n📊 Total storage used: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        
        if not valid_files:
            return
        
        print(f"\n🎯 What would you like to do?")
        print("1. Delete all preprocessed windows")
        print("2. Delete specific files")
        print("3. Keep all files")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            confirm = input(f"⚠️ Delete ALL {len(valid_files)} files ({total_size_mb:.1f} MB)? [y/N]: ").strip().lower()
            if confirm == 'y':
                deleted_size = 0
                for windows_file, metadata_file, _ in valid_files:
                    try:
                        deleted_size += os.path.getsize(windows_file) / (1024 * 1024)
                        os.remove(windows_file)
                        os.remove(metadata_file)
                    except Exception as e:
                        print(f"⚠️ Error deleting {windows_file.name}: {e}")
                
                print(f"✅ Deleted all preprocessed windows ({deleted_size:.1f} MB freed)")
            else:
                print("❌ Cancelled")
        
        elif choice == "2":
            print(f"\n📋 Select files to delete (enter numbers separated by spaces):")
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
                            print(f"✅ Deleted {windows_file.name}")
                        except Exception as e:
                            print(f"⚠️ Error deleting {windows_file.name}: {e}")
                    else:
                        print(f"⚠️ Invalid selection: {sel}")
                
                if deleted_size > 0:
                    print(f"✅ Freed {deleted_size:.1f} MB of storage")
                
            except (ValueError, IndexError):
                print("❌ Invalid input")
        
        elif choice == "3":
            print("✅ Keeping all preprocessed windows")
        
        else:
            print("❌ Invalid choice")

def main():
    """Main function to run the local drone localizer with elevation-based window size selection"""
    print("🚁 Elevation-Aware Drone Video Trajectory Localizer")
    print("=" * 60)
    print("📋 Required packages: torch, torchvision, opencv-python, pillow, faiss-cpu, matplotlib, numpy, tkinter")
    print("💻 Running locally in VS Code")
    print("🆓 Enhanced movement freedom for natural drone tracking")
    print("🎯 Elevation-based window sizing with 4000:2250 drone aspect ratio")
    print("📐 4-level altitude system: so_low → low → mid → high")
    print()

    # Initialize the localizer
    localizer = LocalVideoTrajectoryDroneLocalizer(
        base_cell_size=32,
        rotation_angles=[0, 45, 90, 135, 180, 225, 270, 315]
    )

    print("\n📤 Step 1: Loading map image...")
    try:
        localizer.load_map_from_file()
    except Exception as e:
        print(f"❌ Error loading map: {e}")
        return

    print("\n🔍 Step 2: Visualizing cell grid...")
    try:
        localizer.visualize_cell_grid()
    except Exception as e:
        print(f"❌ Error visualizing grid: {e}")
        return

    print("\n�️ Step 3: Understanding elevation-based window system...")
    try:
        localizer.display_elevation_system_info()
    except Exception as e:
        print(f"❌ Error displaying elevation info: {e}")
        return

    print("\n🎯 Step 4: Choosing drone altitude-based window sizes...")
    try:
        chosen_sizes = localizer.choose_window_sizes()
        print(f"✅ Selected window sizes: {chosen_sizes}")
        if hasattr(localizer, 'current_elevation_level'):
            print(f"📏 Elevation level: {localizer.current_elevation_level}")
    except Exception as e:
        print(f"❌ Error choosing window sizes: {e}")
        return

    print("\n👀 Step 5: Previewing selected window sizes...")
    try:
        localizer.show_selected_sizes_preview()
    except Exception as e:
        print(f"❌ Error showing preview: {e}")
        return

    print("\n� Step 6: Checking for preprocessed windows...")
    try:
        # Check if we can use preprocessed windows
        windows_loaded = localizer.ask_about_preprocessed_windows(localizer.current_map_path)
        
        if not windows_loaded:
            print("\n�🔄 Step 7: Processing map with selected window sizes...")
            localizer.process_map()
        else:
            print("\n✅ Step 7: Using preprocessed windows (skipped processing)")
    except Exception as e:
        print(f"❌ Error with preprocessed windows: {e}")
        return

    print("\n🎬 Step 8: Choose analysis mode...")
    try:
        localizer.run_local_analysis()
    finally:
        # Cleanup if user chose not to save windows
        if not localizer.save_windows_for_reuse:
            print("\n🧹 Final cleanup (user chose not to save windows)...")
            localizer.cleanup_preprocessed_windows()

if __name__ == "__main__":
    main()