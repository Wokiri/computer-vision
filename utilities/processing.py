"""
utilities/processing.py
Image processing utilities for loading, resizing, and saving images
"""

from pathlib import Path
import cv2 as cv
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

from utilities.seam_carving_algorithms import SeamCarver, SeamCarverHubble001, SeamCarverHubble002

class ImageProcessor:
    def __init__(self, file_path):
        self.image_path = Path(file_path)
        self.current_image = None
        self.original_image = None
        self.last_timing_info = None
        self.last_seam_info = None
        self.last_algorithm_used = None
        
        self.load_image()
        
    def load_image(self):
        """Load an image from file path"""
        try:
            image_path = self.image_path
            if not self._is_valid_image_path(image_path):
                return False
            
            self.original_image = cv.imread(str(image_path))
            if self.original_image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            self.current_image = self.original_image.copy()
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def _is_valid_image_path(self, image_path):
        """Validate if the file path exists and is a valid image file"""
        if not image_path or not isinstance(image_path, Path):
            print("Error: Invalid file path provided")
            return False
            
        if not image_path.exists():
            print(f"Error: File does not exist: {image_path}")
            return False
            
        if not image_path.is_file():
            print(f"Error: Path is not a file: {image_path}")
            return False
            
        return True
    
    def is_image_loaded(self):
        """Check if an image is currently loaded"""
        return self.current_image is not None and self.original_image is not None
    
    def get_image_info(self):
        """Get information about the currently loaded image"""
        if not self.is_image_loaded():
            return "No image loaded"
        
        height, width = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
        
        file_size = self.image_path.stat().st_size if self.image_path and self.image_path.exists() else 0
        
        return {
            'path': str(self.image_path),
            'dimensions': f"{width}x{height}",
            'channels': channels,
            'size': f"{file_size / 1024:.2f} KB"
        }

    def get_image_directory(self):
        """Get the directory containing the current image"""
        if self.image_path:
            return self.image_path.parent
        return None
    
    def get_image_filename(self):
        """Get the filename without extension"""
        if self.image_path:
            return self.image_path.stem
        return None
    
    def get_image_extension(self):
        """Get the file extension"""
        if self.image_path:
            return self.image_path.suffix
        return None
    
    def get_image_dimensions(self):
        """Get current image dimensions"""
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            return width, height
        return 0, 0
    
    def resize_image(self, new_width=None, new_height=None, **kwargs):
        """Resize the current image
        
        Args:
            new_width: Target width (required)
            new_height: Target height (required)
            **kwargs: Additional options:
                - content_aware: Use content-aware resizing (default: False)
                - content_aware_alg: Algorithm for content-aware resizing
                - progress_callback: Callback for progress updates
        
        Returns:
            success: Boolean indicating if resize was successful
        """
        if self.current_image is None:
            return False
        
        # We need target dimensions
        if new_width is None or new_height is None:
            raise ValueError("Width and height are required for content-aware resize")
        
        # Convert to integers and validate
        try:
            new_width = int(new_width)
            new_height = int(new_height)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Width and height must be numeric values: {e}")
        
        # We need target dimensions
        if new_width <= 0 or new_height <= 0:
            raise ValueError("Width and height values must be greater than 0")
        
        # Extract kwargs with default values
        content_aware = kwargs.get('content_aware', False)
        content_aware_alg = kwargs.get('content_aware_alg', None)
        progress_callback = kwargs.get('progress_callback', None)
        
        try:
            if content_aware:
                # Store the algorithm used
                self.last_algorithm_used = content_aware_alg
                
                # Use seam carving with seam information
                result = self._seam_carving_resize(
                    new_width, new_height, 
                    algorithm=content_aware_alg,
                    progress_callback=progress_callback,
                    return_seams=True
                )
                
                if result:
                    # result should be (image, seam_info) tuple when return_seams=True
                    if isinstance(result, tuple) and len(result) == 2:
                        self.current_image, self.last_seam_info = result
                    else:
                        # If not a tuple, just get the image
                        self.current_image = result
                        self.last_seam_info = {}
                    
                    # Get timing info
                    self.last_timing_info = self.last_seam_info.get('timing', {}) if self.last_seam_info else {}
                    
                    # Create seam visualizations if we have seam info
                    if self.last_seam_info and self.original_image is not None:
                        # Create seam carver instance using explicit class
                        seam_carver = self._create_seam_carver(self.last_algorithm_used)
                        
                        # Call the method that returns a tuple of three images
                        visualizations = seam_carver.create_seam_visualization(
                            self.original_image,
                            self.last_seam_info
                        )
                        
                        # Unpack and store the three visualization types
                        all_seams_img, removed_img, inserted_img = visualizations
                        self.last_seam_info['all_seams_image'] = all_seams_img
                        self.last_seam_info['removed_seams_image'] = removed_img
                        self.last_seam_info['inserted_seams_image'] = inserted_img
                else:
                    return False
            else:
                # Traditional resize
                self.current_image = cv.resize(self.current_image, (new_width, new_height), interpolation=cv.INTER_AREA)
                self.last_timing_info = {'algorithm': 0.0, 'total': 0.0, 'algorithm_name': 'Traditional'}
                self.last_seam_info = None
                self.last_algorithm_used = None
            
            return True
        except Exception as e:
            print(f"Error resizing image: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_seam_carver(self, algorithm: str) -> SeamCarver:
        """Helper method to create seam carver instances"""
        if algorithm == "Hubble 001":
            return SeamCarverHubble001()
        elif algorithm == "Hubble 002":
            return SeamCarverHubble002()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def get_seam_visualizations(self):
        """Get seam visualization images"""
        if self.last_seam_info and self.last_algorithm_used:
            return {
                'all': self.last_seam_info.get('all_seams_image'),
                'removed': self.last_seam_info.get('removed_seams_image'),
                'inserted': self.last_seam_info.get('inserted_seams_image')
            }
        return None
    
    def reset_to_original(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.last_timing_info = None
            self.last_seam_info = None
            self.last_algorithm_used = None
            return True
        return False
        
    def _seam_carving_resize(self, new_width, new_height, algorithm, progress_callback=None, return_seams=False):
        """Content-aware resizing using seam carving"""
        # Create seam carver instance
        seam_carver = self._create_seam_carver(algorithm)
        
        # Use the seam carver to resize the current image
        result = seam_carver.carve(
            img=self.current_image.copy(),
            target_width=new_width,
            target_height=new_height,
            progress_callback=progress_callback,
            return_seams=return_seams
        )
        
        return result
    
    def get_processed_image(self):
        """Get the current processed image with timing and seam info"""
        if self.current_image is not None:
            return self.current_image.copy(), self.last_timing_info, self.last_seam_info
        return None
    
    def save_image(self, save_path, img = None):
        """Save image to file"""
        if img is None:
            if self.current_image is None:
                return False
            img = self.current_image
        
        try:
            cv.imwrite(save_path, img)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def convert_cv_image(self, image=None, output_format="qpixmap"):
        """
        Convert OpenCV image to QImage or QPixmap
        
        Args:
            image: OpenCV image (if None, uses self.current_image)
            output_format: "qimage" or "qpixmap"
        
        Returns:
            QImage or QPixmap object, or empty object if conversion fails
        """
        if image is None:
            image = self.current_image
            
        if image is None:
            return QImage() if output_format.lower() == "qimage" else QPixmap()
        
        try:
            # Convert BGR to RGB for color images
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w if len(rgb_image.shape) == 3 else w
            
            # Create QImage
            if len(rgb_image.shape) == 3:
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            # Return based on requested format
            if output_format.lower() == "qimage":
                return q_img
            else:  # qpixmap
                return QPixmap.fromImage(q_img)
                
        except Exception as e:
            print(f"Error converting image: {e}")
            return QImage() if output_format.lower() == "qimage" else QPixmap()