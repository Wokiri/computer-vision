from pathlib import Path
import cv2 as cv
from PyQt5.QtGui import QImage, QPixmap

from utilities.seam_carving_algorithms import SeamCarving

class ImageProcessor:
    def __init__(self, file_path):
        self.image_path = Path(file_path)
        self.current_image = None
        self.original_image = None
        
        self.load_image()
        
    def load_image(self):
        """Load an image from file path"""
        try:
            # Convert to Path object and validate
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
            
        # Check for common image file extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        file_extension = image_path.suffix.lower()
        
        if file_extension not in valid_extensions:
            print(f"Warning: File extension {file_extension} may not be supported")
            # Don't return False here as OpenCV might still be able to read it
            
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
        if new_width <= 0 or new_height  <= 0:
            raise ValueError("Width and height values must be greator than 0")
        
        # Extract kwargs with default values
        content_aware = kwargs.get('content_aware', False)
        content_aware_alg = kwargs.get('content_aware_alg', None)
        progress_callback = kwargs.get('progress_callback', None)
        
        try:
            if content_aware:
                self.current_image = self._seam_carving_resize(
                    new_width, new_height, 
                    algorithm=content_aware_alg,
                    progress_callback=progress_callback
                )
            else:
                self.current_image = cv.resize(self.current_image, (new_width, new_height), interpolation=cv.INTER_AREA)
            
            return True
        except Exception as e:
            print(f"Error resizing image: {e}")
            return False
        
    def _seam_carving_resize(self, new_width, new_height, algorithm, progress_callback=None):
        """Content-aware resizing using seam carving"""
        if algorithm in ["Hubble 001", "Hubble 002"]:
            # Create SeamCarving instance with the specified algorithm
            seam_carver = SeamCarving(algorithm=algorithm)
            
            # Use the seam carver to resize the current image
            result = seam_carver.carve(
                img=self.current_image.copy(),
                target_width=new_width,
                target_height=new_height,
                progress_callback=progress_callback
            )
            
            # Update the current image and return
            self.current_image = result
            return result
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    
    def reset_to_original(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            return True
        return False
    
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
        