from pathlib import Path
import cv2 as cv
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

class ImageProcessor:
    def __init__(self, file_path=None):
        self.current_image = None
        self.original_image = None
        self.image_path = None
        
        # If a file path is provided during instantiation, load the image
        if file_path:
            self.load_image(file_path)
        
    def load_image(self, image_path):
        """Load an image from file path"""
        try:
            # Convert to Path object and validate
            image_path = Path(image_path)
            if not self._is_valid_image_path(image_path):
                return False
                
            self.image_path = image_path
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
    
    def resize_image(self, new_width, new_height, maintain_aspect_ratio=True, content_aware=False):
        """Resize the current image"""
        if self.current_image is None:
            return False
            
        try:
            if maintain_aspect_ratio and new_width > 0 and new_height > 0:
                # Calculate aspect ratio
                h, w = self.current_image.shape[:2]
                aspect_ratio = w / h
                
                if new_width / new_height > aspect_ratio:
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_height = int(new_width / aspect_ratio)
            
            if content_aware:
                # Use seam carving for content-aware resizing
                self.current_image = self._seam_carving_resize(new_width, new_height)
            else:
                # Standard resizing
                self.current_image = cv.resize(self.current_image, (new_width, new_height), interpolation=cv.INTER_AREA)
            
            return True
        except Exception as e:
            print(f"Error resizing image: {e}")
            return False
    
    def _seam_carving_resize(self, new_width, new_height):
        """Content-aware resizing using seam carving"""
        # This is a simplified implementation - you might want to use a proper seam carving library
        # For now, we'll use standard resizing as a placeholder
        return cv.resize(self.current_image, (new_width, new_height), interpolation=cv.INTER_AREA)
    
    def apply_filter(self, filter_type, **kwargs):
        """Apply various filters to the image"""
        if self.current_image is None:
            return False
            
        try:
            if filter_type == "grayscale":
                self.current_image = cv.cvtColor(self.current_image, cv.COLOR_BGR2GRAY)
                self.current_image = cv.cvtColor(self.current_image, cv.COLOR_GRAY2BGR)
                
            elif filter_type == "blur":
                kernel_size = kwargs.get('kernel_size', 5)
                self.current_image = cv.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)
                
            elif filter_type == "sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                self.current_image = cv.filter2D(self.current_image, -1, kernel)
                
            elif filter_type == "edge_detection":
                gray = cv.cvtColor(self.current_image, cv.COLOR_BGR2GRAY)
                edges = cv.Canny(gray, 100, 200)
                self.current_image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
                
            elif filter_type == "sepia":
                kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
                self.current_image = cv.transform(self.current_image, kernel)
                self.current_image = np.clip(self.current_image, 0, 255)
                
            elif filter_type == "brightness":
                value = kwargs.get('value', 0)
                hsv = cv.cvtColor(self.current_image, cv.COLOR_BGR2HSV)
                h, s, v = cv.split(hsv)
                v = cv.add(v, value)
                v = np.clip(v, 0, 255)
                final_hsv = cv.merge((h, s, v))
                self.current_image = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
                
            elif filter_type == "contrast":
                alpha = kwargs.get('alpha', 1.0)
                self.current_image = cv.convertScaleAbs(self.current_image, alpha=alpha, beta=0)
                
            return True
        except Exception as e:
            print(f"Error applying filter: {e}")
            return False
    
    def detect_objects(self, model_type="default"):
        """Detect objects in the image"""
        if self.current_image is None:
            return None, self.current_image
            
        try:
            # This is a placeholder - you would integrate with actual object detection models
            # like YOLO, SSD, etc.
            if model_type == "default":
                # Simple face detection using Haar cascades as an example
                gray = cv.cvtColor(self.current_image, cv.COLOR_BGR2GRAY)
                face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                result_image = self.current_image.copy()
                for (x, y, w, h) in faces:
                    cv.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv.putText(result_image, 'Face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detection_info = [{"label": "Face", "confidence": 1.0, "bbox": (x, y, w, h)} for (x, y, w, h) in faces]
                return detection_info, result_image
                
            return [], self.current_image
        except Exception as e:
            print(f"Error in object detection: {e}")
            return [], self.current_image
    
    def reset_to_original(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            return True
        return False
    
    def save_image(self, save_path):
        """Save current image to file"""
        if self.current_image is not None:
            try:
                cv.imwrite(save_path, self.current_image)
                return True
            except Exception as e:
                print(f"Error saving image: {e}")
                return False
        return False
    
    def cv_to_qpixmap(self, image=None):
        """Convert OpenCV image to QPixmap"""
        if image is None:
            image = self.current_image
            
        if image is None:
            return QPixmap()
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w if len(rgb_image.shape) == 3 else w
            
            if len(rgb_image.shape) == 3:
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
                
            return QPixmap.fromImage(q_img)
        except Exception as e:
            print(f"Error converting image to QPixmap: {e}")
            return QPixmap()
    
    def get_image_info(self):
        """Get information about the current image"""
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            channels = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1
            return {
                'width': width,
                'height': height,
                'channels': channels,
                'file_path': self.image_path
            }
        return {}