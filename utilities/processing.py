from pathlib import Path
from typing import List
import cv2 as cv
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

class ImageProcessor:
    def __init__(self, file_path):
        self.current_image = None
        self.original_image = None
        self.image_path = Path(file_path)
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
        
    def _seam_carving_resize(self, new_width, new_height, algorithm="Hubble 001", progress_callback=None):
        """Content-aware resizing using seam carving"""
        if algorithm == "Hubble 001":
            current_img = self.current_image.copy()
            
            # Calculate differences
            width_diff = current_img.shape[1] - new_width
            height_diff = current_img.shape[0] - new_height
            
            # Handle width adjustment
            if width_diff != 0:
                current_img = self._adjust_width(current_img, width_diff, progress_callback)
            
            # Handle height adjustment
            if height_diff != 0:
                current_img = self._adjust_height(current_img, height_diff, progress_callback)
            
            # Return result
            self.current_image = current_img
            return current_img
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
    def _find_vertical_seam(self, energy:np.ndarray):
        """Use dynamic programming to find the vertical seam with minimum energy"""
        h, w = energy.shape
        M = energy.copy().astype(np.float64)
        backtrack = np.zeros_like(M, dtype=np.int32)
        
        # Build cumulative energy matrix
        for i in range(1, h):
            for j in range(0, w):
                # Handle boundary cases
                left = M[i-1, j-1] if j > 0 else float('inf')
                middle = M[i-1, j]
                right = M[i-1, j+1] if j < w-1 else float('inf')
                
                # Find minimum energy path
                min_energy = min(left, middle, right)
                M[i, j] += min_energy
                
                # Store backtracking information
                if min_energy == left:
                    backtrack[i, j] = -1  # come from left
                elif min_energy == middle:
                    backtrack[i, j] = 0   # come from middle
                else:
                    backtrack[i, j] = 1   # come from right
        
        # Find the starting point of the seam (minimum energy in last row)
        seam = []
        j = np.argmin(M[-1])
        seam.append(j)
        
        # Backtrack to find the complete seam
        for i in range(h-1, 0, -1):
            j = j + backtrack[i, j]
            seam.append(j)
        
        return seam[::-1]  # reverse to go from top to bottom

    def _remove_vertical_seam(self, img:np.ndarray, seam:List[np.int64]):
        """Remove vertical seam from image"""
        h, w, c = img.shape
        new_img = np.zeros((h, w-1, c), dtype=img.dtype)
        
        for i in range(h):
            j = seam[i]
            new_img[i, :, :] = np.delete(img[i, :, :], j, axis=0)
        
        return new_img

    def _insert_vertical_seam(self, img:np.ndarray, seam:List[np.int64]):
        """Insert vertical seam by duplicating pixel and averaging with neighbors"""
        h, w, c = img.shape
        new_img = np.zeros((h, w + 1, c), dtype=img.dtype)
        
        for i in range(h):
            j = seam[i]
            
            # Copy all pixels before the seam position
            new_img[i, :j, :] = img[i, :j, :]
            
            # Create new pixel at seam position by averaging neighbors
            if j == 0:
                # Left edge case - average with right neighbor only
                new_pixel = (img[i, j, :] + img[i, j+1, :]) // 2
            elif j == w - 1:
                # Right edge case - average with left neighbor only  
                new_pixel = (img[i, j-1, :] + img[i, j, :]) // 2
            else:
                # General case - average left and right neighbors
                new_pixel = (img[i, j-1, :] + img[i, j, :] + img[i, j+1, :]) // 3
            
            # Insert the new pixel and copy the original
            new_img[i, j, :] = new_pixel
            new_img[i, j+1, :] = img[i, j, :]
            
            # Copy all pixels after the seam position (shifted by 1)
            new_img[i, j+2:, :] = img[i, j+1:, :]
        
        return new_img

    def _calculate_energy(self, img):
        """Calculate energy map using Sobel operators"""
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        energy = np.abs(grad_x) + np.abs(grad_y)
        return energy

    def _reduce_width(self, img:np.ndarray, num_seams:int, progress_callback=None):
        """Reduce image width by removing vertical seams"""
        current_img = img.copy()
        
        for seam_num in range(num_seams):
            energy = self._calculate_energy(current_img)
            seam = self._find_vertical_seam(energy)
            current_img = self._remove_vertical_seam(current_img, seam)

            if progress_callback:
                progress_callback(seam_num + 1)
        
        return current_img

    def _enlarge_width(self, img:np.ndarray, num_seams, progress_callback=None):
        """Enlarge image width by inserting vertical seams"""
        current_img = img.copy()
        
        if progress_callback:
            progress_callback(0, num_seams * 2)
        
        # First, find all the seams we would remove (in order of importance)
        seams_to_duplicate = []
        temp_img = current_img.copy()
        
        for i in range(num_seams):
            energy = self._calculate_energy(temp_img)
            seam = self._find_vertical_seam(energy)
            seams_to_duplicate.append(seam)
            temp_img = self._remove_vertical_seam(temp_img, seam)

            if progress_callback:
                progress_callback(i + 1)
        
        # Now insert the seams in reverse order (least important first)
        for i, seam in enumerate(reversed(seams_to_duplicate)):
            current_img = self._insert_vertical_seam(current_img, seam)
            if progress_callback:
                progress_callback(num_seams + i + 1)
        
        return current_img

    def _adjust_width(self, img:np.ndarray, width_diff:int, progress_callback=None):
        """Adjust image width by removing or adding seams"""
        if width_diff > 0:
            # Width reduction
            return self._reduce_width(img, width_diff, progress_callback)
        else:
            # Width enlargement
            return self._enlarge_width(img, abs(width_diff), progress_callback)

    def _adjust_height(self, img:np.ndarray, height_diff:int, progress_callback=None):
        """Adjust image height by removing or adding horizontal seams"""
        if height_diff > 0:
            # Height reduction
            return self._reduce_height(img, height_diff, progress_callback)
        else:
            # Height enlargement  
            return self._enlarge_height(img, abs(height_diff), progress_callback)

    def _reduce_height(self, img:np.ndarray, num_seams:int, progress_callback=None):
        """Reduce image height by removing horizontal seams (using rotation)"""
        # Rotate to treat height as width
        current_img = np.rot90(img, 1)
        
        current_img = self._reduce_width(current_img, num_seams, progress_callback)
        
        # Rotate back
        return np.rot90(current_img, 3)

    def _enlarge_height(self, img:np.ndarray, num_seams:int, progress_callback=None):
        """Enlarge image height by inserting horizontal seams (using rotation)"""
        # Rotate to work with horizontal seams as vertical
        current_img = np.rot90(img, 1)
        
        current_img = self._enlarge_width(current_img, num_seams, progress_callback)
        
        # Rotate back
        return np.rot90(current_img, 3)
    
    
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
        