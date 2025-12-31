"""
utilities/seam_carving_algorithms.py
Modular seam carving implementations for content-aware image resizing
"""

import numpy as np
import cv2 as cv
import time
from typing import List, Callable, Optional, Tuple, Dict, Union
from abc import ABC, abstractmethod


class SeamCarver(ABC):
    """Abstract base class for seam carving algorithms"""
    
    def __init__(self, name: str = "Base"):
        """
        Initialize SeamCarver with algorithm name.
        
        Args:
            name: Algorithm name/identifier
        """
        self.name = name
        self.default_color_space = 'rgb'
        self.performance_metrics = {}

    def get_performance_metrics(self):
        """Get performance metrics for comparative analysis"""
        return {
            'name': self.name,
            'color_space': self.default_color_space,
            'backtrack_type': 'absolute' if getattr(self, 'use_absolute_backtrack', False) else 'relative',
            **self.performance_metrics
        }
        
    def calculate_energy(self, img: np.ndarray, color_space: str = None) -> np.ndarray:
        """
        Return energy map (gradient magnitude) for the given image.
        
        Args:
            img: Input image
            color_space: Override default color space. If None, uses algorithm default.
        """
        if color_space is None:
            color_space = self.default_color_space
            
        if color_space.lower() == 'rgb':
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        elif color_space.lower() == 'bgr':
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            raise ValueError("color_space must be either 'rgb' or 'bgr'")
        
        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
        energy = np.abs(grad_x) + np.abs(grad_y)
        return energy
    
    def _mark_seam_on_image(self, img: np.ndarray, seam: List[int], 
                           color: Tuple[int, int, int] = (0, 255, 0), 
                           thickness: int = 1) -> np.ndarray:
        """
        Mark a seam on the image with a colored line (for visualization only).
        
        Args:
            img: Input image (BGR format)
            seam: List of column indices for the seam (length = height)
            color: BGR color tuple for the seam (default: green)
            thickness: Thickness of the seam line
        
        Returns:
            Image with seam marked (BGR format)
        """
        marked_img = img.copy()
        h, w = img.shape[:2]
        
        for i in range(h):
            j = seam[i]
            # Draw a line for better visibility
            for offset in range(-thickness // 2, thickness // 2 + 1):
                col = j + offset
                if 0 <= col < w:
                    marked_img[i, col] = color
        return marked_img
    
    @abstractmethod
    def _find_vertical_seam(self, energy: np.ndarray) -> List[int]:
        """Find vertical seam using dynamic programming (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def carve(self, img: np.ndarray, target_width: int, target_height: int | None = None, 
          progress_callback: Optional[Callable] = None,
          return_seams: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Main seam carving function (to be implemented by subclasses)"""
        pass
    
    def create_seam_visualization(self, original_img: np.ndarray, seam_info: Dict, 
                      removed_color: Tuple[int, int, int] = (0, 0, 255),  # Red
                      inserted_color: Tuple[int, int, int] = (0, 255, 0),  # Green
                      thickness: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create seam visualizations for removed and inserted seams.
        
        Args:
            original_img: Original image (BGR format)
            seam_info: Dictionary containing seam information
            removed_color: Color for removed seams (BGR)
            inserted_color: Color for inserted seams (BGR)
            thickness: Thickness of seam lines
        
        Returns:
            Tuple of (all_seams_image, removed_seams_image, inserted_seams_image)
        """
        # Create copies for visualization
        all_seams_img = original_img.copy()
        removed_img = original_img.copy()
        inserted_img = original_img.copy()
        
        h, w = original_img.shape[:2]
        
        # Check if seam_info is valid
        if not seam_info or not isinstance(seam_info, dict):
            return all_seams_img, removed_img, inserted_img
        
        # Initialize default structure if missing
        if 'removed_seams' not in seam_info:
            seam_info['removed_seams'] = {'vertical': [], 'horizontal': []}
        if 'inserted_seams' not in seam_info:
            seam_info['inserted_seams'] = {'vertical': [], 'horizontal': []}
        
        # Helper function to mark a VERTICAL seam on an image
        def mark_vertical_seam(img, seam, color):
            """Mark a vertical seam on the image with the given color"""
            marked_img = img.copy()
            if not seam:
                return marked_img
                
            img_h, img_w = img.shape[:2]
            
            # Handle seam length mismatch
            if len(seam) != img_h:
                # Create adjusted seam
                adjusted_seam = []
                for i in range(img_h):
                    if i < len(seam):
                        adjusted_seam.append(seam[i])
                    else:
                        # Extend with last value or 0
                        adjusted_seam.append(seam[-1] if seam else 0)
                seam = adjusted_seam
            
            for i in range(img_h):
                j = seam[i]
                # Ensure j is within bounds and handle edge cases
                if 0 <= j < img_w:
                    # Draw thicker seam for better visibility
                    start_col = max(0, j - thickness // 2)
                    end_col = min(img_w, j + thickness // 2 + 1)
                    marked_img[i, start_col:end_col] = color
            return marked_img
        
        # Helper function to mark a HORIZONTAL seam on an image
        def mark_horizontal_seam(img, seam, color):
            """Mark a horizontal seam on the image with the given color"""
            marked_img = img.copy()
            if not seam:
                return marked_img
            
            img_h, img_w = img.shape[:2]
            
            # CRITICAL FIX: Horizontal seams in original space come from 
            # vertical seams in rotated space. The seam array we have
            # is actually for the ROTATED image, not the original!
            
            # We need to understand what the seam represents:
            # - In rotated space: seam[i] gives column index at row i
            # - After rotating back: this becomes a horizontal seam
            # - But the coordinates need transformation!
            
            # For now, let's handle the simple case where seam length = img_w
            if len(seam) == img_w:
                # Mark as horizontal line (parallel to x-axis)
                for j in range(img_w):
                    i = seam[j]
                    if 0 <= i < img_h:
                        start_row = max(0, i - thickness // 2)
                        end_row = min(img_h, i + thickness // 2 + 1)
                        marked_img[start_row:end_row, j] = color
            else:
                # Try to handle gracefully
                print(f"WARNING: Horizontal seam length mismatch: {len(seam)} != {img_w}")
                
                # If seam is longer than width, truncate
                if len(seam) > img_w:
                    seam = seam[:img_w]
                # If shorter, extend with last value
                else:
                    seam = list(seam) + [seam[-1]] * (img_w - len(seam))
                
                # Mark the adjusted seam
                for j in range(img_w):
                    i = seam[j]
                    if 0 <= i < img_h:
                        start_row = max(0, i - thickness // 2)
                        end_row = min(img_h, i + thickness // 2 + 1)
                        marked_img[start_row:end_row, j] = color
            
            return marked_img
        
        # DEBUG: Print seam information
        print(f"\n=== Seam Visualization Debug ===")
        print(f"Image dimensions: {h}x{w}")
        
        # Check vertical seams
        vert_removed = seam_info['removed_seams'].get('vertical', [])
        vert_inserted = seam_info['inserted_seams'].get('vertical', [])
        print(f"Vertical removed seams: {len(vert_removed)}")
        print(f"Vertical inserted seams: {len(vert_inserted)}")
        
        # Check horizontal seams
        horiz_removed = seam_info['removed_seams'].get('horizontal', [])
        horiz_inserted = seam_info['inserted_seams'].get('horizontal', [])
        print(f"Horizontal removed seams: {len(horiz_removed)}")
        print(f"Horizontal inserted seams: {len(horiz_inserted)}")
        
        # Mark ALL vertical seams on all images
        for seam in vert_removed:
            if seam:
                removed_img = mark_vertical_seam(removed_img, seam, removed_color)
                all_seams_img = mark_vertical_seam(all_seams_img, seam, removed_color)
        
        for seam in vert_inserted:
            if seam:
                inserted_img = mark_vertical_seam(inserted_img, seam, inserted_color)
                all_seams_img = mark_vertical_seam(all_seams_img, seam, inserted_color)
        
        # Mark ALL horizontal seams on all images
        for seam in horiz_removed:
            if seam:
                # DEBUG: Check this seam
                print(f"  Horizontal removed seam length: {len(seam)}")
                
                removed_img = mark_horizontal_seam(removed_img, seam, removed_color)
                all_seams_img = mark_horizontal_seam(all_seams_img, seam, removed_color)
        
        for seam in horiz_inserted:
            if seam:
                # DEBUG: Check this seam
                print(f"  Horizontal inserted seam length: {len(seam)}")
                
                inserted_img = mark_horizontal_seam(inserted_img, seam, inserted_color)
                all_seams_img = mark_horizontal_seam(all_seams_img, seam, inserted_color)
        
        print("=== End Seam Visualization Debug ===\n")
        
        return all_seams_img, removed_img, inserted_img
    


class SeamCarverHubble001(SeamCarver):
    """Seam carving algorithm variant 001: Sequential seam processing"""
    
    def __init__(self):
        super().__init__("Hubble 001")
        self.default_color_space = 'rgb'
        self.use_absolute_backtrack = False
    
    def _find_vertical_seam(self, energy: np.ndarray) -> List[int]:
        """Hubble 001: Uses relative backtracking offsets"""
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
                
                # Store relative backtracking information
                if min_energy == left:
                    backtrack[i, j] = -1  # come from left
                elif min_energy == middle:
                    backtrack[i, j] = 0   # come from middle
                else:
                    backtrack[i, j] = 1   # come from right
        
        # Find the starting point of the seam
        seam = []
        j = np.argmin(M[-1])
        seam.append(j)
        
        # Backtrack to find the complete seam
        for i in range(h-1, 0, -1):
            j = j + backtrack[i, j]
            seam.append(j)
        
        return seam[::-1]  # reverse to go from top to bottom
    
    def _remove_vertical_seam(self, img: np.ndarray, seam: List[int]) -> np.ndarray:
        """Remove vertical seam from image"""
        h, w, c = img.shape
        new_img = np.zeros((h, w-1, c), dtype=img.dtype)
        
        for i in range(h):
            j = seam[i]
            new_img[i, :, :] = np.delete(img[i, :, :], j, axis=0)
        
        return new_img

    def _insert_vertical_seam(self, img: np.ndarray, seam: List[int]) -> np.ndarray:
        """
        Insert vertical seam by averaging with neighbors.
        Creates a natural-looking seam that blends with the image.
        
        Args:
            img: Input image (BGR format)
            seam: Seam to insert
        
        Returns:
            Image with seam inserted (blended naturally)
        """
        h, w, c = img.shape
        new_img = np.zeros((h, w + 1, c), dtype=img.dtype)
        
        for i in range(h):
            j = seam[i]
            
            # Copy all pixels before the seam position
            new_img[i, :j, :] = img[i, :j, :]
            
            # Create new pixel at seam position by averaging neighbors
            if j == 0:
                # Left edge case - average with right neighbor only
                new_pixel = (img[i, j, :].astype(np.float32) + img[i, j+1, :].astype(np.float32)) / 2
            elif j == w - 1:
                # Right edge case - average with left neighbor only  
                new_pixel = (img[i, j-1, :].astype(np.float32) + img[i, j, :].astype(np.float32)) / 2
            else:
                # General case - average left and right neighbors
                new_pixel = (img[i, j-1, :].astype(np.float32) + 
                            img[i, j, :].astype(np.float32) + 
                            img[i, j+1, :].astype(np.float32)) / 3
            
            # Insert the new pixel and copy the original
            new_img[i, j, :] = new_pixel.astype(img.dtype)
            new_img[i, j+1, :] = img[i, j, :]
            
            # Copy all pixels after the seam position (shifted by 1)
            new_img[i, j+2:, :] = img[i, j+1:, :]
        
        return new_img

    def _reduce_width(self, img: np.ndarray, num_seams: int, 
                     progress_callback: Optional[Callable] = None,
                     return_seams: bool = False) -> np.ndarray | Tuple[np.ndarray, List[List[int]]]:
        """
        Reduce image width by removing vertical seams sequentially.
        
        Returns:
            Tuple of (carved_image, list_of_seams_removed)
        """
        current_img = img.copy()
        seams_removed = []
        
        for seam_num in range(num_seams):
            energy = self.calculate_energy(current_img)
            seam = self._find_vertical_seam(energy)
            seams_removed.append(seam)
            current_img = self._remove_vertical_seam(current_img, seam)

            if progress_callback:
                progress_callback(seam_num + 1)
        
        if return_seams:
            return current_img, seams_removed
        return current_img

    def _enlarge_width(self, img: np.ndarray, num_seams: int, 
                      progress_callback: Optional[Callable] = None,
                      return_seams: bool = False) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Enlarge image width by inserting vertical seams sequentially.
        
        Returns:
            Tuple of (carved_image, list_of_seams_inserted)
        """
        current_img = img.copy()
        seams_inserted = []
        
        if progress_callback:
            progress_callback(0, num_seams * 2)
        
        # First, find all the seams we would remove (in order of importance)
        seams_to_duplicate = []
        temp_img = current_img.copy()
        
        for i in range(num_seams):
            energy = self.calculate_energy(temp_img)
            seam = self._find_vertical_seam(energy)
            seams_to_duplicate.append(seam)
            temp_img = self._remove_vertical_seam(temp_img, seam)

            if progress_callback:
                progress_callback(i + 1)
        
        # Now insert the seams in reverse order (least important first)
        for i, seam in enumerate(reversed(seams_to_duplicate)):
            seams_inserted.append(seam)
            current_img = self._insert_vertical_seam(current_img, seam)
            if progress_callback:
                progress_callback(num_seams + i + 1)
        
        if return_seams:
            return current_img, seams_inserted
        return current_img

    def carve(self, img: np.ndarray, target_width: int, target_height: int | None = None, 
              progress_callback: Optional[Callable] = None,
              return_seams: bool = False):
        """
        Main seam carving function to resize image with timing.
        
        Args:
            img: Input image (BGR format)
            target_width: Target width for output
            target_height: Target height for output (optional)
            progress_callback: Callback for progress updates
            return_seams: If True, returns both carved image and seam information
        
        Returns:
            If return_seams=True: (carved_image, seam_info_dict)
            If return_seams=False: carved_image only
        """
        start_time = time.time()
        
        if target_height is None:
            target_height: int = img.shape[0]

        # Initialize seam info
        seam_info = {
            'removed_seams': {'vertical': [], 'horizontal': []},
            'inserted_seams': {'vertical': [], 'horizontal': []},
            'timing': {}
        }

        # Start algorithm-specific timing
        algorithm_start = time.time()
        
        result = self._seam_carving_resize(
            img, target_width, target_height, progress_callback, 
            return_seams, seam_info
        )
        
        # Calculate timing
        algorithm_time = time.time() - algorithm_start
        total_time = time.time() - start_time
        
        seam_info['timing'] = {
            'algorithm': algorithm_time,
            'total': total_time,
            'algorithm_name': self.name
        }
        
        if return_seams:
            # Check if result is already a tuple (image, seam_info)
            if isinstance(result, tuple) and len(result) == 2:
                # Unpack and update seam_info
                carved_img, existing_seam_info = result
                # Merge timing info into existing seam_info
                existing_seam_info['timing'] = seam_info['timing']
                return carved_img, existing_seam_info
            else:
                # Just return result with our seam_info
                return result, seam_info
        else:
            # If result is a tuple, extract just the image
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]  # Return just the image
            else:
                return result  # Return the image directly
    
    def _seam_carving_resize(self, img: np.ndarray, new_width: int, new_height: int, 
                             progress_callback: Optional[Callable] = None,
                             return_seams: bool = False,
                             seam_info: Dict | None = None):
        """Sequential seam carving with seam tracking"""
        current_img = img.copy()
        
        # Calculate differences
        width_diff = current_img.shape[1] - new_width
        height_diff = current_img.shape[0] - new_height
        
        # Initialize seam info
        if seam_info is None:
            seam_info = {
                'removed_seams': {'vertical': [], 'horizontal': []},
                'inserted_seams': {'vertical': [], 'horizontal': []}
            }
        
        # Handle width adjustment
        if width_diff != 0:
            if width_diff > 0:
                # Width reduction
                current_img, removed_seams = self._reduce_width(
                    current_img, width_diff, progress_callback, return_seams=True
                )
                seam_info['removed_seams']['vertical'] = removed_seams
            else:
                # Width enlargement
                current_img, inserted_seams = self._enlarge_width(
                    current_img, abs(width_diff), progress_callback, return_seams=True
                )
                seam_info['inserted_seams']['vertical'] = inserted_seams
        
        # Handle height adjustment
        if height_diff != 0:
            if height_diff > 0:
                # Height reduction (rotate to work with horizontal seams)
                rotated_img = np.rot90(current_img, 1)
                rotated_img, removed_seams = self._reduce_width(
                    rotated_img, height_diff, progress_callback, return_seams=True
                )
                current_img = np.rot90(rotated_img, 3)
                seam_info['removed_seams']['horizontal'] = removed_seams
            else:
                # Height enlargement
                rotated_img = np.rot90(current_img, 1)
                rotated_img, inserted_seams = self._enlarge_width(
                    rotated_img, abs(height_diff), progress_callback, return_seams=True
                )
                current_img = np.rot90(rotated_img, 3)
                seam_info['inserted_seams']['horizontal'] = inserted_seams
        
        if return_seams:
            return current_img, seam_info
        return current_img


class SeamCarverHubble002(SeamCarver):
    """Seam carving algorithm variant 002: Bulk seam processing"""
    
    def __init__(self):
        super().__init__("Hubble 002")
        self.default_color_space = 'rgb'
        self.use_absolute_backtrack = True
    
    def _find_vertical_seam(self, energy: np.ndarray) -> List[int]:
        """Hubble 002: Uses absolute backtracking indices"""
        h, w = energy.shape
        M = energy.copy().astype(np.float64)
        backtrack = np.zeros_like(M, dtype=np.int32)

        # Build cumulative energy matrix
        for i in range(1, h):
            for j in range(w):
                left = M[i - 1, j - 1] if j - 1 >= 0 else np.inf
                middle = M[i - 1, j]
                right = M[i - 1, j + 1] if j + 1 < w else np.inf

                min_prev = min(left, middle, right)
                M[i, j] += min_prev

                # Store absolute backtracking information
                if min_prev == left:
                    backtrack[i, j] = j - 1
                elif min_prev == middle:
                    backtrack[i, j] = j
                else:
                    backtrack[i, j] = j + 1

        # Find position of smallest element in last row
        seam = [0] * h
        j = int(np.argmin(M[-1]))
        seam[-1] = j
        for i in range(h - 1, 0, -1):
            j = int(backtrack[i, j])
            seam[i - 1] = j
        return seam
    
    def _remove_multiple_seams_bulk(self, img: np.ndarray, seams_original: List[List[int]]) -> np.ndarray:
        """
        Remove multiple seams from the ORIGINAL image in one synchronized pass.
        """
        h, w, _ = img.shape
        result_rows = []

        for i in range(h):
            cols_to_remove = {seam[i] for seam in seams_original}
            keep_cols = [col for col in range(w) if col not in cols_to_remove]
            new_row = img[i, keep_cols, :]
            result_rows.append(new_row)

        new_img = np.stack(result_rows, axis=0)
        return new_img

    def _insert_vertical_seam_bulk(self, img: np.ndarray, seam: List[int]) -> np.ndarray:
        """Insert a single vertical seam into img with blending."""
        h, w, c = img.shape
        new_img = np.zeros((h, w + 1, c), dtype=img.dtype)

        for i in range(h):
            j = seam[i]
            if j > 0:
                new_img[i, :j, :] = img[i, :j, :]
            
            # Create blended pixel at seam position
            if j == 0:
                # Left edge case
                new_pixel = (img[i, j, :].astype(np.float32) + img[i, j+1, :].astype(np.float32)) / 2
            elif j == w - 1:
                # Right edge case
                new_pixel = (img[i, j-1, :].astype(np.float32) + img[i, j, :].astype(np.float32)) / 2
            else:
                # General case - average neighbors
                new_pixel = (img[i, j-1, :].astype(np.float32) + 
                            img[i, j, :].astype(np.float32) + 
                            img[i, j+1, :].astype(np.float32)) / 3
            
            new_img[i, j, :] = new_pixel.astype(img.dtype)
            new_img[i, j + 1, :] = img[i, j, :]
            
            if j + 1 < w:
                new_img[i, j + 2:, :] = img[i, j + 1:, :]

        return new_img
    
    def carve(self, img: np.ndarray, target_width: int, target_height: int | None = None, 
              progress_callback: Optional[Callable] = None,
              return_seams: bool = False):
        """
        Main seam carving function to resize image with timing.
        
        Args:
            img: Input image (BGR format)
            target_width: Target width for output
            target_height: Target height for output (optional)
            progress_callback: Callback for progress updates
            return_seams: If True, returns both carved image and seam information
        
        Returns:
            If return_seams=True: (carved_image, seam_info_dict)
            If return_seams=False: carved_image only
        """
        start_time = time.time()
        
        if target_height is None:
            target_height = img.shape[0]

        # Initialize seam info
        seam_info = {
            'removed_seams': {'vertical': [], 'horizontal': []},
            'inserted_seams': {'vertical': [], 'horizontal': []},
            'timing': {}
        }

        # Start algorithm-specific timing
        algorithm_start = time.time()
        
        result = self._seam_carving_resize(
            img, target_width, target_height, progress_callback, 
            return_seams, seam_info
        )
        
        # Calculate timing
        algorithm_time = time.time() - algorithm_start
        total_time = time.time() - start_time
        
        seam_info['timing'] = {
            'algorithm': algorithm_time,
            'total': total_time,
            'algorithm_name': self.name
        }
        
        if return_seams:
            # Check if result is already a tuple (image, seam_info)
            if isinstance(result, tuple) and len(result) == 2:
                # Unpack and update seam_info
                carved_img, existing_seam_info = result
                # Merge timing info into existing seam_info
                existing_seam_info['timing'] = seam_info['timing']
                return carved_img, existing_seam_info
            else:
                # Just return result with our seam_info
                return result, seam_info
        else:
            # If result is a tuple, extract just the image
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]  # Return just the image
            else:
                return result  # Return the image directly
    
    def _seam_carving_resize(self, img: np.ndarray, new_width: int, new_height: int,
                             progress_callback: Optional[Callable] = None,
                             return_seams: bool = False,
                             seam_info: Dict = None):
        """
        Bulk seam carving implementation.
        """
        h, w, _ = img.shape
        width_diff = w - new_width
        height_diff = h - new_height

        if width_diff == 0 and height_diff == 0:
            if return_seams:
                return img, seam_info
            return img

        # Initialize seam info
        if seam_info is None:
            seam_info = {
                'removed_seams': {'vertical': [], 'horizontal': []},
                'inserted_seams': {'vertical': [], 'horizontal': []}
            }

        # Store original dimensions for seam tracking
        orig_h, orig_w = h, w

        # Helper progress updater
        def _progress(step, total=None):
            if progress_callback:
                try:
                    if total is None:
                        progress_callback(step)
                    else:
                        progress_callback(step, total)
                except Exception:
                    pass

        current_img = img.copy()
        
        # Handle width adjustment
        if width_diff != 0:
            if width_diff > 0:
                # Need to reduce width by k seams
                k = int(width_diff)
                temp_img = current_img.copy()
                idx_map = np.tile(np.arange(w), (h, 1))

                seams_original = []  # store seams as original column indices
                for s in range(k):
                    energy = self.calculate_energy(temp_img)
                    seam = self._find_vertical_seam(energy)  # seam indices for temp_img
                    seam_original = [int(idx_map[i, seam[i]]) for i in range(h)]
                    seams_original.append(seam_original)
                    
                    # Update temp image and index map
                    new_temp_rows = []
                    new_idx_rows = []
                    for i in range(h):
                        new_temp_rows.append(np.delete(temp_img[i, :, :], seam[i], axis=0))
                        new_idx_rows.append(np.delete(idx_map[i, :], seam[i], axis=0))
                    temp_img = np.stack(new_temp_rows, axis=0)
                    idx_map = np.stack(new_idx_rows, axis=0)

                    _progress(s + 1, k)
                
                # Remove all seams at once
                current_img = self._remove_multiple_seams_bulk(current_img, seams_original)
                seam_info['removed_seams']['vertical'] = seams_original
                
                _progress(k, k)
            else:
                # Width enlargement
                k = int(-width_diff)
                temp_img = current_img.copy()
                idx_map = np.tile(np.arange(w), (h, 1))
                seams_original = []
                for s in range(k):
                    energy = self.calculate_energy(temp_img)
                    seam = self._find_vertical_seam(energy)
                    seam_original = [int(idx_map[i, seam[i]]) for i in range(h)]
                    seams_original.append(seam_original)
                    
                    new_temp_rows = []
                    new_idx_rows = []
                    for i in range(h):
                        new_temp_rows.append(np.delete(temp_img[i, :, :], seam[i], axis=0))
                        new_idx_rows.append(np.delete(idx_map[i, :], seam[i], axis=0))
                    temp_img = np.stack(new_temp_rows, axis=0)
                    idx_map = np.stack(new_idx_rows, axis=0)
                    _progress(s + 1, k)
                
                # Insert seams one by one (with blending)
                for s_idx, seam_orig in enumerate(seams_original):
                    seam_current = []
                    for i in range(h):
                        inserts_before = 0
                        for prev in seams_original[:s_idx]:
                            if prev[i] <= seam_orig[i]:
                                inserts_before += 1
                        current_pos = seam_orig[i] + inserts_before
                        # Clamp to valid range
                        current_pos = max(0, min(current_img.shape[1] - 1, current_pos))
                        seam_current.append(current_pos)

                    current_img = self._insert_vertical_seam_bulk(current_img, seam_current)
                    _progress(s_idx + 1, k)

                seam_info['inserted_seams']['vertical'] = seams_original

        # Handle height adjustment (rotate for horizontal seams)
        if height_diff != 0:
            # Rotate to work with horizontal seams
            rotated_img = np.rot90(current_img, 1)
            
            # Call recursively - pass None for seam_info to get new one for rotated space
            if return_seams:
                rotated_result, rotated_seam_info = self._seam_carving_resize(
                    rotated_img, new_height, new_width, progress_callback, 
                    return_seams, None  # Create new seam info for rotated space
                )
                
                # Convert rotated seams back to original orientation
                # Rotated vertical seams are horizontal in original, and vice versa
                for seam in rotated_seam_info['removed_seams']['vertical']:
                    # This seam was vertical in rotated space = horizontal in original
                    seam_info['removed_seams']['horizontal'].append(seam)
                    
                for seam in rotated_seam_info['inserted_seams']['vertical']:
                    # This seam was vertical in rotated space = horizontal in original
                    seam_info['inserted_seams']['horizontal'].append(seam)
                    
                # Note: In rotated space, we only track vertical seams since we rotate
                # to convert horizontal to vertical
            else:
                rotated_result = self._seam_carving_resize(
                    rotated_img, new_height, new_width, progress_callback, 
                    return_seams, None
                )
            
            current_img = np.rot90(rotated_result, 3)

        if return_seams:
            return current_img, seam_info
        return current_img

