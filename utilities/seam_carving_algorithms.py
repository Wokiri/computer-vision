import numpy as np
import cv2 as cv
from typing import List, Callable, Optional

class SeamCarving:
    def __init__(self, algorithm: str = "Hubble 001"):
        """
        Initialize SeamCarving with specified algorithm.
        
        Args:
            algorithm: "Hubble 001" or "Hubble 002" for different algorithm variants
        """
        self.algorithm = algorithm
        
        # Set defaults based on algorithm
        if algorithm == "Hubble 001":
            self.default_color_space = 'rgb'
            self.use_absolute_backtrack = False
        elif algorithm == "Hubble 002":
            self.default_color_space = 'bgr' 
            self.use_absolute_backtrack = True
        else:
            raise ValueError("Algorithm must be 'Hubble 001' or 'Hubble 002'")
    
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
    
    def _find_vertical_seam(self, energy: np.ndarray) -> List[int]:
        """
        Find vertical seam using dynamic programming.
        Uses different implementations based on algorithm.
        """
        if self.use_absolute_backtrack:
            return self._find_vertical_seam_absolute(energy)
        else:
            return self._find_vertical_seam_relative(energy)
    
    def _find_vertical_seam_relative(self, energy: np.ndarray) -> List[int]:
        """Algorithm Hubble 001: Uses relative backtracking offsets"""
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
    
    def _find_vertical_seam_absolute(self, energy: np.ndarray) -> List[int]:
        """Algorithm Hubble 002: Uses absolute backtracking indices"""
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

    # Hubble 001 methods - Sequential seam processing
    def _remove_vertical_seam_hubble001(self, img: np.ndarray, seam: List[int]) -> np.ndarray:
        """Hubble 001: Remove vertical seam from image"""
        h, w, c = img.shape
        new_img = np.zeros((h, w-1, c), dtype=img.dtype)
        
        for i in range(h):
            j = seam[i]
            new_img[i, :, :] = np.delete(img[i, :, :], j, axis=0)
        
        return new_img

    def _insert_vertical_seam_hubble001(self, img: np.ndarray, seam: List[int]) -> np.ndarray:
        """Hubble 001: Insert vertical seam by duplicating pixel and averaging with neighbors"""
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

    def _reduce_width_hubble001(self, img: np.ndarray, num_seams: int, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Hubble 001: Reduce image width by removing vertical seams sequentially"""
        current_img = img.copy()
        
        for seam_num in range(num_seams):
            energy = self.calculate_energy(current_img)
            seam = self._find_vertical_seam(energy)
            current_img = self._remove_vertical_seam_hubble001(current_img, seam)

            if progress_callback:
                progress_callback(seam_num + 1)
        
        return current_img

    def _enlarge_width_hubble001(self, img: np.ndarray, num_seams: int, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Hubble 001: Enlarge image width by inserting vertical seams sequentially"""
        current_img = img.copy()
        
        if progress_callback:
            progress_callback(0, num_seams * 2)
        
        # First, find all the seams we would remove (in order of importance)
        seams_to_duplicate = []
        temp_img = current_img.copy()
        
        for i in range(num_seams):
            energy = self.calculate_energy(temp_img)
            seam = self._find_vertical_seam(energy)
            seams_to_duplicate.append(seam)
            temp_img = self._remove_vertical_seam_hubble001(temp_img, seam)

            if progress_callback:
                progress_callback(i + 1)
        
        # Now insert the seams in reverse order (least important first)
        for i, seam in enumerate(reversed(seams_to_duplicate)):
            current_img = self._insert_vertical_seam_hubble001(current_img, seam)
            if progress_callback:
                progress_callback(num_seams + i + 1)
        
        return current_img

    def _reduce_height_hubble001(self, img: np.ndarray, num_seams: int, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Hubble 001: Reduce image height by removing horizontal seams (using rotation)"""
        # Rotate to treat height as width
        current_img = np.rot90(img, 1)
        
        current_img = self._reduce_width_hubble001(current_img, num_seams, progress_callback)
        
        # Rotate back
        return np.rot90(current_img, 3)

    def _enlarge_height_hubble001(self, img: np.ndarray, num_seams: int, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Hubble 001: Enlarge image height by inserting horizontal seams (using rotation)"""
        # Rotate to work with horizontal seams as vertical
        current_img = np.rot90(img, 1)
        
        current_img = self._enlarge_width_hubble001(current_img, num_seams, progress_callback)
        
        # Rotate back
        return np.rot90(current_img, 3)

    # Hubble 002 methods - Bulk seam processing
    def _remove_multiple_seams_bulk_hubble002(self, img: np.ndarray, seams_original: List[List[int]]) -> np.ndarray:
        """
        Hubble 002: Remove multiple seams from the ORIGINAL image in one synchronized pass.
        seams_original is a list of seams; each seam is a list of original-column-indices (length = h).
        For each row, collect columns to remove (set of indices) and then build the new row.
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

    def _insert_vertical_seam_bulk_hubble002(self, img: np.ndarray, seam: List[int]) -> np.ndarray:
        """Hubble 002: Insert a single vertical seam into img; seam indices are with respect to current img columns."""
        h, w, c = img.shape
        new_img = np.zeros((h, w + 1, c), dtype=img.dtype)

        for i in range(h):
            j = seam[i]
            if j > 0:
                new_img[i, :j, :] = img[i, :j, :]
            if j == 0:
                new_pixel = (img[i, j, :].astype(np.int32) + img[i, j + 1, :].astype(np.int32)) // 2
            elif j == w - 1:
                new_pixel = (img[i, j - 1, :].astype(np.int32) + img[i, j, :].astype(np.int32)) // 2
            else:
                new_pixel = (img[i, j - 1, :].astype(np.int32) + img[i, j, :].astype(np.int32) + img[i, j + 1, :].astype(np.int32)) // 3
            new_img[i, j, :] = new_pixel.astype(img.dtype)
            new_img[i, j + 1, :] = img[i, j, :]
            if j + 1 < w:
                new_img[i, j + 2:, :] = img[i, j + 1:, :]

        return new_img

    # Main carving method
    def carve(self, img: np.ndarray, target_width: int, target_height: int = None, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Main seam carving function to resize image.
        
        Args:
            img: Input image
            target_width: Target width for output
            target_height: Target height for output (optional)
            progress_callback: Callback for progress updates
        """
        if target_height is None:
            target_height = img.shape[0]

        print()
        print(f"Using algorithm {self.algorithm}")
        print(f"Default color space: {self.default_color_space}")
        print(f"Backtrack method: {'absolute' if self.use_absolute_backtrack else 'relative'}")

        if self.algorithm == "Hubble 001":
            return self._seam_carving_resize_hubble001(img, target_width, target_height, progress_callback)
        elif self.algorithm == "Hubble 002":
            return self._seam_carving_resize_hubble002(img, target_width, target_height, progress_callback)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _seam_carving_resize_hubble001(self, img: np.ndarray, new_width: int, new_height: int, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Hubble 001: Sequential seam carving"""
        current_img = img.copy()
        
        # Calculate differences
        width_diff = current_img.shape[1] - new_width
        height_diff = current_img.shape[0] - new_height
        
        # Handle width adjustment
        if width_diff != 0:
            if width_diff > 0:
                # Width reduction
                current_img = self._reduce_width_hubble001(img, width_diff, progress_callback)
            else:
                # Width enlargement
                current_img = self._enlarge_width_hubble001(img, abs(width_diff), progress_callback)
        
        # Handle height adjustment
        if height_diff != 0:
            if height_diff > 0:
                # Height reduction
                current_img = self._reduce_height_hubble001(img, height_diff, progress_callback)
            else:
                # Height enlargement  
                current_img = self._enlarge_height_hubble001(img, abs(height_diff), progress_callback)
        
        return current_img
    
    def _seam_carving_resize_hubble002(self, img: np.ndarray, new_width: int, new_height: int, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Hubble 002: Bulk seam carving.
        - Finds multiple seams (k) by repeatedly finding seams on a temporary copy and
          recording seam indices mapped to original columns using an index map.
        - Removes (or inserts) them in a single coordinated pass on the original image.
        """
        h, w, _ = img.shape

        # compute diffs
        width_diff = w - new_width
        height_diff = h - new_height

        # handle no-op
        if width_diff == 0 and height_diff == 0:
            return img

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

        # Handle width adjustment for Hubble 002
        if width_diff != 0:
            if width_diff > 0:
                # need to reduce width by k seams
                k = int(width_diff)
                temp_img = img.copy()

                idx_map = np.tile(np.arange(w), (h, 1))

                seams_original = []  # store seams as original column indices
                for s in range(k):
                    energy = self.calculate_energy(temp_img)
                    seam = self._find_vertical_seam(energy)  # seam indices for temp_img
                    seam_original = [int(idx_map[i, seam[i]]) for i in range(h)]
                    seams_original.append(seam_original)
                    new_temp_rows = []
                    new_idx_rows = []
                    for i in range(h):
                        # delete the seam column from the row
                        new_temp_rows.append(np.delete(temp_img[i, :, :], seam[i], axis=0))
                        new_idx_rows.append(np.delete(idx_map[i, :], seam[i], axis=0))
                    temp_img = np.stack(new_temp_rows, axis=0)
                    idx_map = np.stack(new_idx_rows, axis=0)

                    _progress(s + 1, k)
                img = self._remove_multiple_seams_bulk_hubble002(img, seams_original)
                _progress(k, k)
            else:
                # Width enlargement for Hubble 002
                k = int(-width_diff)
                temp_img = img.copy()
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
                
                current_img = img.copy()
                for s_idx, seam_orig in enumerate(seams_original):
                    seam_current = []
                    for i in range(h):
                        inserts_before = 0
                        for prev in seams_original[:s_idx]:
                            if prev[i] <= seam_orig[i]:
                                inserts_before += 1
                        current_pos = seam_orig[i] + inserts_before
                        # clamp
                        if current_pos < 0:
                            current_pos = 0
                        if current_pos > current_img.shape[1] - 1:
                            current_pos = current_img.shape[1] - 1
                        seam_current.append(current_pos)

                    current_img = self._insert_vertical_seam_bulk_hubble002(current_img, seam_current)
                    _progress(s_idx + 1, k)

                img = current_img

        # Handle height adjustment (rotate for horizontal seams)
        if height_diff != 0:
            # Rotate to work with horizontal seams
            img = np.rot90(img, 1)
            img = self._seam_carving_resize_hubble002(img, new_height, new_width, progress_callback)
            img = np.rot90(img, 3)

        return img