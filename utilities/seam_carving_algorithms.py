import numpy as np
import cv2 as cv
from typing import List, Callable, Optional


class SeamCarving:
    def __init__(self, algorithm: str = "Hubble 001"):
        """
        Initialize SeamCarving with specified algorithm.
        
        Args:
            algorithm: "Hubble 001" or "Hubble 002" or "Hubble 003" for different algorithm variants
        """
        self.algorithm = algorithm
        
        # Set defaults based on algorithm
        if algorithm == "Hubble 001":
            self.default_color_space = 'rgb' 
            self.use_absolute_backtrack = False
        elif algorithm == "Hubble 002":
            self.default_color_space = 'bgr' 
            self.use_absolute_backtrack = True
        elif algorithm == "Hubble 003":
            self.default_color_space = 'bgr' 
            self.use_absolute_backtrack = False
        else:
            raise ValueError("Algorithm must be 'Hubble 001' or 'Hubble 002' or 'Hubble 003' ")
    
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
    
    # Hubble 003 methods - Combination of bulk processing and energy-based protection
    def _calculate_local_std_hubble003(self, gray: np.ndarray, window_size: int = 7) -> np.ndarray:
        
        """ Returns a local standard deviation map """

        h, w = gray.shape
        half_window = window_size // 2
        # Pad the image
        padded = np.pad(gray.astype(np.float64), half_window, mode='reflect')
        
        # Fast mean calculation using 2D convolution
        gray_padded_sq = padded ** 2
        
        # Manual convolution for mean
        mean = np.zeros((h, w), dtype=np.float64)
        mean_sq = np.zeros((h, w), dtype=np.float64)
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+window_size, j:j+window_size]
                window_sq = gray_padded_sq[i:i+window_size, j:j+window_size]
                mean[i, j] = np.mean(window)
                mean_sq[i, j] = np.mean(window_sq)
        
        # Calculate variance and std
        variance = mean_sq - mean ** 2
        variance[variance < 0] = 0  # Handle numerical errors
        return np.sqrt(variance)
    
    def calculate_energy_hubble003(self, img: np.ndarray) -> np.ndarray:
        """
        Calculate energy map with MAXIMUM edge and structure protection.
        Enhanced to prevent wavy distortions in vertical structures.
        
        Returns an energy map with heavily protected structures
        """
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate gradients with kernel larger than hubble 001 for better edge detection
        grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
        grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
        
        # Base energy: gradient magnitude
        energy = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10) * 255
        
        # Multi-scale edge detection with MORE aggressive settings
        edges1 = cv.Canny(gray.astype(np.uint8), 15, 60)    # Extra sensitive
        edges2 = cv.Canny(gray.astype(np.uint8), 30, 90)    # Very sensitive
        edges3 = cv.Canny(gray.astype(np.uint8), 50, 150)   # Sensitive
        edges4 = cv.Canny(gray.astype(np.uint8), 100, 200)  # Strong
        
        # Vectorized maximum across all edge maps
        combined_edges = np.maximum.reduce([edges1, edges2, edges3, edges4])
        
        # EXTREME energy boost at edges 
        edge_energy = combined_edges.astype(np.float64) / 255.0
        energy *= (1.0 + edge_energy * 60.0)
        
        # Protect VERTICAL structures MORE AGGRESSIVELY (towers, building edges)
        vertical_strength = np.abs(grad_x)
        v_threshold = np.percentile(vertical_strength, 50)  
        vertical_mask = (vertical_strength > v_threshold).astype(np.uint8)
        
        # MUCH wider vertical protection with larger kernel
        kernel_v = np.ones((1, 13), np.uint8)  
        vertical_protected = cv.dilate(vertical_mask, kernel_v, iterations=6) 
        energy[vertical_protected > 0] *= 20.0  
        
        # Extra protection for STRONG vertical edges (tower edges)
        strong_v_threshold = np.percentile(vertical_strength, 75)
        strong_vertical = (vertical_strength > strong_v_threshold).astype(np.uint8)
        kernel_strong_v = np.ones((1, 17), np.uint8)
        strong_vertical_protected = cv.dilate(strong_vertical, kernel_strong_v, iterations=8)
        energy[strong_vertical_protected > 0] *= 40.0
        
        # Protect horizontal structures (grass line, horizons, castle base)
        horizontal_strength = np.abs(grad_y)
        h_threshold = np.percentile(horizontal_strength, 50)  
        horizontal_mask = (horizontal_strength > h_threshold).astype(np.uint8)
        
        # Wider dilation for horizontal features
        kernel_h = np.ones((13, 1), np.uint8)
        horizontal_protected = cv.dilate(horizontal_mask, kernel_h, iterations=6) 
        energy[horizontal_protected > 0] *= 18.0  
        
        # Special protection for castle base area
        bottom_region = int(h * 0.45)  
        energy[-bottom_region:, :] *= 3.5  
        
        # Detect strong horizontal edges in bottom region
        bottom_horizontal = horizontal_strength[-bottom_region:, :]
        strong_h_threshold = np.percentile(bottom_horizontal, 65)  
        strong_horizontal = bottom_horizontal > strong_h_threshold
        
        # Create mask for entire image (vectorized)
        strong_h_mask = np.zeros((h, w), dtype=np.uint8)
        strong_h_mask[-bottom_region:, :] = strong_horizontal.astype(np.uint8)
        
        # Apply MASSIVE protection to critical horizontal structures (ensures main structures are protected)
        kernel_critical = np.ones((17, 5), np.uint8)  
        critical_protected = cv.dilate(strong_h_mask, kernel_critical, iterations=8)  
        energy[critical_protected > 0] *= 35.0  
        
        # SUPER-protect intersections (e.g. corners)
        corners = (horizontal_protected > 0) & (vertical_protected > 0)
        corners_critical = (critical_protected > 0) & (vertical_protected > 0)
        corners_super = (critical_protected > 0) & (strong_vertical_protected > 0)
        
        energy[corners] *= 10.0  
        energy[corners_critical] *= 40.0 
        energy[corners_super] *= 60.0  
        
        # Add texture-based protection
        texture = self._calculate_local_std_hubble003(gray, window_size=7)
        texture_normalized = (texture - texture.min()) / (texture.max() - texture.min() + 1e-10)
        high_texture = texture_normalized > 0.40  
        energy[high_texture] *= 5.0  
        
        # Detect color discontinuities (e.g. castle vs sky boundary)
        if len(img.shape) == 3:
            # Calculate color gradients for all channels at once
            color_diff = np.zeros((h, w), dtype=np.float64)
            for c in range(3):
                grad_cx = cv.Sobel(img[:, :, c], cv.CV_64F, 1, 0, ksize=5)
                grad_cy = cv.Sobel(img[:, :, c], cv.CV_64F, 0, 1, ksize=5)
                color_diff += np.sqrt(grad_cx**2 + grad_cy**2)
            
            color_diff /= 3.0
            color_threshold = np.percentile(color_diff, 55)  
            strong_color_boundaries = (color_diff > color_threshold).astype(np.uint8)
            
            kernel_color = np.ones((9, 9), np.uint8)  
            color_protected = cv.dilate(strong_color_boundaries, kernel_color, iterations=4)  
            energy[color_protected > 0] *= 8.0  
        
        # Detect continuous vertical lines (tower edges) using Hough transform
        edges_for_hough = cv.Canny(gray.astype(np.uint8), 50, 150)
        lines = cv.HoughLinesP(edges_for_hough, 1, np.pi/180, threshold=50, 
                            minLineLength=h//4, maxLineGap=10)
        
        if lines is not None:
            line_mask = np.zeros((h, w), dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly vertical
                if abs(x2 - x1) < abs(y2 - y1) * 0.3:  
                    cv.line(line_mask, (x1, y1), (x2, y2), 255, thickness=2)
            
            # Apply huge energy boost
            kernel_lines = np.ones((1, 15), np.uint8)
            line_protected = cv.dilate(line_mask, kernel_lines, iterations=7)
            energy[line_protected > 0] *= 50.0  
        
        # Minimal smoothing to preserve all detail
        energy = cv.GaussianBlur(energy, (3, 3), 0.3) 
        
        return energy
    
    def _find_vertical_seam_hubble003(self, energy: np.ndarray) -> List[int]:
        """
        Find lowest energy vertical seam 
            
        Returns a list of column indices representing the seam
        """
        h, w = energy.shape
        M = energy.copy().astype(np.float64)
        backtrack = np.zeros_like(M, dtype=np.int32)
        
        # Vectorized cumulative energy calculation
        for i in range(1, h):
            # Create shifted versions for vectorized comparison
            left = np.roll(M[i-1], 1)
            left[0] = np.inf  # Prevent wrapping
            
            center = M[i-1]
            
            right = np.roll(M[i-1], -1)
            right[-1] = np.inf  # Prevent wrapping
            
            # Add smoothness penalties
            left += 8 
            right += 8
            
            # Stack and find minimum (vectorized)
            options = np.stack([left, center, right])
            min_indices = np.argmin(options, axis=0)
            min_values = np.min(options, axis=0)
            
            # Update cumulative energy
            M[i] += min_values
            
            # Store backtrack indices
            backtrack[i] = min_indices - 1 + np.arange(w)
            backtrack[i, 0] = max(0, backtrack[i, 0])
            backtrack[i, -1] = min(w-1, backtrack[i, -1])
        
        # Backtrack to find seam
        seam = np.zeros(h, dtype=np.int32)
        seam[-1] = np.argmin(M[-1])
        
        for i in range(h - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]
        
        return seam.tolist()
    
    def _remove_seams_bulk_hubble003(self, img: np.ndarray, seams: List[List[int]]) -> np.ndarray:
        """
        Remove multiple seams from image in one pass.
            
        Returns an image with seams removed
        """
        h, w, c = img.shape
        num_seams = len(seams)
        
        # Create removal mask (vectorized)
        mask = np.ones((h, w), dtype=bool)
        for i in range(h):
            cols_to_remove = [seam[i] for seam in seams]
            mask[i, cols_to_remove] = False
        
        # Apply mask to all channels at once
        result = np.zeros((h, w - num_seams, c), dtype=img.dtype)
        for i in range(h):
            result[i] = img[i, mask[i], :]
        
        return result
    
    def _insert_seams_bulk_hubble003(self, img: np.ndarray, seams: List[List[int]]) -> np.ndarray:
        """
        Insert multiple seams into image in one pass.
          
        Returns an image with seams inserted
        """
        h, w, c = img.shape
        
        # Build result row by row
        result_rows = []
        
        for row_idx in range(h):
            # Get all seam positions for this row and sort them
            seam_positions = sorted([seam[row_idx] for seam in seams])
            
            row = img[row_idx]
            new_row_pixels = []
            
            src_idx = 0  # Current position in source row
            
            for seam_pos in seam_positions:
                # Copy pixels from src_idx up to and including seam_pos
                for j in range(src_idx, seam_pos + 1):
                    new_row_pixels.append(row[j])
                
                # Calculate and add the averaged pixel
                if seam_pos == 0:
                    avg_pixel = (row[0].astype(np.int32) + row[min(1, w-1)].astype(np.int32)) // 2
                elif seam_pos >= w - 1:
                    avg_pixel = (row[w-2].astype(np.int32) + row[w-1].astype(np.int32)) // 2
                else:
                    avg_pixel = (row[seam_pos-1].astype(np.int32) + 
                            row[seam_pos].astype(np.int32) + 
                            row[seam_pos+1].astype(np.int32)) // 3
                
                new_row_pixels.append(avg_pixel.astype(img.dtype))
                
                src_idx = seam_pos + 1
            
            # Copy remaining pixels
            for j in range(src_idx, w):
                new_row_pixels.append(row[j])
            
            # Stack pixels into row and add to result
            result_rows.append(np.stack(new_row_pixels, axis=0))
        
        return np.stack(result_rows, axis=0)
    
    def _process_dimension_hubble003(self, img: np.ndarray, target_size: int, 
                  is_width: bool, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Process one dimension (width or height) using bulk seam carving.
        
        Returns a processed image
        """
        current_size = img.shape[1] if is_width else img.shape[0]
        diff = current_size - target_size
        
        if diff == 0:
            return img
        
        # For height processing, rotate image
        if not is_width:
            img = np.rot90(img, 1)
            # After rotation, we need to process the WIDTH to change the final HEIGHT
            current_size = img.shape[1]  
            # Recalculate diff based on rotated dimensions
            diff = current_size - target_size
        
        h, w, c = img.shape  
        
        if diff > 0:
            # Reduction: remove seams
            k = int(diff)
            temp_img = img.copy()
            idx_map = np.tile(np.arange(w), (h, 1))
            seams_original = []
            
            for s in range(k):
                energy = self.calculate_energy_hubble003(temp_img)
                seam = self._find_vertical_seam_hubble003(energy)
                
                seam_original = [int(idx_map[i, seam[i]]) for i in range(h)]
                seams_original.append(seam_original)
                
                # Use vectorized deletion
                temp_w = temp_img.shape[1]
                mask = np.ones(temp_w, dtype=bool)
                
                new_temp_rows = []
                new_idx_rows = []
                for i in range(h):
                    mask[:] = True
                    mask[seam[i]] = False
                    new_temp_rows.append(temp_img[i, mask, :])
                    new_idx_rows.append(idx_map[i, mask])
                
                temp_img = np.stack(new_temp_rows, axis=0)
                idx_map = np.stack(new_idx_rows, axis=0)
                
                if progress_callback:
                    progress_callback(s + 1, k)
            
            # Remove all seams at once
            img = self._remove_seams_bulk_hubble003(img, seams_original)
            
        else:
            # Seam insertion
            k = int(-diff)
            temp_img = img.copy()
            idx_map = np.tile(np.arange(w), (h, 1))
            seams_original = []
            
            for s in range(k):
                energy = self.calculate_energy_hubble003(temp_img)
                seam = self._find_vertical_seam_hubble003(energy)
                
                seam_original = [int(idx_map[i, seam[i]]) for i in range(h)]
                seams_original.append(seam_original)
                
                # Use vectorized deletion
                temp_w = temp_img.shape[1]
                mask = np.ones(temp_w, dtype=bool)
                
                new_temp_rows = []
                new_idx_rows = []
                for i in range(h):
                    mask[:] = True
                    mask[seam[i]] = False
                    new_temp_rows.append(temp_img[i, mask, :])
                    new_idx_rows.append(idx_map[i, mask])
                
                temp_img = np.stack(new_temp_rows, axis=0)
                idx_map = np.stack(new_idx_rows, axis=0)
                
                if progress_callback:
                    progress_callback(s + 1, k)
            
            # Insert all seams at once
            img = self._insert_seams_bulk_hubble003(img, seams_original)
        
        # Rotate back if processing height
        if not is_width:
            img = np.rot90(img, 3)
        
        return img
    

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
        elif self.algorithm == "Hubble 003":
            return self._seam_carving_resize_hubble003(img, target_width, target_height, progress_callback)
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
    
    def _seam_carving_resize_hubble003(self, img: np.ndarray, target_width: int, target_height: int = None, progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Resize image using advanced seam carving.
        """
        if target_height is None:
            target_height = img.shape[0]
        
        current_img = img.copy()
        
        print(f"\nAdvanced Seam Carving")
        print(f"Original size: {img.shape[1]}×{img.shape[0]}")
        print(f"Target size: {target_width}×{target_height}")
        
        # Process width
        if current_img.shape[1] != target_width:
            print(f"Processing width: {current_img.shape[1]} → {target_width}")
            current_img = self._process_dimension_hubble003(current_img, target_width, True, progress_callback)
        
        # Process height
        if current_img.shape[0] != target_height:
            print(f"Processing height: {current_img.shape[0]} → {target_height}")
            current_img = self._process_dimension_hubble003(current_img, target_height, False, progress_callback)
        
        print("Done!")
        return current_img