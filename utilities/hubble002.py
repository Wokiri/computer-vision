# hubble002.py
"""
hubble002.py
A bulk-seam-carving implementation (Hubble 002) that finds multiple
vertical seams, maps them back to original column indices, and then
removes (or inserts) them in bulk while attempting to preserve edges.
Function exported:
    seam_carving_resize_hubble002(self, new_width, new_height, algorithm, progress_callback)
"""
from typing import List, Callable, Optional
import numpy as np
import cv2 as cv


def _calculate_energy(img: np.ndarray) -> np.ndarray:
    """Return energy map (gradient magnitude) for the given BGR image."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


def _find_vertical_seam(energy: np.ndarray) -> List[int]:
    """
    Dynamic programming to find one vertical seam (top->bottom) of minimum cumulative energy.
    Returns list of column indices (one per row).
    """
    h, w = energy.shape
    M = energy.copy().astype(np.float64)
    backtrack = np.zeros_like(M, dtype=np.int32)

    # build cumulative energy M
    for i in range(1, h):
        for j in range(w):
            left = M[i - 1, j - 1] if j - 1 >= 0 else np.inf
            middle = M[i - 1, j]
            right = M[i - 1, j + 1] if j + 1 < w else np.inf

            min_prev = min(left, middle, right)
            M[i, j] += min_prev

            if min_prev == left:
                backtrack[i, j] = j - 1
            elif min_prev == middle:
                backtrack[i, j] = j
            else:
                backtrack[i, j] = j + 1

    # find position of smallest element in last row
    seam = [0] * h
    j = int(np.argmin(M[-1]))
    seam[-1] = j
    for i in range(h - 1, 0, -1):
        j = int(backtrack[i, j])
        seam[i - 1] = j
    return seam

def _remove_seam_from_image(img: np.ndarray, seam: List[int]) -> np.ndarray:
    """Remove a single vertical seam from img."""
    h, w, c = img.shape
    new_img = np.zeros((h, w - 1, c), dtype=img.dtype)
    for i in range(h):
        j = seam[i]
        new_img[i, :, :] = np.delete(img[i, :, :], j, axis=0)
    return new_img


def _remove_multiple_seams_from_original(img: np.ndarray, seams_original: List[List[int]]) -> np.ndarray:
    """
    Remove multiple seams from the ORIGINAL image in one synchronized pass.
    seams_original is a list of seams; each seam is a list of original-column-indices (length = h).
    For each row, collect columns to remove (set of indices) and then build the new row.
    """
    h, w, c = img.shape
    k = len(seams_original)
    result_rows = []

    for i in range(h):
        cols_to_remove = {seam[i] for seam in seams_original}
        keep_cols = [col for col in range(w) if col not in cols_to_remove]
        new_row = img[i, keep_cols, :]
        result_rows.append(new_row)

    new_img = np.stack(result_rows, axis=0)
    return new_img


def _insert_vertical_seam_into_image(img: np.ndarray, seam: List[int]) -> np.ndarray:
    """Insert a single vertical seam into img; seam indices are with respect to current img columns."""
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


def seam_carving_resize_hubble002(self,
                                  new_width: int,
                                  new_height: int,
                                  algorithm: Optional[str] = "Hubble 002",
                                  progress_callback: Optional[Callable] = None) -> np.ndarray:
    """
    Hubble 002: Bulk seam carving.
    - Finds multiple seams (k) by repeatedly finding seams on a temporary copy and
      recording seam indices mapped to original columns using an index map.
    - Removes (or inserts) them in a single coordinated pass on the original image.
    """
    # use the same image as the ImageProcessor instance
    img = self.current_image.copy()
    h, w, c = img.shape

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
    if width_diff > 0:
        # need to reduce width by k seams
        k = int(width_diff)
        temp_img = img.copy()
        h_t, w_t, c_t = temp_img.shape

        idx_map = np.tile(np.arange(w), (h, 1))

        seams_original = []  # store seams as original column indices
        for s in range(k):
            energy = _calculate_energy(temp_img)
            seam = _find_vertical_seam(energy)  # seam indices for temp_img
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
        new_img = _remove_multiple_seams_from_original(img, seams_original)
        _progress(k, k)
        return new_img

    elif width_diff < 0:
        k = int(-width_diff)
        temp_img = img.copy()
        idx_map = np.tile(np.arange(w), (h, 1))
        seams_original = []
        for s in range(k):
            energy = _calculate_energy(temp_img)
            seam = _find_vertical_seam(energy)
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
            h_c, w_c, c_c = current_img.shape
            insert_offsets = np.zeros(w + s_idx, dtype=int)  # upper bound
            prev_inserted = []
            for prev in seams_original[:s_idx]:
                prev_inserted.append(prev)
            def original_to_current_index(orig_idx, prev_inserted_list):
                offset = 0
                for prev in prev_inserted_list:
                    pass
                return None
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

            
            current_img = _insert_vertical_seam_into_image(current_img, seam_current)
            _progress(s_idx + 1, k)

        return current_img

    else:
        
        return img
