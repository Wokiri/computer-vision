from typing import List, Callable, Optional
import numpy as np
import cv2 as cv


def calculate_energy(img: np.ndarray, color_space: str = 'rgb') -> np.ndarray:
    """Return energy map (gradient magnitude) for the given image.
    
    Args:
        img: Input image (BGR or RGB format)
        color_space: Color space of input image - 'rgb' or 'bgr'. 
                    Defaults to 'rgb'.
    
    Returns:
        Energy map (gradient magnitude) as numpy array
    """
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


def find_vertical_seam(energy: np.ndarray) -> List[int]:
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