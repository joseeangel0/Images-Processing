import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# NumPy vectorized implementations for Gaussian, Sobel, and Median filters

def apply_gaussian_numpy(image):
    """Applies a 3x3 Gaussian filter using NumPy matrix operations."""
    kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ], dtype=np.float32)
    
    padded = np.pad(image, 1, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(3):
        for j in range(3):
            output += padded[i:i+image.shape[0], j:j+image.shape[1]] * kernel[i, j]
            
    return output.astype(np.uint8)

def apply_sobel_numpy(image):
    """Applies a 3x3 Sobel filter for edge detection using NumPy."""
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    Ky = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    # Pad with 0 for Sobel
    padded = np.pad(image, 1, mode='constant', constant_values=0)
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)
    
    for i in range(3):
        for j in range(3):
            region = padded[i:i+image.shape[0], j:j+image.shape[1]]
            gx += region * Kx[i, j]
            gy += region * Ky[i, j]
            
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def apply_median_numpy(image):
    """Applies a 3x3 Median filter using NumPy sliding_window_view."""
    padded = np.pad(image, 1, mode='edge')
    # Create a view of 3x3 windows into the padded image
    windows = sliding_window_view(padded, (3, 3))
    # Calculate the median over the 3x3 windows
    return np.median(windows, axis=(2, 3)).astype(np.uint8)
