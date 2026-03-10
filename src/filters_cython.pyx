import numpy as np
cimport numpy as cnp
import cython
from libc.math cimport sqrt

# Cython optimized implementations for Gaussian, Sobel, and Median filters

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_gaussian_cython(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.zeros((height, width), dtype=np.uint8)
    
    cdef float kernel[3][3] 
    kernel[0][0] = 1.0/16.0; kernel[0][1] = 2.0/16.0; kernel[0][2] = 1.0/16.0
    kernel[1][0] = 2.0/16.0; kernel[1][1] = 4.0/16.0; kernel[1][2] = 2.0/16.0
    kernel[2][0] = 1.0/16.0; kernel[2][1] = 2.0/16.0; kernel[2][2] = 1.0/16.0

    cdef int i, j, ki, kj
    cdef float pixel_val
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            pixel_val = 0.0
            for ki in range(3):
                for kj in range(3):
                    pixel_val += image[i - 1 + ki, j - 1 + kj] * kernel[ki][kj]
            output[i, j] = <cnp.uint8_t>pixel_val
            
    # Copy edges
    for i in range(height):
        output[i, 0] = image[i, 0]
        output[i, width - 1] = image[i, width - 1]
    for j in range(width):
        output[0, j] = image[0, j]
        output[height - 1, j] = image[height - 1, j]
        
    return output

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_sobel_cython(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.zeros((height, width), dtype=np.uint8)
    
    cdef float Kx[3][3]
    Kx[0][0] = -1; Kx[0][1] = 0; Kx[0][2] = 1
    Kx[1][0] = -2; Kx[1][1] = 0; Kx[1][2] = 2
    Kx[2][0] = -1; Kx[2][1] = 0; Kx[2][2] = 1
    
    cdef float Ky[3][3]
    Ky[0][0] = -1; Ky[0][1] = -2; Ky[0][2] = -1
    Ky[1][0] = 0;  Ky[1][1] = 0;  Ky[1][2] = 0
    Ky[2][0] = 1;  Ky[2][1] = 2;  Ky[2][2] = 1

    cdef int i, j, ki, kj
    cdef float gx, gy, val, magnitude
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = 0.0
            gy = 0.0
            for ki in range(3):
                for kj in range(3):
                    val = image[i - 1 + ki, j - 1 + kj]
                    gx += val * Kx[ki][kj]
                    gy += val * Ky[ki][kj]
            
            magnitude = sqrt(gx * gx + gy * gy)
            if magnitude > 255.0:
                magnitude = 255.0
            output[i, j] = <cnp.uint8_t>magnitude
            
    return output

cdef void sort_9(cnp.uint8_t arr[9]) nogil:
    cdef int i, j
    cdef cnp.uint8_t temp
    for i in range(1, 9):
        temp = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > temp:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = temp

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_median_cython(cnp.ndarray[cnp.uint8_t, ndim=2] image):
    cdef int height = image.shape[0]
    cdef int width = image.shape[1]
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] output = np.zeros((height, width), dtype=np.uint8)
    
    cdef int i, j, ki, kj, idx
    cdef cnp.uint8_t neighbors[9]
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            idx = 0
            for ki in range(3):
                for kj in range(3):
                    neighbors[idx] = image[i - 1 + ki, j - 1 + kj]
                    idx += 1
            
            sort_9(neighbors)
            output[i, j] = neighbors[4]
            
    # Copy edges
    for i in range(height):
        output[i, 0] = image[i, 0]
        output[i, width - 1] = image[i, width - 1]
    for j in range(width):
        output[0, j] = image[0, j]
        output[height - 1, j] = image[height - 1, j]
        
    return output
