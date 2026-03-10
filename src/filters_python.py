import math

# Pure Python implementations for Gaussian, Sobel, and Median filters
# We expect 'image' to be a 2D list of integers.

def apply_gaussian_python(image):
    """Applies a 3x3 Gaussian filter."""
    height = len(image)
    width = len(image[0]) if height > 0 else 0
    
    kernel = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    
    output = [[0]*width for _ in range(height)]
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            pixel_val = 0.0
            for ki in range(3):
                for kj in range(3):
                    pixel_val += image[i - 1 + ki][j - 1 + kj] * kernel[ki][kj]
            output[i][j] = int(pixel_val)
            
    # Edge handling: copy original borders
    for i in range(height):
        output[i][0] = image[i][0]
        output[i][width - 1] = image[i][width - 1]
    for j in range(width):
        output[0][j] = image[0][j]
        output[height - 1][j] = image[height - 1][j]
        
    return output

def apply_sobel_python(image):
    """Applies a 3x3 Sobel filter for edge detection."""
    height = len(image)
    width = len(image[0]) if height > 0 else 0
    
    Kx = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    
    Ky = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]
    
    output = [[0]*width for _ in range(height)]
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = 0.0
            gy = 0.0
            for ki in range(3):
                for kj in range(3):
                    val = image[i - 1 + ki][j - 1 + kj]
                    gx += val * Kx[ki][kj]
                    gy += val * Ky[ki][kj]
            
            magnitude = math.sqrt(gx*gx + gy*gy)
            # Clip value to 255
            if magnitude > 255:
                magnitude = 255
                
            output[i][j] = int(magnitude)
            
    # Leave borders as 0 (as initialized)
    return output

def apply_median_python(image):
    """Applies a 3x3 Median filter."""
    height = len(image)
    width = len(image[0]) if height > 0 else 0
    
    output = [[0]*width for _ in range(height)]
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Collect neighborhood
            neighbors = []
            for ki in range(3):
                for kj in range(3):
                    neighbors.append(image[i - 1 + ki][j - 1 + kj])
            
            # Sort to find median
            neighbors.sort()
            median_val = neighbors[4] # 9 elements, middle is index 4
            output[i][j] = median_val
            
    # Edge handling: copy original borders
    for i in range(height):
        output[i][0] = image[i][0]
        output[i][width - 1] = image[i][width - 1]
    for j in range(width):
        output[0][j] = image[0][j]
        output[height - 1][j] = image[height - 1][j]
        
    return output
