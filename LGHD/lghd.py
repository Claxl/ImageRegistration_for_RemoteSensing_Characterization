"""
LGHD implementation for phase congruency-based image descriptors
"""
import numpy as np

def lghd(eo_list):
    """
    Log-Gabor Histogram Descriptor
    
    Implementation of the log-Gabor histogram descriptor compatible with the 
    given phase congruency implementation.
    
    Args:
        eo_list: List of complex filter responses for different orientations
                for a single scale
    
    Returns:
        descriptor: LGHD descriptor for the given filter responses (96-dimensional)
    """
    # Validate input
    if not eo_list or len(eo_list) == 0:
        return np.zeros(96)
    
    # Get dimensions from first valid filter response
    for resp in eo_list:
        if resp is not None and hasattr(resp, 'shape'):
            ys, xs = resp.shape
            break
    else:
        return np.zeros(96)
    
    # Create array to hold magnitude of filter responses for each orientation
    im2 = np.zeros((ys, xs, len(eo_list)))
    
    # Store absolute values of filter responses
    for i in range(len(eo_list)):
        if eo_list[i] is not None and hasattr(eo_list[i], 'shape'):
            if eo_list[i].shape == (ys, xs):
                im2[:, :, i] = np.abs(eo_list[i])
    
    # Find orientation with maximum response at each pixel
    mmax = np.max(im2, axis=2)
    maxp = np.argmax(im2, axis=2) + 1  # 1-based indexing for consistency
    
    # Set pixels with no response to a special value
    maxp[mmax == 0] = len(eo_list) + 1
    
    # Initialize histogram array (4x4 subregions x 6 orientations)
    eoh = np.zeros((4, 4, len(eo_list)))
    
    # Compute histograms for each 4x4 subregion
    for j in range(4):
        for i in range(4):
            # Calculate boundaries for current subregion
            y_start = int(round((j) * ys / 4))
            y_end = int(round((j + 1) * ys / 4))
            x_start = int(round((i) * xs / 4))
            x_end = int(round((i + 1) * xs / 4))
            
            # Ensure boundaries are within image dimensions
            y_end = min(y_end, ys)
            x_end = min(x_end, xs)
            
            if y_start >= y_end or x_start >= x_end:
                continue
            
            # Extract subregion orientation map
            clip = maxp[y_start:y_end, x_start:x_end]
            
            # Count occurrences of each orientation in the subregion
            for orient in range(len(eo_list)):
                eoh[j, i, orient] = np.sum(clip == orient + 1)
    
    # Flatten the histogram into a descriptor vector
    descriptor = np.zeros(4 * 4 * len(eo_list))  # 4x4 regions x number of orientations
    idx = 0
    for i in range(4):
        for j in range(4):
            for o in range(len(eo_list)):
                descriptor[idx] = eoh[i, j, o]
                idx += 1
    
    # Normalize descriptor
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor