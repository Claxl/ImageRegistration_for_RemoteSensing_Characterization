import numpy as np
import cv2
from scipy import ndimage

def ehd(im, edge_map=None, num_blocks=3, norm_flag=0):
    """
    Edge Histogram Descriptor
    
    Args:
        im: Input image
        edge_map: Pre-computed edge map (optional)
        num_blocks: Number of blocks in each dimension
        norm_flag: Normalization flag (0: no normalization, 1: normalize)
    
    Returns:
        eh: Edge histogram descriptor
    """
    # If edge map is not provided, compute it
    if edge_map is None:
        # Compute edge map using Sobel operators
        sobel_x = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and direction
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        
        # Threshold the magnitude to get edge map
        edge_map = magnitude > np.mean(magnitude)
    
    # Get image dimensions
    h, w = edge_map.shape if edge_map is not None else im.shape
    
    # Compute block size
    block_h = h // num_blocks
    block_w = w // num_blocks
    
    # Initialize descriptor
    eh = np.zeros(5 * num_blocks * num_blocks)
    
    # For each block in the image
    for i in range(num_blocks):
        for j in range(num_blocks):
            # Extract block
            y1 = i * block_h
            y2 = min((i + 1) * block_h, h)
            x1 = j * block_w
            x2 = min((j + 1) * block_w, w)
            
            block = edge_map[y1:y2, x1:x2] if edge_map is not None else im[y1:y2, x1:x2]
            
            # Compute histogram for the block
            # 5 directions: vertical, horizontal, 45 degrees, 135 degrees, non-directional
            if edge_map is None:
                # Compute direction for the block
                sobel_x_block = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y_block = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
                magnitude_block = np.sqrt(sobel_x_block**2 + sobel_y_block**2)
                direction_block = np.arctan2(sobel_y_block, sobel_x_block) * 180 / np.pi
                edge_block = magnitude_block > np.mean(magnitude_block)
            else:
                # Use existing edge map and direction
                block_coords = np.where(block)
                if len(block_coords[0]) > 0:  # If there are any edges in the block
                    direction_block = direction[y1:y2, x1:x2]
                    edge_block = block
                else:
                    # No edges, skip direction calculation
                    eh[(i * num_blocks + j) * 5:(i * num_blocks + j + 1) * 5] = [0, 0, 0, 0, 0]
                    continue
            
            # Count edges for each direction
            vertical = np.sum((direction_block > -22.5) & (direction_block <= 22.5) & edge_block)
            horizontal = np.sum(((direction_block > 67.5) & (direction_block <= 112.5)) & edge_block)
            diagonal_45 = np.sum(((direction_block > 22.5) & (direction_block <= 67.5)) & edge_block)
            diagonal_135 = np.sum(((direction_block > 112.5) & (direction_block <= 157.5)) | 
                                  ((direction_block > -157.5) & (direction_block <= -112.5)) & edge_block)
            non_directional = np.sum(edge_block) - vertical - horizontal - diagonal_45 - diagonal_135
            
            # Store in descriptor
            idx = (i * num_blocks + j) * 5
            eh[idx:idx+5] = [vertical, horizontal, diagonal_45, diagonal_135, non_directional]
    
    # Normalize if required
    if norm_flag and np.linalg.norm(eh) > 0:
        eh = eh / np.linalg.norm(eh)
    
    return eh

def lghd(eo_list):
    """
    Log-Gabor Histogram Descriptor
    
    Args:
        eo_list: List of complex filter responses for different orientations
        
    Returns:
        eh: LGHD descriptor
    """
    # Get dimensions
    ys, xs = eo_list[0].shape
    
    # Create a 3D array for the filter responses
    im2 = np.zeros((ys, xs, 6))
    
    # Fill the array with the magnitudes of filter responses
    for i in range(6):
        im2[:, :, i] = np.abs(eo_list[i])
    
    # Find the maximum response orientation at each pixel
    mmax = np.max(im2, axis=2)
    maxp = np.argmax(im2, axis=2) + 1  # 1-based indexing to match MATLAB
    
    # Set background pixels (where max is 0) to a special value
    maxp[mmax == 0] = 7
    
    # Initialize histograms for each subregion
    eoh = np.zeros((4, 4, 6))
    
    # Compute histograms for each 4x4 subregion
    for j in range(4):
        for i in range(4):
            # Extract subregion
            y_start = int(round(j * ys / 4))
            y_end = int(round((j + 1) * ys / 4))
            x_start = int(round(i * xs / 4))
            x_end = int(round((i + 1) * xs / 4))
            
            clip = maxp[y_start:y_end, x_start:x_end]
            
            # Compute histogram for each orientation (1-6)
            for bin_idx in range(6):
                eoh[j, i, bin_idx] = np.sum(clip == bin_idx + 1)
    
    # Build descriptor by flattening the histograms
    d1 = []
    for i in range(4):
        for j in range(4):
            for bin_idx in range(6):
                d1.append(eoh[i, j, bin_idx])
    
    # Convert to numpy array
    d1 = np.array(d1)
    
    # Normalize
    norm_value = np.linalg.norm(d1)
    if norm_value != 0:
        d1 = d1 / norm_value
    
    return d1

def hysthresh(im, T1, T2):
    """
    Hysteresis thresholding
    
    Args:
        im: Image to be thresholded (assumed to be non-negative)
        T1: Upper threshold value
        T2: Lower threshold value
    
    Returns:
        bw: Thresholded image (containing values 0 or 1)
    """
    # Ensure T1 > T2
    if T1 < T2:
        T1, T2 = T2, T1
    
    # Find points above lower threshold
    above_T2 = im > T2
    
    # Find points above upper threshold
    above_T1 = np.zeros_like(above_T2)
    above_T1[im > T1] = 1
    
    # Label the connected components in above_T1
    labeled_array, num_features = ndimage.label(above_T1)
    
    # For each connected component in above_T1, find the corresponding pixels in above_T2
    bw = np.zeros_like(above_T2, dtype=bool)
    for i in range(1, num_features+1):
        # Create a mask for the current connected component
        mask = labeled_array == i
        # Find the corresponding region in above_T2 using the mask
        if np.any(mask & above_T2):
            # Create a temporary mask for the connected component in above_T2
            temp = np.zeros_like(above_T2, dtype=bool)
            temp[mask] = True
            # Grow the region using binary dilation with above_T2 as mask
            temp = ndimage.binary_dilation(temp, structure=np.ones((3,3)), iterations=-1, mask=above_T2)
            # Add to the output
            bw = bw | temp
    
    return bw

def nonmaxsup(inimage, orient, radius):
    """
    Non-maxima suppression
    
    Args:
        inimage: Image to be non-maxima suppressed
        orient: Image containing feature normal orientation angles in degrees (0-180)
        radius: Distance in pixel units to be looked at on each side of each pixel
    
    Returns:
        im: Non maximally suppressed image
        location: Complex valued image holding subpixel locations of edge points
    """
    if inimage.shape != orient.shape:
        raise ValueError('image and orientation image are of different sizes')
    
    if radius < 1:
        raise ValueError('radius must be >= 1')
    
    rows, cols = inimage.shape
    im = np.zeros((rows, cols))
    location = np.zeros((rows, cols), dtype=complex)
    
    iradius = int(np.ceil(radius))
    
    # Precalculate x and y offsets relative to centre pixel for each orientation angle
    angle = np.radians(np.arange(181))  # 0 to 180 degrees in radians
    xoff = radius * np.cos(angle)
    yoff = radius * np.sin(angle)
    
    hfrac = xoff - np.floor(xoff)  # Fractional offset of xoff
    vfrac = yoff - np.floor(yoff)  # Fractional offset of yoff
    
    # Convert orientations to indices (0-180)
    orient = np.fix(orient).astype(int)
    
    # Run through the image interpolating grey values on each side of the centre pixel
    for row in range(iradius, rows - iradius):
        for col in range(iradius, cols - iradius):
            or_idx = orient[row, col]
            if or_idx >= len(angle):  # Safety check
                continue
                
            # First point
            x = col + xoff[or_idx]
            y = row - yoff[or_idx]
            
            fx = int(np.floor(x))
            cx = int(np.ceil(x))
            fy = int(np.floor(y))
            cy = int(np.ceil(y))
            
            # Boundary check
            if fx < 0 or cx >= cols or fy < 0 or cy >= rows:
                continue
            
            tl = inimage[fy, fx]
            tr = inimage[fy, cx]
            bl = inimage[cy, fx]
            br = inimage[cy, cx]
            
            # Bilinear interpolation
            upperavg = tl + hfrac[or_idx] * (tr - tl)
            loweravg = bl + hfrac[or_idx] * (br - bl)
            v1 = upperavg + vfrac[or_idx] * (loweravg - upperavg)
            
            if inimage[row, col] > v1:
                # Second point
                x = col - xoff[or_idx]
                y = row + yoff[or_idx]
                
                fx = int(np.floor(x))
                cx = int(np.ceil(x))
                fy = int(np.floor(y))
                cy = int(np.ceil(y))
                
                # Boundary check
                if fx < 0 or cx >= cols or fy < 0 or cy >= rows:
                    continue
                
                tl = inimage[fy, fx]
                tr = inimage[fy, cx]
                bl = inimage[cy, fx]
                br = inimage[cy, cx]
                
                # Bilinear interpolation
                upperavg = tl + hfrac[or_idx] * (tr - tl)
                loweravg = bl + hfrac[or_idx] * (br - bl)
                v2 = upperavg + vfrac[or_idx] * (loweravg - upperavg)
                
                if inimage[row, col] > v2:
                    # This is a local maximum
                    im[row, col] = inimage[row, col]
                    
                    # Subpixel localization
                    c = inimage[row, col]
                    a = (v1 + v2) / 2 - c
                    b = a + c - v1
                    
                    # Location where maxima of fitted parabola occurs
                    if a != 0:
                        r = -b / (2 * a)
                        location[row, col] = complex(row + r * yoff[or_idx], col - r * xoff[or_idx])
    
    # Thin the non-maximally suppressed image
    skel = ndimage.morphology.binary_skeletonize(im > 0)
    im = im * skel
    location = location * skel
    
    return im, location