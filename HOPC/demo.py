#!/usr/bin/env python3
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from phasecong3 import phasecong  # Make sure this dependency is installed

def NCC(v1, v2):
    """
    Calculate Normalized Cross Correlation between two vectors
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        ncc: Normalized Cross Correlation value
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions don't match: {len(v1)} != {len(v2)}")
        
    return np.mean(np.multiply((v1-np.mean(v1)),(v2-np.mean(v2))))/(np.std(v1)*np.std(v2))

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def apply_transformation(img, params, center=None):
    """
    Apply affine transformation to an image
    
    Args:
        img: Input image
        params: Transformation parameters [tx, ty, angle, scale]
        center: Center of rotation (default is image center)
        
    Returns:
        transformed_img: Transformed image
    """
    height, width = img.shape
    if center is None:
        center = (width // 2, height // 2)
    
    # Extract parameters
    tx, ty, angle, scale = params
    
    # Create the transformation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply the transformation
    transformed_img = cv2.warpAffine(img, M, (width, height), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=0)
    return transformed_img

def objective_function(params, img1, img2, cell_size=12, bin_size=8):
    """
    Objective function to minimize (negative NCC to maximize NCC)
    
    Args:
        params: Transformation parameters [tx, ty, angle, scale]
        img1: Reference image (SAR)
        img2: Moving image (Optical)
        cell_size: Cell size for HOPC
        bin_size: Bin size for HOPC
        
    Returns:
        -ncc: Negative NCC value (to maximize NCC)
    """
    # Apply transformation to the optical image
    transformed_img = apply_transformation(img2, params)
    
    # Extract HOPC features from both images
    HOPC1 = HOPC_descriptor(img1, cell_size=cell_size, bin_size=bin_size)
    vector1, _ = HOPC1.extract()
    
    HOPC2 = HOPC_descriptor(transformed_img, cell_size=cell_size, bin_size=bin_size)
    vector2, _ = HOPC2.extract()
    
    # Calculate NCC
    try:
        ncc_value = NCC(vector1, vector2)
        return -ncc_value  # Negative because we want to maximize NCC
    except Exception as e:
        print(f"Error in NCC calculation: {e}")
        return 0  # Return a default value in case of error
    
def optimize_transformation(img_ref, img_moving, initial_params=None):
    """
    Optimize transformation parameters to maximize NCC
    
    Args:
        img_ref: Reference image (SAR)
        img_moving: Moving image (Optical)
        initial_params: Initial transformation parameters [tx, ty, angle, scale]
        
    Returns:
        optimal_params: Optimized transformation parameters
        max_ncc: Maximum NCC value achieved
    """
    if initial_params is None:
        # Default initial parameters: [tx, ty, angle, scale]
        initial_params = [0, 0, 0, 1.0]
    
    # Bounds for parameters: [tx_min, tx_max], [ty_min, ty_max], [angle_min, angle_max], [scale_min, scale_max]
    bounds = [(-50, 50), (-50, 50), (-30, 30), (0.8, 1.2)]
    
    # Perform optimization
    result = minimize(
        objective_function,
        initial_params,
        args=(img_ref, img_moving),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    optimal_params = result.x
    max_ncc = -result.fun  # Convert back to positive NCC
    
    return optimal_params, max_ncc

class HOPC_descriptor():
    """
    Histogram of Phase Congruency (HOPC) descriptor for template matching
    """
    def __init__(self, img, cell_size=16, bin_size=8):
        """
        Initialize HOPC descriptor
        
        Args:
            img: Input image
            cell_size: Size of cell for histogram computation
            bin_size: Number of bins in histogram
        """
        self.img = img
        
        # Normalize image if not already normalized
        if np.max(img) > 1.0:
            self.img = np.sqrt(img / float(np.max(img)))
            self.img = self.img * 255
        
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 // self.bin_size
        self.NumberScales = 4
        self.NumberAngles = 6
        
        # Validate parameters
        assert type(self.bin_size) == int, "bin_size should be integer"
        assert type(self.cell_size) == int, "cell_size should be integer"
        assert type(self.angle_unit) == int, "bin_size should be divisible by 360"

    def extract(self):
        """
        Extract HOPC descriptors from the entire image
        
        Returns:
            hopc_vector: List of HOPC descriptors
            hopc_image: Visualization of HOPC features
        """
        height, width = self.img.shape
        
        # Compute phase congruency
        pc_result = phasecong(self.img, nscale=self.NumberScales, norient=self.NumberAngles)
        pc_magnitude, _, _, pc_angle, _, _, _ = pc_result
        
        # Take absolute magnitude
        pc_magnitude = abs(pc_magnitude)
        
        # Initialize cell vector array
        cell_pc_vector = np.zeros((height // self.cell_size, width // self.cell_size, self.bin_size))
        
        # Process each cell
        for i in range(cell_pc_vector.shape[0]):
            for j in range(cell_pc_vector.shape[1]):
                # Extract cell
                cell_magnitude = pc_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = pc_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                
                # Compute cell histogram
                cell_pc_vector[i][j] = self.cell_pc(cell_magnitude, cell_angle)
        
        # Create visualization
        hopc_image = self.render_pc(np.zeros([height, width]), cell_pc_vector)
        
        # Compute block descriptors
        hopc_vector = []
        for i in range(cell_pc_vector.shape[0] - 1):
            for j in range(cell_pc_vector.shape[1] - 1):
                # Create block from 2x2 cells
                block_vector = []
                block_vector.extend(cell_pc_vector[i][j])
                block_vector.extend(cell_pc_vector[i][j + 1])
                block_vector.extend(cell_pc_vector[i + 1][j])
                block_vector.extend(cell_pc_vector[i + 1][j + 1])
                
                # Normalize block vector
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector)
                
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                    
                hopc_vector.append(block_vector)
                
        return hopc_vector, hopc_image

    def cell_pc(self, cell_magnitude, cell_angle):
        """
        Compute histogram for a cell
        
        Args:
            cell_magnitude: Cell magnitude from phase congruency
            cell_angle: Cell angle from phase congruency
            
        Returns:
            orientation_centers: Histogram of oriented phase congruency
        """
        orientation_centers = [0] * self.bin_size
        
        # Convert angles from radians to degrees if needed
        if np.max(cell_angle) <= 2*np.pi:
            cell_angle = cell_angle * 180.0 / np.pi
            
        # Ensure angles are positive and within [0, 360)
        cell_angle = cell_angle % 360.0
        
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                
                # Get closest bins and interpolation factor
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                
                # Weight contribution to the two closest bins
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
                
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        """
        Find the two closest bins for interpolation
        
        Args:
            gradient_angle: Phase congruency angle in degrees
            
        Returns:
            min_angle: Index of first bin
            max_angle: Index of second bin
            mod: Offset for interpolation
        """
        # Ensure angle is within [0, 360)
        gradient_angle = gradient_angle % 360.0
        
        # Calculate bin index and offset
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        
        # Handle edge case
        if idx == self.bin_size:
            return idx - 1, (idx) % self.bin_size, mod
            
        return idx, (idx + 1) % self.bin_size, mod

    def render_pc(self, image, cell_gradient):
        """
        Visualize cells by drawing lines representing the histogram bins
        
        Args:
            image: Background image for visualization
            cell_gradient: Cell gradients to visualize
            
        Returns:
            image: Image with cell visualization
        """
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                cell_grad = cell_gradient[x][y]
                
                # Normalize cell gradient
                if max_mag > 0:
                    cell_grad = cell_grad / max_mag
                
                angle = 0
                angle_gap = self.angle_unit
                
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    
                    # Calculate line endpoints
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    
                    # Draw line
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    
                    angle += angle_gap
                    
        return image


if __name__ == '__main__':
    # Define the directory for the input images
    #input_dir = '../../DATASET/RemoteSensing/SAR_Optical'
    
    # Ensure the directory exists
   # ensure_directory_exists(input_dir)
    
    # Define the file paths
   # path1 = os.path.join(input_dir, 'SO1b.png')
   # path2 = os.path.join(input_dir, 'SO1a.png')
    
    # Check if the SAR image exists
   # if not os.path.exists(path2):
   #     print(f"Error: SAR image not found at {path2}")
   #     print("Please ensure the SAR.tif file is in the correct directory")
   #     exit(1)
    
    # Read the SAR image
    img2 = cv2.imread('SO1a.png', 0)
    if img2 is None:
       # print(f'Error: Failed to read image from {path2}')
        exit(1)
    
    print(f"Image shape: {img2.shape}")
   # print(f"Processing SAR image: {path2}")
    
    # Extract HOPC features from the SAR image
    HOPC2 = HOPC_descriptor(img2, cell_size=12, bin_size=8)
    vector2, image_hopc2 = HOPC2.extract()
    
    # Save HOPC visualization for SAR image
    cv2.imwrite('SAR_HOPC.png', image_hopc2)
    
    H, W = img2.shape
    
    # Check if the Optical image exists
   # if not os.path.exists(path1):
     #   print(f"Error: Optical image not found at {path1}")
      #  print("Please ensure the Optical.tif file is in the correct directory")
      #  exit(1)
    
    # Read the Optical image
    img1 = cv2.imread('SO1b.png', 0)
    if img1 is None:
     #   print(f'Error: Failed to read image from {path1}')
        exit(1)
    
   #print(f"Processing Optical image: {path1}")
    
    # Visualize the images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img2, cmap='gray')
    plt.title('SAR Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img1, cmap='gray')
    plt.title('Optical Image')
    plt.tight_layout()
    plt.savefig('input_images.png')
    
    # Get image dimensions
    H, W = img2.shape  # SAR image dimensions
    opt_H, opt_W = img1.shape  # Optical image dimensions
    print(f"SAR image size: {H}x{W}")
    print(f"Optical image size: {opt_H}x{opt_W}")
    
    # Determine the best approach based on image dimensions
    if opt_H < H or opt_W < W:
        print("Optical image is smaller than SAR image, resizing optical image")
        img1 = cv2.resize(img1, (W, H))
        opt_H, opt_W = H, W
    
    # Determine target map size
    target_map_size = 64
    
    # Determine maximum valid map size
    max_map_size_row = max(0, opt_H - H)
    max_map_size_col = max(0, opt_W - W)
    map_size = min(target_map_size, min(max_map_size_row, max_map_size_col))
    
    if map_size <= 0:
        # If no valid map size, use padding approach
        print("No valid map size, applying padding")
        pad_rows = max(0, H + target_map_size - opt_H)
        pad_cols = max(0, W + target_map_size - opt_W)
        img1 = np.pad(img1, ((0, pad_rows), (0, pad_cols)), mode='constant')
        map_size = target_map_size
        print(f"Padded optical image size: {img1.shape}")
        
    print(f"Using map size: {map_size}x{map_size}")
    NCC_map = np.zeros([map_size, map_size])
    
    # Define search ranges
    start_row = 0  # Starting row for searching
    start_col = 0  # Starting column for searching
    
    # Calculate NCC for each position with robust checks
    print(f"Calculating NCC map (size: {map_size}x{map_size})...")
    for i in range(map_size):
        for j in range(map_size):
            try:
                # Ensure we stay within image boundaries
                if start_row+i+H > img1.shape[0] or start_col+j+W > img1.shape[1]:
                    continue
                    
                # Extract optical image patch
                patch = img1[start_row+i:start_row+H+i, start_col+j:start_col+W+j]
                
                # Verify patch dimensions
                if patch.shape != img2.shape:
                    print(f"Skipping position ({i}, {j}): Wrong patch dimensions {patch.shape}, expected {img2.shape}")
                    continue
                
                # Extract HOPC features
                HOPC1 = HOPC_descriptor(patch, cell_size=12, bin_size=8)
                vector1, _ = HOPC1.extract()
                
                # Calculate NCC
                if len(vector1) == len(vector2):
                    NCC_map[i, j] = NCC(vector1, vector2)
                else:
                    print(f"Vector length mismatch at ({i}, {j}): {len(vector1)} != {len(vector2)}")
            except Exception as e:
                print(f"Error at position ({i}, {j}): {e}")
                continue
                
        # Print progress
        if (i + 1) % 5 == 0:
            print(f"Progress: {(i+1)/map_size*100:.1f}%")
    
    # Find the maximum correlation position
    i_max, j_max = np.unravel_index(np.argmax(NCC_map), NCC_map.shape)
    print(f"Maximum correlation at position: ({i_max}, {j_max})")
    print(f"Maximum correlation value: {NCC_map[i_max, j_max]:.4f}")
    
    # Extract the best match from the original approach
    best_match_orig = img1[start_row+i_max:start_row+H+i_max, start_col+j_max:start_col+W+j_max]
    
    # Use the best match as the starting point for optimization
    best_match_patch = best_match_orig.copy()
    
    # Initial transformation parameters based on the best match position
    # [tx, ty, angle, scale]
    initial_params = [0, 0, 0, 1.0]  # Start with identity transformation
    
    print("\nOptimizing transformation parameters...")
    optimal_params, max_ncc = optimize_transformation(img2, best_match_patch, initial_params)
    
    print(f"Optimal transformation parameters:")
    print(f"  Translation X: {optimal_params[0]:.2f} pixels")
    print(f"  Translation Y: {optimal_params[1]:.2f} pixels")
    print(f"  Rotation: {optimal_params[2]:.2f} degrees")
    print(f"  Scale: {optimal_params[3]:.3f}")
    print(f"Maximum NCC after optimization: {max_ncc:.4f}")
    
    # Apply the optimal transformation to get the final match
    final_match = apply_transformation(best_match_patch, optimal_params)
    
    # Visualize the results
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 4, 1)
    plt.imshow(img2, cmap='gray')
    plt.title('SAR Image (Reference)')
    
    plt.subplot(1, 4, 2)
    plt.imshow(best_match_orig, cmap='gray')
    plt.title('Best Match (Before Optimization)')
    
    plt.subplot(1, 4, 3)
    plt.imshow(final_match, cmap='gray')
    plt.title('Best Match (After Optimization)')
    
    # Display overlay or difference
    plt.subplot(1, 4, 4)
    
    # Create a pseudo-color overlay
    if final_match.shape == img2.shape:
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[:,:,0] = img2  # Red channel for SAR
        overlay[:,:,1] = final_match  # Green channel for optical
        plt.imshow(overlay)
        plt.title('Overlay (SAR=Red, Optical=Green)')
    else:
        plt.imshow(np.abs(final_match - img2), cmap='hot')
        plt.title('Absolute Difference')
    
    plt.tight_layout()
    plt.savefig('optimized_registration_results.png')
    
    # Compare NCC values before and after optimization
    ncc_before = NCC_map[i_max, j_max]
    ncc_improvement = max_ncc - ncc_before
    print(f"\nNCC improvement: {ncc_improvement:.4f} ({ncc_improvement/ncc_before*100:.2f}%)")
    
    print("Optimization completed successfully!")