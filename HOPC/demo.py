import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from HOPC import HOPC_descriptor
from scipy.optimize import minimize

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

if __name__ == '__main__':
    # Define the directory for the input images
    input_dir = 'DATASET/RemoteSensing/SAR_Optical'
    
    # Ensure the directory exists
    ensure_directory_exists(input_dir)
    
    # Define the file paths
    path1 = os.path.join(input_dir, 'SO1b.png')
    path2 = os.path.join(input_dir, 'SO1a.png')
    
    # Check if the SAR image exists
    if not os.path.exists(path2):
        print(f"Error: SAR image not found at {path2}")
        print("Please ensure the SAR.tif file is in the correct directory")
        exit(1)
    
    # Read the SAR image
    img2 = cv2.imread(path2, 0)
    if img2 is None:
        print(f'Error: Failed to read image from {path2}')
        exit(1)
    
    print(f"Image shape: {img2.shape}")
    print(f"Processing SAR image: {path2}")
    
    # Extract HOPC features from the SAR image
    HOPC2 = HOPC_descriptor(img2, cell_size=12, bin_size=8)
    vector2, image_hopc2 = HOPC2.extract()
    
    # Save HOPC visualization for SAR image
    cv2.imwrite(os.path.join(input_dir, 'SAR_HOPC.png'), image_hopc2)
    
    H, W = img2.shape
    
    # Check if the Optical image exists
    if not os.path.exists(path1):
        print(f"Error: Optical image not found at {path1}")
        print("Please ensure the Optical.tif file is in the correct directory")
        exit(1)
    
    # Read the Optical image
    img1 = cv2.imread(path1, 0)
    if img1 is None:
        print(f'Error: Failed to read image from {path1}')
        exit(1)
    
    print(f"Processing Optical image: {path1}")
    
    # Visualize the images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img2, cmap='gray')
    plt.title('SAR Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img1, cmap='gray')
    plt.title('Optical Image')
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, 'input_images.png'))
    
    # Calculate NCC map (original approach)
    map_size = 96
    NCC_map = np.zeros([map_size, map_size])
    print(f"Calculating NCC map (size: {map_size}x{map_size})...")
    
    # Define search ranges
    start_row = 0  # Starting row for searching
    start_col = 0  # Starting column for searching
    
    # Check if search area is within the optical image
    if start_row + H + map_size > img1.shape[0] or start_col + W + map_size > img1.shape[1]:
        print("Warning: Search area may extend beyond the optical image boundaries")
        print(f"Optical image size: {img1.shape}")
        print(f"SAR image size: {img2.shape}")
        print(f"Search area needs: {start_row + H + map_size} rows, {start_col + W + map_size} columns")
    
    # Try to adjust search area if needed
    if start_row + H + map_size > img1.shape[0]:
        start_row = max(0, img1.shape[0] - H - map_size)
        print(f"Adjusted starting row to: {start_row}")
    
    if start_col + W + map_size > img1.shape[1]:
        start_col = max(0, img1.shape[1] - W - map_size)
        print(f"Adjusted starting column to: {start_col}")
    
    # Calculate NCC for each position
    for i in range(map_size):
        for j in range(map_size):
            try:
                # Extract optical image patch
                patch = img1[start_row+i:start_row+H+i, start_col+j:start_col+W+j]
                
                # Skip if patch has wrong dimensions
                if patch.shape[0] != H or patch.shape[1] != W:
                    print(f"Skipping position ({i}, {j}): Wrong patch dimensions {patch.shape}")
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
    plt.savefig(os.path.join(input_dir, 'optimized_registration_results.png'))
    
    # Compare NCC values before and after optimization
    ncc_before = NCC_map[i_max, j_max]
    ncc_improvement = max_ncc - ncc_before
    print(f"\nNCC improvement: {ncc_improvement:.4f} ({ncc_improvement/ncc_before*100:.2f}%)")
    
    print("Optimization completed successfully!")