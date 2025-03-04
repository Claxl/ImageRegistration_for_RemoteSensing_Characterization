# -*- coding: utf-8 -*-
"""
Fixed Optical-SAR registration demo

@author: zpy (original)
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from HOPC import HOPC_descriptor

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
    
if __name__ == '__main__':
    # Define the directory for the input images
    input_dir = '../DATASET/MRSIDatasets/1optical-optical/'
    
    # Ensure the directory exists
    ensure_directory_exists(input_dir)
    
    # Define the file paths
    path1 = os.path.join(input_dir, '1/pair1-1.jpg')
    path2 = os.path.join(input_dir, '2/pair1-2.jpg')
    
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
    
    # Calculate NCC map
    map_size = 30
    NCC_map = np.zeros([map_size, map_size])
    print(f"Calculating NCC map (size: {map_size}x{map_size})...")
    
    # Define search ranges
    start_row = 217  # Starting row for searching
    start_col = 222  # Starting column for searching
    
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
    
    # Visualize the NCC map
    plt.figure()
    plt.imshow(NCC_map, cmap='jet')
    plt.colorbar()
    plt.title('NCC Map')
    plt.savefig(os.path.join(input_dir, 'ncc_map.png'))
    
    # Find the maximum correlation position
    i_max, j_max = np.unravel_index(np.argmax(NCC_map), NCC_map.shape)
    print(f"Maximum correlation at position: ({i_max}, {j_max})")
    print(f"Maximum correlation value: {NCC_map[i_max, j_max]:.4f}")
    
    # Visualize the best match
    best_match = img1[start_row+i_max:start_row+H+i_max, start_col+j_max:start_col+W+j_max]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img2, cmap='gray')
    plt.title('SAR Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(best_match, cmap='gray')
    plt.title('Best Matched Optical Image')
    
    # Display overlay or difference
    plt.subplot(1, 3, 3)
    
    # Create a pseudo-color overlay
    if best_match.shape == img2.shape:
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[:,:,0] = img2  # Red channel for SAR
        overlay[:,:,1] = best_match  # Green channel for optical
        plt.imshow(overlay)
        plt.title('Overlay (SAR=Red, Optical=Green)')
    else:
        plt.imshow(np.abs(best_match - img2), cmap='hot')
        plt.title('Absolute Difference')
        
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, 'best_match_comparison.png'))
    
    print("Processing completed successfully!")