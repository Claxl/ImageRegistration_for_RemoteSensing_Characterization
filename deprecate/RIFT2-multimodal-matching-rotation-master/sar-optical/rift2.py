import numpy as np
import cv2
from scipy import ndimage
from skimage import feature
import matplotlib.pyplot as plt
from phasecong3 import phasecong
import time

def feature_detection(image, scales, orientations, num_points):
    """
    Detect feature points using phase congruency.
    
    Args:
        image: Input image
        scales: Number of wavelet scales
        orientations: Number of filter orientations
        num_points: Maximum number of points to detect
    
    Returns:
        keypoints: Detected keypoints
        m: Maximum moment of phase congruency (edge strength)
        eo: Filter convolution results
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to float
    image = image.astype(np.float64)
    
    # Compute phase congruency
    m, min_moment, _, _, _, eo, _ = phasecong(image, scales, orientations, 3, 1.6, 0.75, 3, 1)
    
    # Normalize m to [0,1]
    a, b = np.max(m), np.min(m)
    m = (m - b) / (a - b)
    
    # Detect FAST corners on minimum moment image (equivalent to cornerness)
    min_moment_norm = (min_moment - np.min(min_moment)) / (np.max(min_moment) - np.min(min_moment))
    corners = feature.corner_fast(min_moment_norm, threshold=0.0001)
    corner_y, corner_x = np.where(corners)
    
    # Combine corner and edge points, up to num_points
    keypoints_x = corner_x
    keypoints_y = corner_y
    
    # Get total number of points
    num_kps = len(keypoints_x)
    
    # If we need more points, add edge points from maximum moment
    if num_kps < num_points:
        # Use Canny edge detector on maximum moment
        edges = cv2.Canny(np.uint8(m * 255), 50, 150)
        edge_y, edge_x = np.where(edges > 0)
        
        # Add edge points (up to the remaining slots)
        if len(edge_x) > 0:
            remaining = num_points - num_kps
            if len(edge_x) > remaining:
                # Select a subset of edge points
                idx = np.random.choice(len(edge_x), remaining, replace=False)
                edge_x = edge_x[idx]
                edge_y = edge_y[idx]
                
            keypoints_x = np.concatenate([keypoints_x, edge_x])
            keypoints_y = np.concatenate([keypoints_y, edge_y])
    
    # Limit to num_points if we have too many
    if len(keypoints_x) > num_points:
        idx = np.random.choice(len(keypoints_x), num_points, replace=False)
        keypoints_x = keypoints_x[idx]
        keypoints_y = keypoints_y[idx]
    
    keypoints = np.vstack((keypoints_x, keypoints_y)).T
    
    return keypoints, m, eo

def kpts_orientation(keypoints, image, compute_orientation=True, patch_size=96):
    """
    Calculate dominant orientation for keypoints.
    
    Args:
        keypoints: Array of keypoints [x, y]
        image: Input image
        compute_orientation: Whether to compute orientation
        patch_size: Size of patch for orientation calculation
    
    Returns:
        oriented_keypoints: Keypoints with orientation [x, y, angle]
    """
    if compute_orientation:
        # Number of orientation bins
        n = 24
        ORI_PEAK_RATIO = 0.8
        
        # Compute gradient
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        gradient_angle[gradient_angle < 0] += 360
    
    oriented_keypoints = []
    
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        r = int(patch_size / 2)
        
        # Check if patch fits within image boundaries
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(x + r, image.shape[1])
        y2 = min(y + r, image.shape[0])
        
        if y2 - y1 != patch_size or x2 - x1 != patch_size:
            continue
        
        if compute_orientation:
            # Extract patch for orientation calculation
            patch_magnitude = gradient_magnitude[y1:y2, x1:x2]
            patch_angle = gradient_angle[y1:y2, x1:x2]
            
            # Compute histogram of orientations
            hist = np.zeros(n)
            for i in range(patch_magnitude.shape[0]):
                for j in range(patch_magnitude.shape[1]):
                    bin_idx = int(patch_angle[i, j] * n / 360.0) % n
                    hist[bin_idx] += patch_magnitude[i, j]
            
            # Smooth histogram
            hist = np.convolve(np.hstack((hist[-2:], hist, hist[:2])), np.array([1, 4, 6, 4, 1])/16.0, mode='valid')
            
            # Find peaks in orientation histogram
            max_idx = np.argmax(hist)
            max_val = hist[max_idx]
            angles = []
            
            # Add the maximum peak
            angles.append((max_idx * 360.0 / n) % 360)
            
            # Check if other peaks are within ratio of the maximum peak
            for i in range(n):
                if i == max_idx:
                    continue
                if hist[i] > max_val * ORI_PEAK_RATIO:
                    angles.append((i * 360.0 / n) % 360)
            
            # Add keypoint for each significant orientation
            for angle in angles:
                oriented_keypoints.append([x, y, angle])
        else:
            oriented_keypoints.append([x, y, 0])
    
    return np.array(oriented_keypoints)

def extract_patches(image, keypoints, patch_size):
    """
    Extract oriented patches around keypoints.
    
    Args:
        image: Input image
        keypoints: Keypoints with orientation [x, y, angle]
        patch_size: Size of the patch
    
    Returns:
        patches: Extracted patches
    """
    patches = []
    half_size = patch_size // 2
    
    for kp in keypoints:
        x, y, angle = kp
        
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Extract rotated patch
        patch = cv2.getRectSubPix(image, (patch_size, patch_size), (x, y))
        
        if angle != 0:
            # Rotate patch
            rot_mat = cv2.getRotationMatrix2D((half_size, half_size), angle, 1.0)
            patch = cv2.warpAffine(patch, rot_mat, (patch_size, patch_size))
        
        patches.append(patch)
    
    return patches

def compute_max_index_map(image, eo, patch_size, orientations):
    """
    Compute maximum index map for the entire image.
    
    Args:
        image: Input image
        eo: Filter convolution results from phase congruency
        patch_size: Patch size
        orientations: Number of orientations
    
    Returns:
        mim: Maximum index map for the entire image
    """
    # Initialize MIM with the same size as the image
    mim = np.zeros(image.shape, dtype=np.int32)
    
    # For each pixel in the image
    h, w = image.shape
    for y in range(h):
        for x in range(w):
            # Compute amplitude sum for each orientation
            amplitude_sums = np.zeros(orientations)
            for o in range(orientations):
                for s in range(len(eo[o])):
                    amplitude_sums[o] += np.abs(eo[o][s][y, x])
            
            # Find orientation with maximum amplitude
            max_orientation = np.argmax(amplitude_sums)
            mim[y, x] = max_orientation + 1  # Make it 1-based
    
    return mim

def feature_describe(image, eo, keypoints, patch_size, grid_x, grid_y):
    """
    Generate RIFT2 descriptors.
    
    Args:
        image: Input image
        eo: Filter convolution results
        keypoints: Keypoints with orientation [x, y, angle]
        patch_size: Size of the patch
        grid_x: Number of grid cells in x direction
        grid_y: Number of grid cells in y direction
    
    Returns:
        descriptors: RIFT2 descriptors for keypoints
    """
    # Convert image to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute maximum index map for the entire image
    n_orientations = len(eo)
    mim = compute_max_index_map(image, eo, patch_size, n_orientations)
    
    descriptors = []
    half_size = patch_size // 2
    
    for kp in keypoints:
        x, y, angle = kp
        x, y = int(x), int(y)
        
        # Skip keypoints too close to the boundary
        if (x < half_size or x >= image.shape[1] - half_size or 
            y < half_size or y >= image.shape[0] - half_size):
            continue
        
        # Extract patch from MIM
        patch_mim = np.zeros((patch_size, patch_size), dtype=np.int32)
        for i in range(patch_size):
            for j in range(patch_size):
                # Compute patch coordinates
                patch_y = y - half_size + i
                patch_x = x - half_size + j
                
                if (patch_x >= 0 and patch_x < image.shape[1] and 
                    patch_y >= 0 and patch_y < image.shape[0]):
                    patch_mim[i, j] = mim[patch_y, patch_x]
        
        # Compute histogram of MIM patch
        hist_bins = np.arange(1, n_orientations + 2)  # Bins from 1 to n_orientations+1
        flat_mim = patch_mim.flatten()
        hist, _ = np.histogram(flat_mim, bins=hist_bins)
        
        # Find dominant orientation for rotation invariance
        max_bin_idx = np.argmax(hist)
        dominant_idx = max_bin_idx + 1  # +1 because bins are 1-based
        
        # Get second highest peak
        hist_copy = hist.copy()
        hist_copy[max_bin_idx] = 0
        second_max_bin_idx = np.argmax(hist_copy)
        second_dominant_idx = second_max_bin_idx + 1
        
        # Check if second peak is significant
        create_second_descriptor = (hist[second_max_bin_idx] > 0.8 * hist[max_bin_idx])
        
        # Recode MIM patch based on dominant index
        recoded_mim = np.zeros_like(patch_mim)
        for i in range(patch_size):
            for j in range(patch_size):
                if patch_mim[i, j] >= dominant_idx:
                    recoded_mim[i, j] = patch_mim[i, j] - dominant_idx + 1
                else:
                    recoded_mim[i, j] = patch_mim[i, j] + n_orientations - dominant_idx + 1
        
        # Divide patch into grid cells
        cell_h = patch_size // grid_y
        cell_w = patch_size // grid_x
        
        # Compute histogram for each grid cell
        descriptor = []
        for gi in range(grid_y):
            for gj in range(grid_x):
                # Extract grid cell
                cell = recoded_mim[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w]
                
                # Compute histogram for this cell
                cell_hist, _ = np.histogram(cell, bins=hist_bins)
                
                # Normalize histogram
                if np.sum(cell_hist) > 0:
                    cell_hist = cell_hist / np.sum(cell_hist)
                
                descriptor.extend(cell_hist)
        
        descriptors.append(descriptor)
        
        # Create second descriptor if needed
        if create_second_descriptor:
            # Recode MIM patch based on second dominant index
            recoded_mim2 = np.zeros_like(patch_mim)
            for i in range(patch_size):
                for j in range(patch_size):
                    if patch_mim[i, j] >= second_dominant_idx:
                        recoded_mim2[i, j] = patch_mim[i, j] - second_dominant_idx + 1
                    else:
                        recoded_mim2[i, j] = patch_mim[i, j] + n_orientations - second_dominant_idx + 1
            
            # Compute descriptor for second recoding
            descriptor2 = []
            for gi in range(grid_y):
                for gj in range(grid_x):
                    cell = recoded_mim2[gi*cell_h:(gi+1)*cell_h, gj*cell_w:(gj+1)*cell_w]
                    cell_hist, _ = np.histogram(cell, bins=hist_bins)
                    if np.sum(cell_hist) > 0:
                        cell_hist = cell_hist / np.sum(cell_hist)
                    descriptor2.extend(cell_hist)
            
            descriptors.append(descriptor2)
    
    return np.array(descriptors)

def match_features(des1, des2, ratio_threshold=1.0, match_threshold=100):
    """
    Match features using nearest neighbor ratio test.
    
    Args:
        des1: Descriptors from first image
        des2: Descriptors from second image
        ratio_threshold: Ratio test threshold (1.0 means no ratio test)
        match_threshold: Maximum allowed distance for good matches
    
    Returns:
        matches: Indices of matching descriptors
    """
    # Compute all pairwise distances
    distances = np.zeros((des1.shape[0], des2.shape[0]))
    for i in range(des1.shape[0]):
        for j in range(des2.shape[0]):
            distances[i, j] = np.sum((des1[i] - des2[j])**2)
    
    matches = []
    for i in range(distances.shape[0]):
        # Sort distances for current descriptor
        sorted_idx = np.argsort(distances[i])
        best_idx = sorted_idx[0]
        best_dist = distances[i, best_idx]
        
        # Check if match distance is below threshold
        if best_dist > match_threshold:
            continue
            
        # Ratio test
        if ratio_threshold < 1.0:
            second_best_idx = sorted_idx[1]
            second_best_dist = distances[i, second_best_idx]
            
            if best_dist / max(second_best_dist, 1e-10) > ratio_threshold:
                continue
        
        matches.append((i, best_idx))
    
    return np.array(matches)

def ransac_transform(points1, points2, transform_type='similarity', error_threshold=3, max_iterations=1000):
    """
    Use RANSAC to find a robust transformation between point sets.
    
    Args:
        points1: First set of points
        points2: Second set of points
        transform_type: Type of transformation ('similarity', 'affine', 'perspective')
        error_threshold: Maximum error for inliers
        max_iterations: Maximum RANSAC iterations
    
    Returns:
        transformation: Transformation matrix
        inliers: Indices of inlier points
    """
    best_model = None
    best_inliers = []
    n_points = points1.shape[0]
    
    if transform_type == 'similarity':
        min_points = 2
    elif transform_type == 'affine':
        min_points = 3
    else:  # perspective
        min_points = 4
    
    if n_points < min_points:
        return None, []
    
    # Limit iterations
    iterations = min(max_iterations, 10000)
    
    for _ in range(iterations):
        # Select random subset
        indices = np.random.choice(n_points, min_points, replace=False)
        sample1 = points1[indices]
        sample2 = points2[indices]
        
        # Estimate model
        if transform_type == 'similarity':
            model = cv2.estimateAffinePartial2D(sample1, sample2)[0]
            # Add row to make it 3x3
            model = np.vstack((model, [0, 0, 1]))
        elif transform_type == 'affine':
            model = cv2.estimateAffine2D(sample1, sample2)[0]
            # Add row to make it 3x3
            model = np.vstack((model, [0, 0, 1]))
        else:  # perspective
            model = cv2.findHomography(sample1, sample2)[0]
        
        if model is None:
            continue
        
        # Apply transformation to all points
        if transform_type == 'perspective':
            # For homography, need to convert to homogeneous coordinates
            homogeneous_points = np.hstack((points1, np.ones((n_points, 1))))
            transformed_points = np.dot(model, homogeneous_points.T).T
            
            # Convert back from homogeneous
            transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
        else:
            # For affine/similarity, directly apply the top 2x3 of the matrix
            transformed_points = np.dot(points1, model[:2, :2].T) + model[:2, 2]
        
        # Calculate errors
        errors = np.sqrt(np.sum((transformed_points - points2)**2, axis=1))
        
        # Find inliers
        inliers = np.where(errors < error_threshold)[0]
        
        # Update best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = model
    
    # Refine model with all inliers
    if len(best_inliers) >= min_points:
        inlier_points1 = points1[best_inliers]
        inlier_points2 = points2[best_inliers]
        
        if transform_type == 'similarity':
            refined_model = cv2.estimateAffinePartial2D(inlier_points1, inlier_points2)[0]
            refined_model = np.vstack((refined_model, [0, 0, 1]))
        elif transform_type == 'affine':
            refined_model = cv2.estimateAffine2D(inlier_points1, inlier_points2)[0]
            refined_model = np.vstack((refined_model, [0, 0, 1]))
        else:  # perspective
            refined_model = cv2.findHomography(inlier_points1, inlier_points2)[0]
        
        best_model = refined_model
    
    return best_model, best_inliers

def image_fusion(image1, image2, transform):
    """
    Fuse two images based on a transformation.
    
    Args:
        image1: First image (reference)
        image2: Second image (to be transformed)
        transform: Transformation matrix
    
    Returns:
        result: Fused image
    """
    # Convert images to correct format
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8)
    
    # Get dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Create larger canvas (3 times the size)
    canvas_width = 3 * w1
    canvas_height = 3 * h1
    
    # Create translation to center image1
    centering_transform = np.array([
        [1, 0, w1],
        [0, 1, h1],
        [0, 0, 1]
    ])
    
    # Calculate the total transformation for image2
    total_transform = np.dot(centering_transform, transform)
    
    # Warp the images
    warped_image1 = cv2.warpPerspective(image1, centering_transform, (canvas_width, canvas_height))
    warped_image2 = cv2.warpPerspective(image2, total_transform, (canvas_width, canvas_height))
    
    # Create masks for non-zero areas
    if len(warped_image1.shape) == 3:
        mask1 = np.any(warped_image1 > 0, axis=2)
        mask2 = np.any(warped_image2 > 0, axis=2)
    else:
        mask1 = warped_image1 > 0
        mask2 = warped_image2 > 0
    
    # Create the fused image
    fusion = np.zeros_like(warped_image1)
    
    # Process regions
    overlap = np.logical_and(mask1, mask2)
    only_img1 = np.logical_and(mask1, ~mask2)
    only_img2 = np.logical_and(~mask1, mask2)
    
    # Copy regions
    if len(fusion.shape) == 3:
        for c in range(fusion.shape[2]):
            fusion[only_img1, c] = warped_image1[only_img1, c]
            fusion[only_img2, c] = warped_image2[only_img2, c]
            fusion[overlap, c] = (warped_image1[overlap, c] // 2 + warped_image2[overlap, c] // 2)
    else:
        fusion[only_img1] = warped_image1[only_img1]
        fusion[only_img2] = warped_image2[only_img2]
        fusion[overlap] = (warped_image1[overlap] // 2 + warped_image2[overlap] // 2)
    
    # Crop to remove unnecessary parts
    # Find bounding box of non-zero region
    if len(fusion.shape) == 3:
        nonzero_rows = np.any(np.any(fusion > 0, axis=2), axis=1)
        nonzero_cols = np.any(np.any(fusion > 0, axis=2), axis=0)
    else:
        nonzero_rows = np.any(fusion > 0, axis=1)
        nonzero_cols = np.any(fusion > 0, axis=0)
    
    if np.sum(nonzero_rows) > 0 and np.sum(nonzero_cols) > 0:
        row_min, row_max = np.where(nonzero_rows)[0][[0, -1]]
        col_min, col_max = np.where(nonzero_cols)[0][[0, -1]]
        
        # Ensure we don't crop too much
        row_min = min(row_min, h1)
        col_min = min(col_min, w1)
        row_max = max(row_max, 2*h1)
        col_max = max(col_max, 2*w1)
        
        # Crop
        fusion = fusion[row_min:row_max+1, col_min:col_max+1]
    
    return fusion

def create_checkerboard_visualization(image1, image2, transform, grid_size):
    """
    Create a checkerboard visualization of two aligned images.
    
    Args:
        image1: First image
        image2: Second image
        transform: Transformation matrix
        grid_size: Size of checkerboard grid cells
    
    Returns:
        result: Checkerboard visualization
    """
    # Convert images to correct format
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8)
    
    # Get dimensions
    h1, w1 = image1.shape[:2]
    
    # Create larger canvas (3 times the size)
    canvas_width = 3 * w1
    canvas_height = 3 * h1
    
    # Create translation to center image1
    centering_transform = np.array([
        [1, 0, w1],
        [0, 1, h1],
        [0, 0, 1]
    ])
    
    # Calculate the total transformation for image2
    total_transform = np.dot(centering_transform, transform)
    
    # Warp the images
    warped_image1 = cv2.warpPerspective(image1, centering_transform, (canvas_width, canvas_height))
    warped_image2 = cv2.warpPerspective(image2, total_transform, (canvas_width, canvas_height))
    
    # Create a checkerboard pattern mask
    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    for i in range(0, canvas_height, grid_size):
        for j in range(0, canvas_width, grid_size):
            if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                mask[i:i+grid_size, j:j+grid_size] = 1
    
    # Create masks for non-zero areas
    if len(warped_image1.shape) == 3:
        valid1 = np.any(warped_image1 > 0, axis=2)
        valid2 = np.any(warped_image2 > 0, axis=2)
    else:
        valid1 = warped_image1 > 0
        valid2 = warped_image2 > 0
    
    # Create combined valid mask
    valid = np.logical_and(valid1, valid2)
    
    # Create checkerboard
    checkerboard = np.zeros_like(warped_image1)
    if len(checkerboard.shape) == 3:
        for c in range(checkerboard.shape[2]):
            img1_regions = np.logical_and(valid, mask == 1)
            img2_regions = np.logical_and(valid, mask == 0)
            checkerboard[img1_regions, c] = warped_image1[img1_regions, c]
            checkerboard[img2_regions, c] = warped_image2[img2_regions, c]
    else:
        img1_regions = np.logical_and(valid, mask == 1)
        img2_regions = np.logical_and(valid, mask == 0)
        checkerboard[img1_regions] = warped_image1[img1_regions]
        checkerboard[img2_regions] = warped_image2[img2_regions]
    
    # Crop to overlap region
    if np.any(valid):
        nonzero_rows = np.any(valid, axis=1)
        nonzero_cols = np.any(valid, axis=0)
        row_indices = np.where(nonzero_rows)[0]
        col_indices = np.where(nonzero_cols)[0]
        min_row, max_row = np.min(row_indices), np.max(row_indices)
        min_col, max_col = np.min(col_indices), np.max(col_indices)
        checkerboard = checkerboard[min_row:max_row+1, min_col:max_col+1]
    
    return checkerboard

def visualize_matches(image1, image2, points1, points2):
    """
    Visualize matching points between two images.
    
    Args:
        image1: First image
        image2: Second image
        points1: Points in first image
        points2: Points in second image
    
    Returns:
        result: Image showing matches
    """
    # Convert images to correct format
    if image1.dtype != np.uint8:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype != np.uint8:
        image2 = (image2 * 255).astype(np.uint8)
    
    # Create a blank image to draw matches
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Ensure images have the same height
    max_height = max(h1, h2)
    image1_resized = np.zeros((max_height, w1, 3), dtype=np.uint8)
    image2_resized = np.zeros((max_height, w2, 3), dtype=np.uint8)
    
    # Convert to 3 channels if needed
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    image1_resized[:h1] = image1
    image2_resized[:h2] = image2
    
    # Create combined image
    combined = np.hstack((image1_resized, image2_resized))
    
    # Draw lines
    for pt1, pt2 in zip(points1, points2):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Adjust x2 to account for image1's width
        x2 += w1
        
        # Draw line
        cv2.line(combined, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # Draw points
        cv2.circle(combined, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(combined, (x2, y2), 3, (0, 255, 0), -1)
    
    return combined

def rift2_match(image1, image2, num_points=5000, patch_size=96):
    """
    Complete RIFT2 matching pipeline.
    
    Args:
        image1: First image
        image2: Second image
        num_points: Maximum number of keypoints
        patch_size: Size of patch for description
    
    Returns:
        matched_points1: Matching points in first image
        matched_points2: Matching points in second image
        transform: Transformation matrix
        visualization: Visualization of matches
    """
    start_time = time.time()
    
    # Feature detection
    print("RIFT2 feature detection")
    kpts1, m1, eo1 = feature_detection(image1, 4, 6, num_points)
    kpts2, m2, eo2 = feature_detection(image2, 4, 6, num_points)
    
    # Orientation calculation
    print("RIFT2 main orientation calculation")
    oriented_kpts1 = kpts_orientation(kpts1, m1, True, patch_size)
    oriented_kpts2 = kpts_orientation(kpts2, m2, True, patch_size)
    
    # Feature description
    print("RIFT2 feature description")
    des1 = feature_describe(image1, eo1, oriented_kpts1, patch_size, 6, 6)
    des2 = feature_describe(image2, eo2, oriented_kpts2, patch_size, 6, 6)
    
    # Feature matching
    print("RIFT2 feature matching")
    matches = match_features(des1, des2, 1.0, 100)
    
    if len(matches) == 0:
        print("No matches found")
        return None, None, None, None
    
    # Get matched points
    matched_pts1 = np.array([oriented_kpts1[idx][:2] for idx, _ in matches])
    matched_pts2 = np.array([oriented_kpts2[idx][:2] for _, idx in matches])
    
    # Remove duplicate points
    _, unique_idx1 = np.unique(matched_pts1, axis=0, return_index=True)
    matched_pts1 = matched_pts1[unique_idx1]
    matched_pts2 = matched_pts2[unique_idx1]
    
    _, unique_idx2 = np.unique(matched_pts2, axis=0, return_index=True)
    matched_pts1 = matched_pts1[unique_idx2]
    matched_pts2 = matched_pts2[unique_idx2]
    
    # Outlier removal using RANSAC
    print("Outlier removal")
    transform, inliers = ransac_transform(matched_pts1, matched_pts2, 'similarity', 3)
    
    if transform is None or len(inliers) == 0:
        print("Couldn't find transformation")
        return matched_pts1, matched_pts2, None, None
    
    # Get inlier points
    inlier_pts1 = matched_pts1[inliers]
    inlier_pts2 = matched_pts2[inliers]
    
    # Visualize matches
    print("Visualizing matches")
    vis = visualize_matches(image1, image2, inlier_pts1, inlier_pts2)
    
    end_time = time.time()
    print(f"RIFT2 total execution time: {end_time - start_time:.2f} seconds")
    
    return inlier_pts1, inlier_pts2, transform, vis

def demo_rift2(image1_path, image2_path):
    """
    Demo function to run RIFT2 on a pair of images.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
    """
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    if image1 is None or image2 is None:
        print(f"Error loading images: {image1_path}, {image2_path}")
        return
    
    # Convert BGR to RGB for display
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Run RIFT2 matching
    matched_pts1, matched_pts2, transform, vis = rift2_match(image1_rgb, image2_rgb)
    
    if matched_pts1 is None:
        print("Matching failed")
        return
    
    print(f"Number of matches: {len(matched_pts1)}")
    
    # Display visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"RIFT2 Matches: {len(matched_pts1)} points")
    plt.axis('off')
    plt.show()
    
    # Fuse images if transform was found
    if transform is not None:
        print("Registration result")
        fusion = image_fusion(image1_rgb, image2_rgb, transform)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(fusion)
        plt.title("Fused Image")
        plt.axis('off')
        plt.show()

        # Create checkerboard visualization
        grid_size = min(fusion.shape[0], fusion.shape[1]) // 5
        checkerboard = create_checkerboard_visualization(image1_rgb, image2_rgb, transform, grid_size)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(checkerboard)
        plt.title("Checkerboard Visualization")
        plt.axis('off')
        plt.show()


# Demo
if __name__ == '__main__':
    demo_rift2('pair1.jpg', 'pair2.jpg')