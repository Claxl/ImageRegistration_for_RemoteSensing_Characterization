import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import random
from .sar_sift import Keypoint


# Constants ported from match.h
DIS_RATIO = 0.9  # Threshold ratio for nearest to second nearest distance
RANSAC_ERROR = 1.5  # RANSAC error threshold
FSC_RATIO_LOW = 0.8  # Low ratio for FSC algorithm
FSC_RATIO_UP = 1.0  # High ratio for FSC algorithm


class DistanceCriterion(Enum):
    """Enum for distance criteria."""
    EUCLIDEAN = 0
    COS = 1


class DMatch:
    """Class representing a descriptor match."""
    def __init__(self, queryIdx: int, trainIdx: int, distance: float):
        self.queryIdx = queryIdx  # Index of query descriptor
        self.trainIdx = trainIdx  # Index of train descriptor
        self.distance = distance  # Distance between descriptors


def match_descriptors(des_1: np.ndarray, des_2: np.ndarray, 
                      dist_criterion: DistanceCriterion = DistanceCriterion.EUCLIDEAN) -> List[List[DMatch]]:
    """
    Match descriptors between two images.
    
    Args:
        des_1: Descriptors from first image (reference)
        des_2: Descriptors from second image (target)
        dist_criterion: Distance criterion to use (EUCLIDEAN or COS)
        
    Returns:
        List of matches, where each match contains the best and second best matches
    """
    num_des_1 = des_1.shape[0]
    num_des_2 = des_2.shape[0]
    dims_des = des_1.shape[1]
    matches = []
    
    if dist_criterion == DistanceCriterion.EUCLIDEAN:
        # For each descriptor in the first image
        for i in range(num_des_1):
            # Find the two nearest matches
            min_dist1 = float('inf')
            min_dist2 = float('inf')
            min_idx1 = -1
            min_idx2 = -1
            
            for j in range(num_des_2):
                # Calculate Euclidean distance
                diff = des_1[i] - des_2[j]
                curr_dist = np.sqrt(np.sum(diff * diff))
                
                if curr_dist < min_dist1:
                    min_dist2 = min_dist1
                    min_idx2 = min_idx1
                    min_dist1 = curr_dist
                    min_idx1 = j
                elif curr_dist < min_dist2:
                    min_dist2 = curr_dist
                    min_idx2 = j
            
            # Create match objects
            match = [
                DMatch(i, min_idx1, min_dist1),
                DMatch(i, min_idx2, min_dist2)
            ]
            matches.append(match)
            
    elif dist_criterion == DistanceCriterion.COS:
        # Calculate dot products for all pairs (optimized matrix multiplication)
        # Normalize the descriptors for cosine similarity
        des_1_norm = des_1 / np.sqrt(np.sum(des_1 * des_1, axis=1))[:, np.newaxis]
        des_2_norm = des_2 / np.sqrt(np.sum(des_2 * des_2, axis=1))[:, np.newaxis]
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity_matrix = np.dot(des_1_norm, des_2_norm.T)
        
        for i in range(num_des_1):
            # Get similarities for current descriptor
            sim_values = similarity_matrix[i]
            
            # Find top two indices (argsort and take last two)
            indices = np.argsort(sim_values)
            min_idx1, min_idx2 = indices[-1], indices[-2]
            
            # Convert to distances (angle in radians)
            min_dist1 = np.arccos(np.clip(sim_values[min_idx1], -1.0, 1.0))
            min_dist2 = np.arccos(np.clip(sim_values[min_idx2], -1.0, 1.0))
            
            # Create match objects
            match = [
                DMatch(i, min_idx1, min_dist1),
                DMatch(i, min_idx2, min_dist2)
            ]
            matches.append(match)
    
    return matches


def compute_lms(points_1: np.ndarray, points_2: np.ndarray, 
                model: str = "affine") -> Tuple[np.ndarray, float]:
    """
    Compute transformation matrix using least mean squares.
    
    Args:
        points_1: Points from first image (N, 2)
        points_2: Points from second image (N, 2)
        model: Transformation model ("similarity", "affine", or "perspective")
        
    Returns:
        Tuple of (transformation matrix, RMSE)
    """
    if points_1.shape[0] != points_2.shape[0]:
        raise ValueError("Number of points must be equal")
    
    if model not in ["similarity", "affine", "perspective"]:
        raise ValueError("Model must be 'similarity', 'affine', or 'perspective'")
    
    N = points_1.shape[0]
    
    # Create homogeneous matrix
    change = np.zeros((3, 3), dtype=np.float32)
    
    if model == "affine":
        # Build matrix A: [x y 0 0 1 0; 0 0 x y 0 1; ...]
        A = np.zeros((2 * N, 6), dtype=np.float32)
        for i in range(N):
            A[2 * i, 0] = points_2[i, 0]  # x
            A[2 * i, 1] = points_2[i, 1]  # y
            A[2 * i, 4] = 1.0
            
            A[2 * i + 1, 2] = points_2[i, 0]
            A[2 * i + 1, 3] = points_2[i, 1]
            A[2 * i + 1, 5] = 1.0
        
        # Build matrix B: [u1 v1 u2 v2 ... un vn]
        B = np.zeros((2 * N, 1), dtype=np.float32)
        for i in range(N):
            B[2 * i, 0] = points_1[i, 0]  # x
            B[2 * i + 1, 0] = points_1[i, 1]  # y
            
        # Solve system of equations
        x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        
        # Construct transformation matrix
        change = np.array([
            [x[0, 0], x[1, 0], x[4, 0]],
            [x[2, 0], x[3, 0], x[5, 0]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Calculate RMSE
        temp_1 = change[:2, :2]
        temp_2 = change[:2, 2:3]
        
        # Apply transformation
        match2_xy_trans = points_2.T
        match1_xy_trans = points_1.T
        
        match2_xy_change = np.dot(temp_1, match2_xy_trans) + np.tile(temp_2, (1, N))
        diff = match2_xy_change - match1_xy_trans
        rmse = np.sqrt(np.sum(diff * diff) / N)
        
    elif model == "perspective":
        # Build matrix for perspective transformation
        A1 = np.zeros((2 * N, 8), dtype=np.float32)
        
        # First build the affine part (same as above)
        for i in range(N):
            A1[2 * i, 0] = points_2[i, 0]  # x
            A1[2 * i, 1] = points_2[i, 1]  # y
            A1[2 * i, 4] = 1.0
            
            A1[2 * i + 1, 2] = points_2[i, 0]
            A1[2 * i + 1, 3] = points_2[i, 1]
            A1[2 * i + 1, 5] = 1.0
        
        # Add perspective part
        for i in range(N):
            A1[2 * i, 6] = -points_1[i, 0] * points_2[i, 0]
            A1[2 * i, 7] = -points_1[i, 0] * points_2[i, 1]
            
            A1[2 * i + 1, 6] = -points_1[i, 1] * points_2[i, 0]
            A1[2 * i + 1, 7] = -points_1[i, 1] * points_2[i, 1]
        
        # Build matrix B (same as above)
        B = np.zeros((2 * N, 1), dtype=np.float32)
        for i in range(N):
            B[2 * i, 0] = points_1[i, 0]  # x
            B[2 * i + 1, 0] = points_1[i, 1]  # y
        
        # Solve system of equations
        x, residuals, rank, s = np.linalg.lstsq(A1, B, rcond=None)
        
        # Construct transformation matrix
        change = np.array([
            [x[0, 0], x[1, 0], x[4, 0]],
            [x[2, 0], x[3, 0], x[5, 0]],
            [x[6, 0], x[7, 0], 1.0]
        ], dtype=np.float32)
        
        # Apply transformation to calculate RMSE
        match2_xy_trans = np.vstack((points_2.T, np.ones((1, N))))
        match1_xy_trans = points_1.T
        
        # Apply perspective transformation
        match2_xy_change = np.dot(change, match2_xy_trans)
        
        # Normalize coordinates
        match2_xy_change_12 = match2_xy_change[:2, :]
        for i in range(N):
            match2_xy_change_12[:, i] /= match2_xy_change[2, i]
        
        diff = match2_xy_change_12 - match1_xy_trans
        rmse = np.sqrt(np.sum(diff * diff) / N)
        
    elif model == "similarity":
        # Build matrix for similarity transformation: [x y 1 0; y -x 0 1] * [a b c d]' = [u v]'
        A3 = np.zeros((2 * N, 4), dtype=np.float32)
        
        for i in range(N):
            A3[2 * i, 0] = points_2[i, 0]  # x
            A3[2 * i, 1] = points_2[i, 1]  # y
            A3[2 * i, 2] = 1.0
            A3[2 * i, 3] = 0.0
            
            A3[2 * i + 1, 0] = points_2[i, 1]  # y
            A3[2 * i + 1, 1] = -points_2[i, 0]  # -x
            A3[2 * i + 1, 2] = 0.0
            A3[2 * i + 1, 3] = 1.0
        
        # Build matrix B (same as above)
        B = np.zeros((2 * N, 1), dtype=np.float32)
        for i in range(N):
            B[2 * i, 0] = points_1[i, 0]  # x
            B[2 * i + 1, 0] = points_1[i, 1]  # y
        
        # Solve system of equations
        x, residuals, rank, s = np.linalg.lstsq(A3, B, rcond=None)
        
        # Construct transformation matrix
        change = np.array([
            [x[0, 0], x[1, 0], x[2, 0]],
            [-x[1, 0], x[0, 0], x[3, 0]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Calculate RMSE
        temp_1 = change[:2, :2]
        temp_2 = change[:2, 2:3]
        
        # Apply transformation
        match2_xy_trans = points_2.T
        match1_xy_trans = points_1.T
        
        match2_xy_change = np.dot(temp_1, match2_xy_trans) + np.tile(temp_2, (1, N))
        diff = match2_xy_change - match1_xy_trans
        rmse = np.sqrt(np.sum(diff * diff) / N)
    
    return change, rmse


def ransac(points_1: np.ndarray, points_2: np.ndarray, 
           model: str = "affine", threshold: float = RANSAC_ERROR, 
           max_iterations: int = 800) -> Tuple[np.ndarray, List[bool], float]:
    """
    RANSAC algorithm for robust estimation of transformation.
    
    Args:
        points_1: Points from first image (N, 2)
        points_2: Points from second image (N, 2)
        model: Transformation model
        threshold: Error threshold for inlier classification
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (transformation matrix, inliers mask, RMSE)
    """
    if points_1.shape[0] != points_2.shape[0]:
        raise ValueError("Number of points must be equal")
    
    if model not in ["similarity", "affine", "perspective"]:
        raise ValueError("Model must be 'similarity', 'affine', or 'perspective'")
    
    N = points_1.shape[0]
    
    # Determine model parameters
    if model == "similarity":
        n = 2  # Minimum number of points needed
    elif model == "affine":
        n = 3
    elif model == "perspective":
        n = 4
    
    # Calculate maximum number of iterations
    if model == "similarity":
        max_iterations_possible = N * (N - 1) // 2
    else:
        max_iterations_possible = N * (N - 1) * (N - 2) // (2 * 3)
    
    iterations = min(max_iterations, max_iterations_possible)
    
    # Convert points to homogeneous coordinates
    arr_1 = np.zeros((3, N), dtype=np.float32)
    arr_2 = np.zeros((3, N), dtype=np.float32)
    
    arr_1[0, :] = points_1[:, 0]  # x
    arr_1[1, :] = points_1[:, 1]  # y
    arr_1[2, :] = 1.0
    
    arr_2[0, :] = points_2[:, 0]  # x
    arr_2[1, :] = points_2[:, 1]  # y
    arr_2[2, :] = 1.0
    
    most_consensus_num = 0  # Most inliers found
    inliers = [False] * N
    best_transformation = None
    
    for _ in range(iterations):
        # Randomly select n different points
        while True:
            indices = random.sample(range(N), n)
            
            # Check if points are not coincident
            valid = True
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if (points_1[indices[i], 0] == points_1[indices[j], 0] and 
                        points_1[indices[i], 1] == points_1[indices[j], 1]):
                        valid = False
                        break
                    if (points_2[indices[i], 0] == points_2[indices[j], 0] and 
                        points_2[indices[i], 1] == points_2[indices[j], 1]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
        
        # Extract selected points
        sub_arr1 = np.zeros((n, 2), dtype=np.float32)
        sub_arr2 = np.zeros((n, 2), dtype=np.float32)
        
        for i in range(n):
            sub_arr1[i, 0] = points_1[indices[i], 0]
            sub_arr1[i, 1] = points_1[indices[i], 1]
            
            sub_arr2[i, 0] = points_2[indices[i], 0]
            sub_arr2[i, 1] = points_2[indices[i], 1]
        
        # Compute transformation matrix
        transformation, _ = compute_lms(sub_arr1, sub_arr2, model)
        
        # Count inliers
        consensus_num = 0
        right = [False] * N
        
        if model == "perspective":
            # Apply perspective transformation
            match2_xy_change = np.dot(transformation, arr_2)
            match2_xy_change_12 = match2_xy_change[:2, :]
            
            # Normalize coordinates (with protection against division by zero)
            for i in range(N):
                # Add small epsilon to avoid division by zero
                denominator = match2_xy_change[2, i]
                if abs(denominator) < 1e-10:
                    denominator = 1e-10 if denominator >= 0 else -1e-10
                match2_xy_change_12[:, i] /= denominator
            
            # Calculate squared distances
            diff = match2_xy_change_12 - arr_1[:2, :]
            dists = np.sum(diff * diff, axis=0)
            
            # Identify inliers
            for i in range(N):
                if dists[i] < threshold:
                    right[i] = True
                    consensus_num += 1
                
        else:  # affine or similarity
            # Apply transformation
            match2_xy_change = np.dot(transformation, arr_2)
            diff = match2_xy_change - arr_1
            
            # Calculate squared distances
            dists = np.sum(diff[:2, :] * diff[:2, :], axis=0)
            
            # Identify inliers
            for i in range(N):
                if dists[i] < threshold:
                    right[i] = True
                    consensus_num += 1
        
        # Update best model if we found more inliers
        if consensus_num > most_consensus_num:
            most_consensus_num = consensus_num
            inliers = right.copy()
            best_transformation = transformation
    
    # Remove duplicate matches
    for i in range(N - 1):
        for j in range(i + 1, N):
            if inliers[i] and inliers[j]:
                if (points_1[i, 0] == points_1[j, 0] and 
                    points_1[i, 1] == points_1[j, 1] and 
                    points_2[i, 0] == points_2[j, 0] and 
                    points_2[i, 1] == points_2[j, 1]):
                    inliers[j] = False
                    most_consensus_num -= 1
    
    # Extract consensus set
    consensus_arr1 = np.zeros((most_consensus_num, 2), dtype=np.float32)
    consensus_arr2 = np.zeros((most_consensus_num, 2), dtype=np.float32)
    
    k = 0
    for i in range(N):
        if inliers[i]:
            consensus_arr1[k, 0] = points_1[i, 0]
            consensus_arr1[k, 1] = points_1[i, 1]
            
            consensus_arr2[k, 0] = points_2[i, 0]
            consensus_arr2[k, 1] = points_2[i, 1]
            k += 1
    
    # Recompute transformation with all inliers
    final_transformation, rmse = compute_lms(consensus_arr1, consensus_arr2, model)
    
    return final_transformation, inliers, rmse


def fsc(points1_low: np.ndarray, points2_low: np.ndarray,
        points1_up: np.ndarray, points2_up: np.ndarray,
        model: str = "affine", threshold: float = RANSAC_ERROR,
        max_iterations: int = 800) -> Tuple[np.ndarray, List[bool], float]:
    """
    Feature Scale Consensus algorithm for robust estimation of transformation.
    This algorithm is an improvement over RANSAC.
    
    Args:
        points1_low: Points from first image with low distance ratio
        points2_low: Points from second image with low distance ratio
        points1_up: Points from first image with high distance ratio
        points2_up: Points from second image with high distance ratio
        model: Transformation model
        threshold: Error threshold for inlier classification
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (transformation matrix, inliers mask, RMSE)
    """
    if points1_low.shape[0] != points2_low.shape[0] or points1_up.shape[0] != points2_up.shape[0]:
        raise ValueError("Number of points in each pair must be equal")
    
    if model not in ["similarity", "affine", "perspective"]:
        raise ValueError("Model must be 'similarity', 'affine', or 'perspective'")
    
    N = points1_low.shape[0]  # Number of low distance ratio points
    M = points1_up.shape[0]  # Number of high distance ratio points (M > N)
    
    # Determine model parameters
    if model == "similarity":
        n = 2  # Minimum number of points needed
    elif model == "affine":
        n = 3
    elif model == "perspective":
        n = 4
    
    # Calculate maximum number of iterations
    if model == "similarity":
        max_iterations_possible = N * (N - 1) // 2
    else:
        max_iterations_possible = N * (N - 1) * (N - 2) // (2 * 3)
    
    iterations = min(max_iterations, max_iterations_possible)
    
    # Convert low ratio points to homogeneous coordinates
    arr1_low = np.zeros((3, N), dtype=np.float32)
    arr2_low = np.zeros((3, N), dtype=np.float32)
    
    arr1_low[0, :] = points1_low[:, 0]  # x
    arr1_low[1, :] = points1_low[:, 1]  # y
    arr1_low[2, :] = 1.0
    
    arr2_low[0, :] = points2_low[:, 0]  # x
    arr2_low[1, :] = points2_low[:, 1]  # y
    arr2_low[2, :] = 1.0
    
    # Convert high ratio points to homogeneous coordinates
    arr1_up = np.zeros((3, M), dtype=np.float32)
    arr2_up = np.zeros((3, M), dtype=np.float32)
    
    arr1_up[0, :] = points1_up[:, 0]  # x
    arr1_up[1, :] = points1_up[:, 1]  # y
    arr1_up[2, :] = 1.0
    
    arr2_up[0, :] = points2_up[:, 0]  # x
    arr2_up[1, :] = points2_up[:, 1]  # y
    arr2_up[2, :] = 1.0
    
    most_consensus_num = 0  # Most inliers found
    inliers = [False] * M
    best_transformation = None
    
    for _ in range(iterations):
        # Randomly select n different points from low ratio points
        while True:
            indices = random.sample(range(N), n)
            
            # Check if points are not coincident
            valid = True
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if (points1_low[indices[i], 0] == points1_low[indices[j], 0] and 
                        points1_low[indices[i], 1] == points1_low[indices[j], 1]):
                        valid = False
                        break
                    if (points2_low[indices[i], 0] == points2_low[indices[j], 0] and 
                        points2_low[indices[i], 1] == points2_low[indices[j], 1]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
        
        # Extract selected points
        sub_arr1 = np.zeros((n, 2), dtype=np.float32)
        sub_arr2 = np.zeros((n, 2), dtype=np.float32)
        
        for i in range(n):
            sub_arr1[i, 0] = points1_low[indices[i], 0]
            sub_arr1[i, 1] = points1_low[indices[i], 1]
            
            sub_arr2[i, 0] = points2_low[indices[i], 0]
            sub_arr2[i, 1] = points2_low[indices[i], 1]
        
        # Compute transformation matrix
        transformation, _ = compute_lms(sub_arr1, sub_arr2, model)
        
        # Count inliers among high ratio points
        consensus_num = 0
        right = [False] * M
        
        if model == "perspective":
            # Apply perspective transformation
            match2_xy_change = np.dot(transformation, arr2_up)
            match2_xy_change_12 = match2_xy_change[:2, :]
            
            # Normalize coordinates (with protection against division by zero)
            for i in range(M):
                # Add small epsilon to avoid division by zero
                denominator = match2_xy_change[2, i]
                if abs(denominator) < 1e-10:
                    denominator = 1e-10 if denominator >= 0 else -1e-10
                match2_xy_change_12[:, i] /= denominator
            
            # Calculate squared distances
            diff = match2_xy_change_12 - arr1_up[:2, :]
            dists = np.sum(diff * diff, axis=0)
            
            # Identify inliers
            for i in range(M):
                if dists[i] < threshold:
                    right[i] = True
                    consensus_num += 1
                
        else:  # affine or similarity
            # Apply transformation
            match2_xy_change = np.dot(transformation, arr2_up)
            diff = match2_xy_change - arr1_up
            
            # Calculate squared distances
            dists = np.sum(diff[:2, :] * diff[:2, :], axis=0)
            
            # Identify inliers
            for i in range(M):
                if dists[i] < threshold:
                    right[i] = True
                    consensus_num += 1
        
        # Update best model if we found more inliers
        if consensus_num > most_consensus_num:
            most_consensus_num = consensus_num
            inliers = right.copy()
            best_transformation = transformation
    
    # Remove duplicate matches
    for i in range(M - 1):
        for j in range(i + 1, M):
            if inliers[i] and inliers[j]:
                if ((points1_up[i, 0] == points1_up[j, 0] and 
                     points1_up[i, 1] == points1_up[j, 1] and 
                     points2_up[i, 0] == points2_up[j, 0] and 
                     points2_up[i, 1] == points2_up[j, 1]) or
                    (points1_up[i, 0] == points1_up[j, 0] and 
                     points1_up[i, 1] == points1_up[j, 1]) or
                    (points2_up[i, 0] == points2_up[j, 0] and 
                     points2_up[i, 1] == points2_up[j, 1])):
                    inliers[j] = False
                    most_consensus_num -= 1
    
    # Extract consensus set
    consensus_arr1 = np.zeros((most_consensus_num, 2), dtype=np.float32)
    consensus_arr2 = np.zeros((most_consensus_num, 2), dtype=np.float32)
    
    k = 0
    for i in range(M):
        if inliers[i]:
            consensus_arr1[k, 0] = points1_up[i, 0]
            consensus_arr1[k, 1] = points1_up[i, 1]
            
            consensus_arr2[k, 0] = points2_up[i, 0]
            consensus_arr2[k, 1] = points2_up[i, 1]
            k += 1
    
    # Recompute transformation with all inliers
    final_transformation, rmse = compute_lms(consensus_arr1, consensus_arr2, model)
    
    return final_transformation, inliers, rmse


def mosaic_map(image_1: np.ndarray, image_2: np.ndarray, 
               width: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a chessboard visualization of two images.
    
    Args:
        image_1: First image
        image_2: Second image
        width: Size of chessboard squares
        
    Returns:
        Tuple of (chessboard_1, chessboard_2, mosaic_image)
    """
    if image_1.shape != image_2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    # Create chessboard pattern for image_1
    chessboard_1 = image_1.copy()
    rows_1, cols_1 = chessboard_1.shape[:2]
    
    row_grids_1 = rows_1 // width
    col_grids_1 = cols_1 // width
    
    for i in range(0, row_grids_1, 2):
        for j in range(1, col_grids_1, 2):
            chessboard_1[i*width:(i+1)*width, j*width:(j+1)*width] = 0
    
    for i in range(1, row_grids_1, 2):
        for j in range(0, col_grids_1, 2):
            chessboard_1[i*width:(i+1)*width, j*width:(j+1)*width] = 0
    
    # Create chessboard pattern for image_2
    chessboard_2 = image_2.copy()
    rows_2, cols_2 = chessboard_2.shape[:2]
    
    row_grids_2 = rows_2 // width
    col_grids_2 = cols_2 // width
    
    for i in range(0, row_grids_2, 2):
        for j in range(0, col_grids_2, 2):
            chessboard_2[i*width:(i+1)*width, j*width:(j+1)*width] = 0
    
    for i in range(1, row_grids_2, 2):
        for j in range(1, col_grids_2, 2):
            chessboard_2[i*width:(i+1)*width, j*width:(j+1)*width] = 0
    
    # Combine the chessboards
    mosaic_image = chessboard_1 + chessboard_2
    
    return chessboard_1, chessboard_2, mosaic_image


def image_fusion(image_1: np.ndarray, image_2: np.ndarray, 
                 T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse two images using the given transformation.
    
    Args:
        image_1: First image (reference)
        image_2: Second image (to be transformed)
        T: Transformation matrix
        
    Returns:
        Tuple of (fusion_image, mosaic_image)
    """
    if len(image_1.shape) != len(image_2.shape):
        # Convert to same number of channels
        if len(image_1.shape) == 3 and len(image_2.shape) == 2:
            image_2_temp = cv2.cvtColor(image_2, cv2.COLOR_GRAY2BGR)
            image_1_temp = image_1
        elif len(image_1.shape) == 2 and len(image_2.shape) == 3:
            image_1_temp = cv2.cvtColor(image_1, cv2.COLOR_GRAY2BGR)
            image_2_temp = image_2
        else:
            image_1_temp = image_1
            image_2_temp = image_2
    else:
        image_1_temp = image_1
        image_2_temp = image_2
    
    rows_1, cols_1 = image_1_temp.shape[:2]
    
    # Create extended space for the transformed images
    T_temp = np.array([
        [1, 0, cols_1],
        [0, 1, rows_1],
        [0, 0, 1]
    ], dtype=np.float32)
    T_1 = np.dot(T_temp, T)
    
    # Warp images
    trans_1 = np.zeros((3 * rows_1, 3 * cols_1, image_1_temp.shape[2] if len(image_1_temp.shape) > 2 else 1), 
                      dtype=image_1_temp.dtype)
    trans_1[rows_1:2*rows_1, cols_1:2*cols_1] = image_1_temp.reshape(rows_1, cols_1, -1)
    
    if len(image_1_temp.shape) > 2:
        trans_2 = cv2.warpPerspective(image_2_temp, T_1, (3*cols_1, 3*rows_1))
    else:
        trans_2 = cv2.warpPerspective(image_2_temp, T_1, (3*cols_1, 3*rows_1))
    
    # Create fusion by blending overlapping regions
    trans = trans_2.copy()
    
    # For overlapping regions, take the average
    mask = (trans_1 > 0) & (trans_2 > 0)
    if len(mask.shape) > 2:
        mask = np.any(mask, axis=2)
        mask = np.stack([mask] * trans_1.shape[2], axis=2)
    
    trans[mask] = (trans_1[mask].astype(np.float32) + trans_2[mask].astype(np.float32)) / 2
    
    # For non-overlapping regions of image_1, use image_1
    mask_1 = (trans_1 > 0) & (trans_2 == 0)
    if len(mask_1.shape) > 2:
        mask_1 = np.any(mask_1, axis=2)
        mask_1 = np.stack([mask_1] * trans_1.shape[2], axis=2)
    
    trans[mask_1] = trans_1[mask_1]
    
    # Calculate region bounds for final image
    corners = np.array([
        [0, 0, 1],
        [0, image_2_temp.shape[0] - 1, 1],
        [image_2_temp.shape[1] - 1, 0, 1],
        [image_2_temp.shape[1] - 1, image_2_temp.shape[0] - 1, 1]
    ])
    
    corners_transformed = np.dot(T_1, corners.T).T
    
    # Normalize homogeneous coordinates
    corners_transformed[:, 0] /= corners_transformed[:, 2]
    corners_transformed[:, 1] /= corners_transformed[:, 2]
    
    # Calculate bounds
    min_x = max(0, int(np.floor(min(corners_transformed[:, 0]))))
    max_x = min(3 * cols_1 - 1, int(np.ceil(max(corners_transformed[:, 0]))))
    min_y = max(0, int(np.floor(min(corners_transformed[:, 1]))))
    max_y = min(3 * rows_1 - 1, int(np.ceil(max(corners_transformed[:, 1]))))
    
    # Ensure we include the original image
    min_x = min(min_x, cols_1)
    max_x = max(max_x, 2 * cols_1 - 1)
    min_y = min(min_y, rows_1)
    max_y = max(max_y, 2 * rows_1 - 1)
    
    # Crop final fused image
    fusion_image = trans[min_y:max_y+1, min_x:max_x+1]
    
    # Create mosaic visualization
    chessboard_1, chessboard_2, mosaic_image = mosaic_map(
        trans_1[min_y:max_y+1, min_x:max_x+1],
        trans_2[min_y:max_y+1, min_x:max_x+1],
        50
    )
    
    return fusion_image, mosaic_image


def match(image_1: np.ndarray, image_2: np.ndarray, 
          matches: List[List[DMatch]], keypoints_1: List[Keypoint], 
          keypoints_2: List[Keypoint], model: str = "affine") -> Tuple[np.ndarray, List[DMatch], np.ndarray]:
    """
    Match features between two images and compute transformation.
    
    Args:
        image_1: First image (reference)
        image_2: Second image (target)
        matches: Initial matches between keypoints
        keypoints_1: Keypoints from first image
        keypoints_2: Keypoints from second image
        model: Transformation model
        
    Returns:
        Tuple of (homography matrix, correct matches, visualization image)
    """
    # Extract initial matching point pairs
    points1_low = []
    points2_low = []
    points1_up = []
    points2_up = []
    low_matches = []
    up_matches = []
    
    for i, match_pair in enumerate(matches):
        dist_1 = match_pair[0].distance
        dist_2 = match_pair[1].distance
        
        # Low distance ratio matches
        if (dist_1 / dist_2) < FSC_RATIO_LOW:
            point1 = keypoints_1[match_pair[0].queryIdx].pt
            point2 = keypoints_2[match_pair[0].trainIdx].pt
            points1_low.append(point1)
            points2_low.append(point2)
            low_matches.append(match_pair[0])
        
        # High distance ratio matches (includes both points)
        if (dist_1 / dist_2) <= FSC_RATIO_UP:
            point1 = keypoints_1[match_pair[0].queryIdx].pt
            point2 = keypoints_2[match_pair[0].trainIdx].pt
            points1_up.append(point1)
            points2_up.append(point2)
            up_matches.append(match_pair[0])
            
            point1 = keypoints_1[match_pair[1].queryIdx].pt
            point2 = keypoints_2[match_pair[1].trainIdx].pt
            points1_up.append(point1)
            points2_up.append(point2)
            up_matches.append(match_pair[1])
    
    print(f"FSC low ratio match count: {len(low_matches)}")
    print(f"FSC high ratio match count: {len(up_matches)}")
    
    # Convert to numpy arrays
    points1_low_np = np.array(points1_low, dtype=np.float32)
    points2_low_np = np.array(points2_low, dtype=np.float32)
    points1_up_np = np.array(points1_up, dtype=np.float32)
    points2_up_np = np.array(points2_up, dtype=np.float32)
    
    # Apply FSC to remove outliers
    homography, inliers, rmse = fsc(
        points1_low_np, points2_low_np,
        points1_up_np, points2_up_np,
        model, RANSAC_ERROR
    )
    
    # Extract correct matches
    right_matches = []
    for i, is_inlier in enumerate(inliers):
        if is_inlier:
            right_matches.append(up_matches[i])
    
    print(f"FSC found {len(right_matches)} correct matches")
    print(f"RMSE: {rmse}")
    
    # Convert our custom DMatch objects to OpenCV DMatch objects
    cv_matches = [cv2.DMatch(_m.queryIdx, _m.trainIdx, _m.distance) for _m in right_matches]
    
    # Create OpenCV keypoints 
    cv_keypoints_1 = [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) 
                    for kp in keypoints_1]
    cv_keypoints_2 = [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) 
                    for kp in keypoints_2]
    
    # Draw matches visualization with compatibility for different OpenCV versions
    try:
        # For newer OpenCV versions that might require matchesThickness
        matched_line = cv2.drawMatches(
            image_1, cv_keypoints_1,
            image_2, cv_keypoints_2,
            cv_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchesThickness=2
        )
    except TypeError:
        # For older OpenCV versions
        matched_line = cv2.drawMatches(
            image_1, cv_keypoints_1,
            image_2, cv_keypoints_2,
            cv_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
    
    return homography, right_matches, matched_line