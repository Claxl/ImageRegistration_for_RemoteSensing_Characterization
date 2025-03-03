import numpy as np
from scipy import linalg

def LSM(cor1, cor2, change_form):
    """
    Least Squares Method for fitting transformation models.
    
    Args:
        cor1: First set of points (Nx2)
        cor2: Second set of points (Nx2)
        change_form: Transformation model ('similarity', 'affine', or 'perspective')
        
    Returns:
        parameters: Transformation parameters
        rmse: Root mean square error
    """
    M = cor1.shape[0]  # Number of point pairs
    
    if change_form == 'similarity':
        # Similarity transformation: [s*cos(θ), -s*sin(θ), tx; s*sin(θ), s*cos(θ), ty]
        A = np.zeros((2 * M, 4))
        b = np.zeros(2 * M)
        
        for i in range(M):
            # For x coordinate
            A[2*i, 0] = cor1[i, 0]     # x scale/rotation component
            A[2*i, 1] = -cor1[i, 1]    # -y scale/rotation component
            A[2*i, 2] = 1              # tx
            A[2*i, 3] = 0
            b[2*i] = cor2[i, 0]        # Target x
            
            # For y coordinate
            A[2*i+1, 0] = cor1[i, 1]   # y scale/rotation component
            A[2*i+1, 1] = cor1[i, 0]   # x scale/rotation component
            A[2*i+1, 2] = 0
            A[2*i+1, 3] = 1            # ty
            b[2*i+1] = cor2[i, 1]      # Target y
        
        # Solve linear system
        x, residuals, rank, s = linalg.lstsq(A, b, rcond=None)
        
        # Construct parameters array
        parameters = np.zeros(8)
        parameters[0] = x[0]   # a
        parameters[1] = x[1]   # b
        parameters[2] = -x[1]  # -b (for orthogonal rotation)
        parameters[3] = x[0]   # a
        parameters[4] = 0      # perspective parameter 1
        parameters[5] = 0      # perspective parameter 2
        parameters[6] = x[2]   # tx
        parameters[7] = x[3]   # ty
        
    elif change_form == 'affine':
        # Affine transformation: [a, b, tx; c, d, ty]
        A = np.zeros((2 * M, 6))
        b = np.zeros(2 * M)
        
        for i in range(M):
            # For x coordinate
            A[2*i, 0] = cor1[i, 0]   # a: x scale/shear
            A[2*i, 1] = cor1[i, 1]   # b: y scale/shear for x
            A[2*i, 2] = 1            # tx
            A[2*i, 3] = 0
            A[2*i, 4] = 0
            A[2*i, 5] = 0
            b[2*i] = cor2[i, 0]      # Target x
            
            # For y coordinate
            A[2*i+1, 0] = 0
            A[2*i+1, 1] = 0
            A[2*i+1, 2] = 0
            A[2*i+1, 3] = cor1[i, 0]  # c: x scale/shear for y
            A[2*i+1, 4] = cor1[i, 1]  # d: y scale/shear
            A[2*i+1, 5] = 1           # ty
            b[2*i+1] = cor2[i, 1]     # Target y
        
        # Solve linear system
        x, residuals, rank, s = linalg.lstsq(A, b, rcond=None)
        
        # Construct parameters array
        parameters = np.zeros(8)
        parameters[0] = x[0]   # a
        parameters[1] = x[1]   # b
        parameters[2] = x[3]   # c
        parameters[3] = x[4]   # d
        parameters[4] = 0      # perspective parameter 1
        parameters[5] = 0      # perspective parameter 2
        parameters[6] = x[2]   # tx
        parameters[7] = x[5]   # ty
        
    elif change_form == 'perspective':
        # Perspective transformation
        A = np.zeros((2 * M, 8))
        b = np.zeros(2 * M)
        
        for i in range(M):
            # For x coordinate
            A[2*i, 0] = cor1[i, 0]                # a: x scale/shear
            A[2*i, 1] = cor1[i, 1]                # b: y scale/shear for x
            A[2*i, 2] = 1                         # tx
            A[2*i, 3] = 0
            A[2*i, 4] = 0
            A[2*i, 5] = 0
            A[2*i, 6] = -cor1[i, 0] * cor2[i, 0]  # perspective term
            A[2*i, 7] = -cor1[i, 1] * cor2[i, 0]  # perspective term
            b[2*i] = cor2[i, 0]                   # Target x
            
            # For y coordinate
            A[2*i+1, 0] = 0
            A[2*i+1, 1] = 0
            A[2*i+1, 2] = 0
            A[2*i+1, 3] = cor1[i, 0]                # c: x scale/shear for y
            A[2*i+1, 4] = cor1[i, 1]                # d: y scale/shear
            A[2*i+1, 5] = 1                         # ty
            A[2*i+1, 6] = -cor1[i, 0] * cor2[i, 1]  # perspective term
            A[2*i+1, 7] = -cor1[i, 1] * cor2[i, 1]  # perspective term
            b[2*i+1] = cor2[i, 1]                   # Target y
        
        # Solve linear system
        x, residuals, rank, s = linalg.lstsq(A, b, rcond=None)
        
        # For perspective, parameters are directly from solution
        parameters = x
    
    # Create transformation matrix
    solution = np.array([
        [parameters[0], parameters[1], parameters[6]],
        [parameters[2], parameters[3], parameters[7]],
        [parameters[4], parameters[5], 1]
    ])
    
    # Calculate RMSE
    if change_form == 'perspective':
        # For perspective transformation
        match1_xy = np.hstack([cor1, np.ones((M, 1))])
        match1_test_trans = match1_xy @ solution.T
        
        # Apply perspective division
        match1_test_trans[:, 0] /= match1_test_trans[:, 2]
        match1_test_trans[:, 1] /= match1_test_trans[:, 2]
        match1_test_trans = match1_test_trans[:, :2]
        
        # Calculate error
        diff = match1_test_trans - cor2
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    else:
        # For similarity and affine transformations
        match1_xy = np.hstack([cor1, np.ones((M, 1))])
        match1_test_trans = match1_xy @ solution.T
        match1_test_trans = match1_test_trans[:, :2]
        
        # Calculate error
        diff = match1_test_trans - cor2
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    return parameters, rmse

def FSC(cor1, cor2, change_form, error_t):
    """
    Fast Sample Consensus for robust model estimation.
    
    Args:
        cor1: First set of points (Nx2)
        cor2: Second set of points (Nx2)
        change_form: Transformation model ('similarity', 'affine', or 'perspective')
        error_t: Error threshold for inliers
        
    Returns:
        solution: Transformation matrix
        rmse: Root mean square error
        cor1_new: Inlier points from cor1
        cor2_new: Inlier points from cor2
    """
    M, N = cor1.shape
    
    # Determine number of minimum points based on transformation model
    if change_form == 'similarity':
        n = 2
        max_iteration = M * (M - 1) // 2
    elif change_form == 'affine':
        n = 3
        max_iteration = M * (M - 1) * (M - 2) // (2 * 3)
    elif change_form == 'perspective':
        n = 4
        max_iteration = M * (M - 1) * (M - 2) // (2 * 3)
    
    # Limit iterations for large datasets
    if max_iteration > 10000:
        iterations = 10000
    else:
        iterations = max_iteration
    
    # Initialize variables
    most_consensus_number = 0
    cor1_new = np.zeros((0, N))
    cor2_new = np.zeros((0, N))
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Main FSC loop
    for i in range(iterations):
        # Randomly select n points
        while True:
            a = np.random.choice(M, n, replace=False)
            cor11 = cor1[a, :2]
            cor22 = cor2[a, :2]
            
            # Check if points are valid based on transformation model
            if change_form == 'similarity':
                if n == 2 and np.any(cor11[0] != cor11[1]) and np.any(cor22[0] != cor22[1]):
                    break
            elif change_form == 'affine':
                if n == 3 and np.any(cor11[0] != cor11[1]) and np.any(cor11[0] != cor11[2]) and np.any(cor11[1] != cor11[2]) and \
                   np.any(cor22[0] != cor22[1]) and np.any(cor22[0] != cor22[2]) and np.any(cor22[1] != cor22[2]):
                    break
            elif change_form == 'perspective':
                if n == 4 and all(np.any(cor11[i] != cor11[j]) for i in range(n) for j in range(i+1, n)) and \
                   all(np.any(cor22[i] != cor22[j]) for i in range(n) for j in range(i+1, n)):
                    break
        
        # Estimate model from random sample
        parameters, _ = LSM(cor11, cor22, change_form)
        
        # Create transformation matrix
        solution = np.array([
            [parameters[0], parameters[1], parameters[6]],
            [parameters[2], parameters[3], parameters[7]],
            [parameters[4], parameters[5], 1]
        ])
        
        # Find inliers
        if change_form == 'perspective':
            # For perspective transformation
            match1_xy = np.hstack([cor1, np.ones((M, 1))])
            match1_test_trans = match1_xy @ solution.T
            
            # Apply perspective division
            match1_test_trans[:, 0] /= match1_test_trans[:, 2]
            match1_test_trans[:, 1] /= match1_test_trans[:, 2]
            match1_test_trans = match1_test_trans[:, :2]
            
            # Calculate distances to corresponding points
            diff = match1_test_trans - cor2[:, :2]
            diff_match2_xy = np.sqrt(np.sum(diff**2, axis=1))
            
            # Find inliers
            index_in = np.where(diff_match2_xy < error_t)[0]
            consensus_num = len(index_in)
        else:
            # For similarity and affine transformations
            match1_xy = np.hstack([cor1, np.ones((M, 1))])
            t_match1_xy = match1_xy @ solution.T
            match2_xy = np.hstack([cor2, np.ones((M, 1))])
            
            # Calculate differences
            diff_match2_xy = t_match1_xy - match2_xy
            diff_match2_xy = np.sqrt(np.sum(diff_match2_xy**2, axis=1))
            
            # Find inliers
            index_in = np.where(diff_match2_xy < error_t)[0]
            consensus_num = len(index_in)
        
        # Update best model if we found more inliers
        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in]
            cor2_new = cor2[index_in]
    
    # If no inliers found, return empty results
    if most_consensus_number == 0:
        return np.eye(3), 0, np.zeros((0, N)), np.zeros((0, N))
    
    # Remove duplicate points
    # First, find unique points in first set
    _, idx = np.unique(cor1_new[:, :2], axis=0, return_index=True)
    cor1_new = cor1_new[np.sort(idx)]
    cor2_new = cor2_new[np.sort(idx)]
    
    # Then, find unique points in second set
    _, idx = np.unique(cor2_new[:, :2], axis=0, return_index=True)
    cor1_new = cor1_new[np.sort(idx)]
    cor2_new = cor2_new[np.sort(idx)]
    
    # Recompute model with all inliers
    parameters, rmse = LSM(cor1_new[:, :2], cor2_new[:, :2], change_form)
    
    # Create final transformation matrix
    solution = np.array([
        [parameters[0], parameters[1], parameters[6]],
        [parameters[2], parameters[3], parameters[7]],
        [parameters[4], parameters[5], 1]
    ])
    
    return solution, rmse, cor1_new, cor2_new