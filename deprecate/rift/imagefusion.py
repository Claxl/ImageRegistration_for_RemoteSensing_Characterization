import numpy as np
import cv2

def mosaic_map(img1, img2, d):
    """
    Create a checkerboard pattern for visualization of registration results.
    
    Args:
        img1: First image
        img2: Second image
        d: Size of checkerboard cells
        
    Returns:
        image1: First image with checkerboard pattern
        image2: Second image with checkerboard pattern
        img3: Combined checkerboard image
    """
    # Get image dimensions
    m1, n1 = img1.shape[:2]
    
    # Determine if images are color or grayscale
    p1 = img1.shape[2] if len(img1.shape) > 2 else 1
    
    # Calculate number of cells
    m11 = int(np.ceil(m1 / d))
    n11 = int(np.ceil(n1 / d))
    
    # Create copy of first image
    img1_copy = img1.copy()
    
    # Apply checkerboard pattern to first image (set alternate cells to zero)
    for i in range(0, m11, 2):
        for j in range(1, n11, 2):
            i_start = i * d
            i_end = min((i + 1) * d, m1)
            j_start = j * d
            j_end = min((j + 1) * d, n1)
            
            if p1 == 3:
                img1_copy[i_start:i_end, j_start:j_end, :] = 0
            else:
                img1_copy[i_start:i_end, j_start:j_end] = 0
    
    for i in range(1, m11, 2):
        for j in range(0, n11, 2):
            i_start = i * d
            i_end = min((i + 1) * d, m1)
            j_start = j * d
            j_end = min((j + 1) * d, n1)
            
            if p1 == 3:
                img1_copy[i_start:i_end, j_start:j_end, :] = 0
            else:
                img1_copy[i_start:i_end, j_start:j_end] = 0
    
    image1 = img1_copy
    
    # Get dimensions of second image
    m2, n2 = img2.shape[:2]
    p2 = img2.shape[2] if len(img2.shape) > 2 else 1
    
    # Calculate number of cells
    m22 = int(np.ceil(m2 / d))
    n22 = int(np.ceil(n2 / d))
    
    # Create copy of second image
    img2_copy = img2.copy()
    
    # Apply complementary checkerboard pattern to second image
    for i in range(0, m22, 2):
        for j in range(0, n22, 2):
            i_start = i * d
            i_end = min((i + 1) * d, m2)
            j_start = j * d
            j_end = min((j + 1) * d, n2)
            
            if p2 == 3:
                img2_copy[i_start:i_end, j_start:j_end, :] = 0
            else:
                img2_copy[i_start:i_end, j_start:j_end] = 0
    
    for i in range(1, m22, 2):
        for j in range(1, n22, 2):
            i_start = i * d
            i_end = min((i + 1) * d, m2)
            j_start = j * d
            j_end = min((j + 1) * d, n2)
            
            if p2 == 3:
                img2_copy[i_start:i_end, j_start:j_end, :] = 0
            else:
                img2_copy[i_start:i_end, j_start:j_end] = 0
    
    image2 = img2_copy
    
    # Create combined image
    img3 = image1 + image2
    
    return image1, image2, img3

def image_fusion(image_1, image_2, solution):
    """
    Fuse two images using a transformation matrix.
    
    Args:
        image_1: First image (target)
        image_2: Second image (source)
        solution: 3x3 transformation matrix
        
    Returns:
        fusion_image: Fused image
        f_3: Fused checkerboard visualization
    """
    # Get image dimensions
    M1, N1 = image_1.shape[:2]
    M2, N2 = image_2.shape[:2]
    
    # Determine color channels
    if len(image_1.shape) == 3 and len(image_2.shape) == 3:
        num1 = image_1.shape[2]
        num2 = image_2.shape[2]
        fusion_image = np.zeros((3 * M1, 3 * N1, num1), dtype=np.uint8)
    elif len(image_1.shape) == 1 and len(image_2.shape) == 3:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    elif len(image_1.shape) == 3 and len(image_2.shape) == 1:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    else:
        fusion_image = np.zeros((3 * M1, 3 * N1), dtype=np.uint8)
    
    # Define transformation matrix for first image
    solution_1 = np.array([
        [1, 0, N1],
        [0, 1, M1],
        [0, 0, 1]
    ])
    
    # Transform first image
    f_1 = cv2.warpPerspective(image_1, solution_1, (3 * N1, 3 * M1))
    
    # Transform second image using composition of transformations
    tform = solution_1 @ solution
    f_2 = cv2.warpPerspective(image_2, tform, (3 * N1, 3 * M1))
    
    # Find overlapping regions
    if len(f_1.shape) == 3:
        mask1 = np.any(f_1 > 0, axis=2)
        mask2 = np.any(f_2 > 0, axis=2)
    else:
        mask1 = f_1 > 0
        mask2 = f_2 > 0
    
    same_index = np.logical_and(mask1, mask2)
    index_1 = np.logical_and(mask1, ~mask2)
    index_2 = np.logical_and(~mask1, mask2)
    
    # Fuse images
    if len(fusion_image.shape) == 3:
        for c in range(fusion_image.shape[2]):
            fusion_image[same_index, c] = f_1[same_index, c] // 2 + f_2[same_index, c] // 2
            fusion_image[index_1, c] = f_1[index_1, c]
            fusion_image[index_2, c] = f_2[index_2, c]
    else:
        fusion_image[same_index] = f_1[same_index] // 2 + f_2[same_index] // 2
        fusion_image[index_1] = f_1[index_1]
        fusion_image[index_2] = f_2[index_2]
    
    # Calculate transformed corners of second image
    corners = np.array([
        [1, 1, 1],
        [1, M2, 1],
        [N2, 1, 1],
        [N2, M2, 1]
    ])
    
    transformed_corners = (solution_1 @ solution @ corners.T).T
    
    # Normalize homogeneous coordinates
    X = transformed_corners[:, 0] / transformed_corners[:, 2]
    Y = transformed_corners[:, 1] / transformed_corners[:, 2]
    
    # Calculate crop boundaries
    X_min = max(int(np.floor(np.min(X))), 1)
    X_max = min(int(np.ceil(np.max(X))), 3 * N1)
    Y_min = max(int(np.floor(np.min(Y))), 1)
    Y_max = min(int(np.ceil(np.max(Y))), 3 * M1)
    
    # Ensure crop region is not too small
    if X_min > N1 + 1:
        X_min = N1 + 1
    if X_max < 2 * N1:
        X_max = 2 * N1
    if Y_min > M1 + 1:
        Y_min = M1 + 1
    if Y_max < 2 * M1:
        Y_max = 2 * M1
    
    # Crop images
    if len(fusion_image.shape) == 3:
        fusion_image = fusion_image[Y_min:Y_max, X_min:X_max, :]
        f_1_cropped = f_1[Y_min:Y_max, X_min:X_max, :]
        f_2_cropped = f_2[Y_min:Y_max, X_min:X_max, :]
    else:
        fusion_image = fusion_image[Y_min:Y_max, X_min:X_max]
        f_1_cropped = f_1[Y_min:Y_max, X_min:X_max]
        f_2_cropped = f_2[Y_min:Y_max, X_min:X_max]
    
    # Create checkerboard visualization
    grid_num = 5
    grid_size = int(np.floor(min(fusion_image.shape[:2]) / grid_num))
    _, _, f_3 = mosaic_map(f_1_cropped, f_2_cropped, grid_size)
    
    return fusion_image, f_3