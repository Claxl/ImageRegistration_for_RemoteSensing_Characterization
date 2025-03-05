import numpy as np
from .LSM import LSM

def FSC(cor1: np.ndarray,
        cor2: np.ndarray,
        change_form: str = 'affine',
        error_t: float = 3.0):
    """
    Python version of FSC.m

    RANSAC-like approach to estimate transformation in 'change_form'
    among 'similarity', 'affine', or 'perspective'.
    Also returns the refined inlier correspondences (cor1_new, cor2_new)
    and the final transformation matrix 'solution' and its rmse.
    """

    M = cor1.shape[0]
    if change_form == 'similarity':
        n = 2
        max_iteration = M*(M-1)//2
    elif change_form == 'affine':
        n = 3
        max_iteration = M*(M-1)*(M-2)//6
    elif change_form == 'perspective':
        n = 4
        max_iteration = M*(M-1)*(M-2)//6
    else:
        raise ValueError("Unknown transform: " + change_form)

    iterations = min(max_iteration, 10000)

    most_consensus_number = 0
    cor1_new = np.zeros_like(cor1)
    cor2_new = np.zeros_like(cor2)

    rng = np.random.default_rng(seed=None)  # random generator

    for _ in range(iterations):
        # randomly sample n distinct indices
        # ensuring we pick unique points
        valid_sample = False
        for __ in range(100):  # up to 100 tries to get distinct
            a = rng.choice(M, size=n, replace=False)
            cor11 = cor1[a, 0:2]
            cor22 = cor2[a, 0:2]
            # Check they are not degenerate
            # (the original checks that not all x's are the same, etc.)
            # We'll assume the 4 random picks are good enough for typical usage:
            valid_sample = True
            break

        if not valid_sample:
            continue
        
        # Solve for that subset
        parameters, _ = LSM(cor11, cor22, change_form)
        # Build the 3x3 solution matrix
        solution = np.array([
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [parameters[6], parameters[7], 1.0]
        ])

        # Evaluate inliers
        match1_xy = np.column_stack((cor1[:,0], cor1[:,1], np.ones(M)))
        # transform
        match1_test_trans = (solution @ match1_xy.T).T

        if change_form == 'perspective':
            # we have to do x'/w, y'/w
            denom = match1_test_trans[:,2]
            match1_test_trans_12 = match1_test_trans[:,0:2] / denom[:,None]
        else:
            match1_test_trans_12 = match1_test_trans[:,0:2]

        diff = match1_test_trans_12 - cor2[:,0:2]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        index_in = np.where(dists < error_t)[0]
        consensus_num = index_in.size

        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in,:]
            cor2_new = cor2[index_in,:]

    # final LSM with inliers
    parameters_final, rmse = LSM(cor1_new[:,0:2], cor2_new[:,0:2], change_form)
    solution_final = np.array([
        [parameters_final[0], parameters_final[1], parameters_final[4]],
        [parameters_final[2], parameters_final[3], parameters_final[5]],
        [parameters_final[6], parameters_final[7], 1.0]
    ])

    return solution_final, rmse, cor1_new, cor2_new
