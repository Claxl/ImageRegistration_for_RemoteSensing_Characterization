import numpy as np

def LSM(match1: np.ndarray,
        match2: np.ndarray,
        change_form: str = 'affine'):
    """
    Python version of LSM.m

    Solve for transformation parameters based on two sets of match points (x,y).
    Return 'parameters' and the RMSE.
    
    The final 'parameters' is always 8 long, with [a1,a2,a3,a4,a5,a6,a7,a8].
    For perspective, we fill all 8. For similarity or affine, some become 0.
    """

    # match1, match2 are Mx2
    M = match1.shape[0]

    # We'll unify the logic. The MATLAB code does a variety of expansions
    # We'll follow that approach.

    if change_form == 'affine':
        # Build big A
        # For each row i, we place:
        # [ x_i, y_i, 0,   0,   1,  0 ]
        # [ 0,   0,   x_i, y_i, 0,  1 ]
        # Then solve A*[a1,a2,a3,a4,a5,a6]^T = b
        # Then we pad with a7=a8=0
        # We do a simple approach in Python
        A_rows = []
        b_rows = []
        for i in range(M):
            x_i, y_i = match1[i]
            u_i, v_i = match2[i]
            A_rows.append([x_i, y_i, 0.0, 0.0, 1.0, 0.0])
            b_rows.append(u_i)
            A_rows.append([0.0, 0.0, x_i, y_i, 0.0, 1.0])
            b_rows.append(v_i)

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)
        # Solve via least-squares
        params, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
        # We pad with a7=0, a8=0
        parameters = np.zeros(8, dtype=np.float64)
        parameters[0:6] = params[0:6]

        # compute rmse
        # transform match1, measure distance to match2
        # The matrix is 2x2 plus translation
        # M = [a1 a2; a3 a4], t=(a5, a6)
        a1,a2,a3,a4,a5,a6,a7,a8 = parameters
        M_mat = np.array([[a1,a2],[a3,a4]])
        t_vec = np.array([a5,a6])
        match1_trans = (M_mat @ match1.T).T + t_vec
        diff = match1_trans - match2
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    elif change_form == 'perspective':
        # The code expands for each row i:
        # [ x_i, y_i, 0,   0,   1,  0,  -u_i*x_i, -u_i*y_i ] => to match u_i
        # [ 0,   0,   x_i, y_i, 0,  1,  -v_i*x_i, -v_i*y_i ] => to match v_i
        # Then solve for [a1..a8].
        A_rows = []
        b_rows = []
        for i in range(M):
            x_i, y_i = match1[i]
            u_i, v_i = match2[i]
            A_rows.append([ x_i,  y_i, 0.0,  0.0,  1.0, 0.0, -u_i*x_i, -u_i*y_i ])
            b_rows.append(u_i)
            A_rows.append([ 0.0,  0.0, x_i,  y_i, 0.0, 1.0, -v_i*x_i, -v_i*y_i ])
            b_rows.append(v_i)
        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)

        params, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
        parameters = params  # length 8

        # compute rmse
        # build 3x3 matrix
        M_mat = np.array([
            [parameters[0], parameters[1], parameters[4]],
            [parameters[2], parameters[3], parameters[5]],
            [parameters[6], parameters[7],           1.0],
        ])
        ones_col = np.ones((M,1))
        match1_h = np.hstack((match1, ones_col))
        proj = (M_mat @ match1_h.T).T
        xy = proj[:,0:2] / proj[:,2:3]
        diff = xy - match2
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    elif change_form == 'similarity':
        # We expect similarity transform: scale+rotation+translation
        # The original code builds specialized A. Let’s do it similarly
        # Each point => 
        # [ x_i, y_i, 1, 0 ] -> u_i
        # [ y_i,-x_i, 0, 1 ] -> v_i
        A_rows = []
        b_rows = []
        for i in range(M):
            x_i, y_i = match1[i]
            u_i, v_i = match2[i]
            A_rows.append([ x_i,  y_i, 1.0, 0.0 ])
            b_rows.append(u_i)
            A_rows.append([ y_i, -x_i, 0.0, 1.0 ])
            b_rows.append(v_i)
        A_ = np.array(A_rows, dtype=np.float64)
        b_ = np.array(b_rows, dtype=np.float64)

        # Solve
        param4, residuals, rank, svals = np.linalg.lstsq(A_, b_, rcond=None)
        # param4 = [a, b, c, d], we then map to [a1,a2,a3,a4,a5,a6,a7,a8]
        # In the MATLAB code, a1=param4[0], a2=param4[1], 
        # a3=-param4[1], a4=param4[0], a5=param4[2], a6=param4[3], a7=a8=0
        parameters = np.zeros(8, dtype=np.float64)
        a = param4[0]
        b = param4[1]
        c = param4[2]
        d = param4[3]
        parameters[0] = a
        parameters[1] = b
        parameters[2] = -b
        parameters[3] = a
        parameters[4] = c
        parameters[5] = d
        parameters[6] = 0
        parameters[7] = 0

        # compute rmse
        M_mat = np.array([[a, b],[ -b, a]])
        t_vec = np.array([c,d])
        match1_trans = (M_mat @ match1.T).T + t_vec
        diff = match1_trans - match2
        rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    else:
        raise ValueError("Unknown transform type in LSM: " + change_form)

    return parameters, rmse
