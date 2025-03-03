import numpy as np

def calculate_orientation_hist(x, y, radius, gradient, angle, n, Sa):
    """
    Python version of calculate_oritation_hist.m
    
    The function uses a local patch of radius and sums up gradient magnitudes,
    weighted by a Gaussian, binned by orientation into n bins.
    Sa is the disk-shaped mask. 
    """

    sigma = radius / 3.0

    h, w = gradient.shape
    # define integer corners
    x_left   = int(x - radius)
    x_right  = int(x + radius)
    y_up     = int(y - radius)
    y_down   = int(y + radius)

    # clip to image boundaries
    x_left   = max(0, x_left);   y_up   = max(0, y_up)
    x_right  = min(w-1, x_right); y_down = min(h-1, y_down)

    sub_gradient = gradient[y_up:y_down+1, x_left:x_right+1]
    sub_angle    = angle[y_up:y_down+1, x_left:x_right+1]

    # We build coordinates for that local patch
    patch_h, patch_w = sub_gradient.shape
    # Typically we'd do something like:
    X = np.arange(-radius, radius+1)
    Y = np.arange(-radius, radius+1)
    XX, YY = np.meshgrid(X, Y)

    # But note that we clipped the patch, so if the patch is truncated, 
    # we align the top-left corner at (y_up, x_left) for indexing, etc.

    # Just compute a Gaussian weighting with the same shape
    # the top-left of subGradient is at (y_up, x_left).
    # The center is still (x, y).
    # So for each sub-pixel we do dist from center
    # We'll do a direct approach:

    # distance from (x,y)
    row_coords = np.arange(y_up, y_up + patch_h)
    col_coords = np.arange(x_left, x_left + patch_w)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing='ij')
    # dist^2
    dist2 = (row_grid - y)**2 + (col_grid - x)**2
    gaussian_weight = np.exp(-dist2 / (2*sigma*sigma))

    # Multiply by the binary disk mask Sa (which might be bigger if we didn't clip).
    # So we only keep the top-left corner from Sa that matches sub_gradient shape
    # In MATLAB, they simply do subGradient .* gaussian_weight .* Sa
    # We'll do the same.
    # We must slice out the top-left portion of Sa if needed
    Sa_h, Sa_w = Sa.shape
    # The center of Sa is at (Sa_h//2, Sa_w//2).
    # We want the region that corresponds to sub_gradient. 
    # For simplicity, let’s do a quick hack: 
    # build an array of same shape as sub_gradient from Sa, with appropriate offsets:

    # row offset in Sa that corresponds to y_up
    center_r = Sa_h//2
    center_c = Sa_w//2
    # The sub-patch in Sa we want is [y_up-y+center_r : y_down-y+center_r, x_left-x+center_c : ...]
    # This can get tricky if the patch is clipped. We can do a small function to slice carefully.

    # A simpler route is to treat Sa as if it’s big enough and do a direct intersection.
    # For clarity, we’ll do a direct “compute inside/outside circle ourselves,”
    # ignoring Sa. If you want to replicate exactly, just slice Sa. 
    # Here we’ll keep consistency with the original code:
    
    # We'll do an integer grid from -r..r:
    # and we have sub-patch of shape (patch_h, patch_w)
    # We'll just slice out the subregion from Sa that corresponds:

    # Indices of subregion in Sa:
    rtop    = max(0, center_r - (y - y_up))
    rbottom = rtop + patch_h
    cleft   = max(0, center_c - (x - x_left))
    cright  = cleft + patch_w

    # This must be int
    rtop    = int(round(rtop))
    rbottom = int(round(rbottom))
    cleft   = int(round(cleft))
    cright  = int(round(cright))

    Sa_sub = np.zeros_like(sub_gradient)
    # Watch for bounds
    if 0 <= rtop < Sa_h and 0 <= rbottom <= Sa_h and \
       0 <= cleft < Sa_w and 0 <= cright <= Sa_w and \
       rbottom-rtop == patch_h and cright-cleft == patch_w:
        Sa_sub = Sa[rtop:rbottom, cleft:cright]
    else:
        # Patch is partially outside the disk region
        # We do an intersection approach
        # We'll fill only those coords which are inside the original disk
        # for each local pixel in sub_gradient, compute offset from center
        # If inside circle => 1, else 0. 
        # This is simpler:
        local_yy = (row_grid - y)
        local_xx = (col_grid - x)
        rr = local_xx**2 + local_yy**2
        Sa_sub = (rr <= radius*radius).astype(np.float64)

    # Weighted gradient
    W1 = sub_gradient * gaussian_weight
    # Then apply the disk mask
    W = W1 * Sa_sub

    # Bin angles
    # The code does: bin = round(sub_angle*n/360)
    # then if bin>=n => bin-n, if bin<0 => bin+ n
    bins = np.round(sub_angle * n / 360.0).astype(int)
    bins[bins >= n] -= n
    bins[bins < 0]  += n

    # Now accumulate
    temp_hist = np.zeros(n, dtype=np.float64)
    for idx_row in range(patch_h):
        for idx_col in range(patch_w):
            b = bins[idx_row, idx_col]
            val = W[idx_row, idx_col]
            temp_hist[b] += val

    # Then do the smoothing of the histogram as in the code
    hist_out = np.zeros_like(temp_hist)
    # The code sets 
    #  hist(1) = (temp_hist(n-1)+temp_hist(3))/16 + ...
    # etc.  We replicate:

    # Because Python is 0-indexed:
    def tH(i):  # a helper for circular indexing
        return temp_hist[i % n]

    hist_out[0] = (tH(n-2) + tH(2))/16 + 4*(tH(n-1) + tH(1))/16 + tH(0)*6/16
    hist_out[1] = (tH(n-1) + tH(3))/16 + 4*(tH(0) + tH(2))/16 + tH(1)*6/16

    for j in range(2, n-2):
        hist_out[j] = (tH(j-2)+tH(j+2))/16 + 4*(tH(j-1)+tH(j+1))/16 + tH(j)*6/16

    hist_out[n-2] = (tH(n-4)+tH(0))/16 + 4*(tH(n-3)+tH(n-1))/16 + tH(n-2)*6/16
    hist_out[n-1] = (tH(n-3)+tH(1))/16 + 4*(tH(n-2)+tH(0))/16 + tH(n-1)*6/16

    max_val = np.max(hist_out)
    return hist_out, max_val
