import numpy as np
import cv2  # for resizing; or use from skimage.transform import resize
from .extract_patches import extract_patches  # Make sure this matches your extract_patches translation

def FeatureDescribe(im, eo, kpts, patch_size, no, nbin):
    """
    Python translation of the MATLAB function FeatureDescribe.m

    Args:
        im          : Input image (H x W x 3 or H x W).
        eo          : A 2D list/tuple of log-Gabor responses, eo[i][j] for scale i, orientation j.
                      'eo' should match the structure you get from phase congruency in your code.
                      This code sums over the first 4 scales for each orientation.
        kpts        : 3 x N array of keypoints.  Each column = [x, y, angle].
        patch_size  : Size of the patch (r in the MATLAB code).
        no          : Number of orientations used to build the maximum index map (often 6).
        nbin        : Number of bins for the per-subregion histogram (often 6).

    Returns:
        des : (no * no * nbin) x N array of feature descriptors
    """

    # Number of keypoints
    n = kpts.shape[1]

    # Build the convolution sequence "CS" of shape (H, W, no).
    # Summation of amplitude across the first 4 scales for each orientation.
    yim, xim = im.shape[:2]
    CS = np.zeros((yim, xim, no), dtype=np.float32)

    for oo in range(no):        # orientation index
        some_sum = np.zeros_like(eo[0][0], dtype=np.float32)  # shape (H,W)
        for ss in range(4):     # scale index
            some_sum += np.abs(eo[oo][ss])

        # Store final sum for this orientation in CS[:,:,oo]
        CS[:, :, oo] = some_sum


    # MIM = argmax along the 3rd dimension => each pixel gets an orientation index
    # (in MATLAB, ~ is the max, MIM is the argmax).
    MIM = np.argmax(CS, axis=2) + 1  # +1 because MATLAB uses 1-based indices

    # We'll output descriptors in shape (no*no*nbin, n).
    des = np.zeros((no * no * nbin, n), dtype=np.float32)

    # Loop over keypoints
    for k in range(n):
        x = kpts[0, k]
        y = kpts[1, k]
        ang = kpts[2, k]
        r = patch_size

        # 1) Extract patch from MIM using your extract_patches function.
        #    The MATLAB code does: extract_patches(MIM, x, y, round(r/2), ang)
        half_r = int(round(r/2))
        patch = extract_patches(MIM, x, y, half_r, ang)

        # 2) Resize patch to (r+1, r+1).  In OpenCV, use (width, height).
        #    If you want strictly float64, adjust as needed:
        patch_resized = cv2.resize(patch.astype(np.float32),
                                   (r+1, r+1),
                                   interpolation=cv2.INTER_LINEAR)

        # 3) Build histogram of the raw patch_resized from 1..6 => 'h' in MATLAB
        #    "hist(patch(:), 1:6)" => we can do:
        #    h, _ = np.histogram( patch_resized.flatten(), bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5] )
        #    That yields 6 counts in h[0..5].
        h, _ = np.histogram(patch_resized,
                            bins=[0.5,1.5,2.5,3.5,4.5,5.5,6.5])

        # 4) [~, idx] = max(h).  In Python => idx = np.argmax(h)
        idx = np.argmax(h)  # 0..5 => means bin #0..5 => corresponds to orientation bin idx+1 in MATLAB

        # 5) patch_rot = patch - idx+1  => in MATLAB, patch - idx + 1
        #    But remember 'idx' is zero-based, so to replicate the
        #    1-based logic, we do patch_resized - (idx+1) + 1 => patch_resized - idx
        patch_rot = patch_resized - idx

        # 6) patch_rot(patch_rot<0) += no
        patch_rot[patch_rot < 0] += no

        # 7) Subdivide patch_rot into no x no blocks.  For each block, compute
        #    hist(clip(:), 1:no).  Then store into histo(j,i,:).
        ys, xs = patch_rot.shape
        histo = np.zeros((no, no, nbin), dtype=np.float32)

        # note: nbin presumably equals 'no' in your usage. But we'll keep it separate.
        block_h = ys / no
        block_w = xs / no

        for jj in range(no):
            for ii in range(no):
                # integer sub-block boundaries
                r0 = int(round(jj * block_h))
                r1 = int(round((jj+1) * block_h))
                c0 = int(round(ii * block_w))
                c1 = int(round((ii+1) * block_w))

                # clip sub-block
                clip = patch_rot[r0:r1, c0:c1].flatten()

                # hist(clip(:), 1:no) => bins = [0.5,1.5,..., no+0.5]
                # we store the 0..(no-1) counts in histo(jj,ii,:)
                hh, _ = np.histogram(clip,
                                     bins=np.arange(0.5, no+1.0, 1.0))
                # hh will have length 'no'.  If no != nbin, you'll need adjustments.
                # The original code used: hist(clip(:), 1:no).
                # If you truly want nbin = no, then this is correct.

                # If nbin != no, you might do something else.
                # For the code snippet, we assume nbin == no => 6 in typical usage.
                # We'll just store min(len(hh), nbin).
                len_ok = min(len(hh), nbin)
                histo[jj, ii, :len_ok] = hh[:len_ok]

        # flatten histo => shape is no*no*nbin
        histo_flat = histo.flatten()

        # 8) Normalize (L2 norm) => if norm(...)~=0 => histo=histo/norm(histo)
        norm_val = np.linalg.norm(histo_flat)
        if norm_val > 1e-12:
            histo_flat = histo_flat / norm_val

        # 9) Store in descriptor => des(:,k)
        des[:, k] = histo_flat

    return des
