import cv2
import numpy as np
from .phasecong3 import phasecong   # As per your assumption

def FeatureDetection(im: np.ndarray,
                     nscale: int,
                     norient: int,
                     npt: int):
    """
    Python version of FeatureDetection.m

    Returns:
       kpts  - Nx2 array of keypoint locations
       m     - the 'm' matrix (maximum moment or a derived measure)
       eo    - the entire set of log-Gabor responses
    """

    # Convert to grayscale if needed
    if len(im.shape) == 3 and im.shape[2] == 3:
        # convert to single channel
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im.copy()

    # Here we call the user-provided phase congruency function
    # The original returns (M, m, or, featType, PC, EO, T, pcSum).
    # We only want M (for corners) and EO
    # In the Matlab code, the variable named "m" is actually the "M" from phasecong3 
    # (the "maximum moment"). The code in the .m is a bit confusingly named. 
    # So we just read them in that same order:
    M, _, _, _, _, eo, _ = phasecong(gray, 
                                        nscale=nscale,
                                        norient=norient,
                                        minWaveLength=3, 
                                        mult=1.6,
                                        sigmaOnf=0.75, 
                                        g=3, 
                                        k=1)
    # Rescale M to [0,1] as in MATLAB
    M_min = np.min(M)
    M_max = np.max(M)
    if M_max != M_min:
        M_norm = (M - M_min) / (M_max - M_min)
    else:
        M_norm = M * 0

    # Now detect FAST keypoints on M_norm
    # OpenCV FAST works on 8-bit images typically, so we convert M_norm to 8-bit:
    M_8u = np.uint8(np.round(M_norm * 255.0))

    fast = cv2.FastFeatureDetector_create(threshold=1, 
                                          nonmaxSuppression=True) 
    # If you want exact replication of "MinContrast=0.0001, MinQuality=0.0001" 
    # from MATLAB's detectFASTFeatures, you might need to tune. 
    # The code below just gives an example usage of OpenCV’s FAST.
    
    keypts = fast.detect(M_8u, None)

    # Sort them by response (similar to 'selectStrongest(npt)')
    keypts_sorted = sorted(keypts, key=lambda kp: kp.response, reverse=True)
    keypts_sorted = keypts_sorted[:npt]

    # Convert to Nx2 array
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypts_sorted], dtype=np.float64).T

    return kpts, M_norm, eo
