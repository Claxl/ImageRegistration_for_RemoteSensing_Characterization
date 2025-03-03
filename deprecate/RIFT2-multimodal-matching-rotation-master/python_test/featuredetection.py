import cv2
import numpy as np
# Suppose your phase congruency code is in phasecong.py
from phasecong3 import phasecong  # or wherever you placed the function

def feature_detection(im, s=4, o=6, npt=5000):
    """
    Python version of the FeatureDetection() function:
      - Convert to grayscale if needed
      - Compute phase congruency
      - Normalize the 'm' matrix
      - Use FAST detection on 'm'
      - Select the strongest N keypoints
      - Return (keypoints, m, eo)
    """
    # 1) Convert to grayscale if it's color
    if im.ndim == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im.astype(np.uint8)

    # 2) Compute phase congruency
    # phasecong returns (M, m, ori, ft, PC, EO, T) in this version
    # We only need M (index=0) and EO (index=5) for the final usage
    M, _, _, _, _, EO, _ = phasecong(gray, 
                                     nscale=s,  # 's' in MATLAB
                                     norient=o, # 'o' in MATLAB
                                     minWaveLength=3, 
                                     mult=1.6, 
                                     sigmaOnf=0.75, 
                                     k=1, 
                                     g=3)
    # 3) Normalize M into [0,1]
    M = M.astype(np.float32)
    m_min, m_max = M.min(), M.max()
    if m_max > m_min:
        M = (M - m_min) / (m_max - m_min)
    else:
        M.fill(0)  # degenerate case

    # 4) Use FAST detection on M
    # We'll build a FAST detector with similar thresholds
    fast_detector = cv2.FastFeatureDetector_create()
    # You can tweak the threshold to reflect 'MinContrast' or 'MinQuality'
    fast_detector.setThreshold(10)  # Example only

    # Keypoint detection
    kp = fast_detector.detect((M * 255).astype(np.uint8))

    # 5) Sort by "response" to pick top npt
    kp = sorted(kp, key=lambda x: x.response, reverse=True)
    kp = kp[:npt]

    # 6) Extract keypoint coordinates into (2 x N) array as in MATLAB
    #    (like double(kpts.Location'), but we store in float32
    keypoints_array = np.array([ [k.pt[0], k.pt[1]] for k in kp ], dtype=np.float32).T

    return keypoints_array, M, EO
