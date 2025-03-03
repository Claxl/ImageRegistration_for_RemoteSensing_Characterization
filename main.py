import numpy as np
import cv2
# Suppose we have all the above functions in a package called rift2
from rift2.FeatureDetection import FeatureDetection
from rift2.kptsOrientation import kptsOrientation
from rift2.FeatureDescribe import FeatureDescribe    # Not shown but see note
from rift2.FSC import FSC
from rift2.image_fusion import image_fusion

def demo_RIFT2(path1: str, path2: str):

    # 1) Read images
    im1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if im1 is None or im2 is None:
        print("Error reading input images.")
        return

    # If single channel, replicate to 3-ch for consistency
    if im1.ndim == 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
    if im2.ndim == 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)

    print("Feature detection")
    # 2) Feature detection
    key1, m1, eo1 = FeatureDetection(im1, 4, 6, 5000)
    key2, m2, eo2 = FeatureDetection(im2, 4, 6, 5000)

    print(len(eo1))

    print("Orientation")

    # 3) Orientation
    kpts1 = kptsOrientation(key1, m1, True, 96)
    kpts2 = kptsOrientation(key2, m2, True, 96)

    # 4) Feature description
    #   You have a MATLAB function FeatureDescribe, which is not shown in your snippet.
    #   Typically it collects local patches in the MIM (or the log-Gabor results) 
    #   and builds a descriptor. You would replicate that code similarly.
    #   For demonstration, we'll assume you have a function:
    #   des1 = FeatureDescribe(im1, eo1, kpts1, 96, 6, 6)
    #   des2 = FeatureDescribe(im2, eo2, kpts2, 96, 6, 6)
    #   Each is e.g. NxD.
    print("Feature description")
    des1 = FeatureDescribe(im1, eo1, kpts1, 96, 6, 6)  # not shown here
    des2 = FeatureDescribe(im2, eo2, kpts2, 96, 6, 6)  # not shown here
    des1 = des1.T  # so it's (numKeypoints1, descriptorDimension)
    des2 = des2.T  # so it's (numKeypoints2, descriptorDimension)


    # 5) Match the descriptors
    #   Suppose they’re arrays shape (N1, D) and (N2, D).
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1.astype(np.float32),
                          des2.astype(np.float32),
                          k=2)

    # We can mimic 'matchFeatures' with ratio test etc.
    good = []
    ratio_thresh = 1.0   # or 0.8, etc., as you like
    for m in matches:
        if len(m) == 2:
            if m[0].distance < ratio_thresh * m[1].distance:
                good.append(m[0])
        elif len(m) == 1:
            good.append(m[0])

    matchedPoints1 = []
    matchedPoints2 = []
    for g in good:
        matchedPoints1.append(kpts1[:2, g.queryIdx])  # (x, y)
        matchedPoints2.append(kpts2[:2, g.trainIdx])

    matchedPoints1 = np.array(matchedPoints1)
    matchedPoints2 = np.array(matchedPoints2)

    # 6) Remove duplicates, etc.
    #   For example:
    matchedPoints2_unique, idxs = np.unique(matchedPoints2, axis=0, return_index=True)
    matchedPoints1_unique = matchedPoints1[idxs]

    # 7) RANSAC or FSC
    H, rmse, c1, c2 = FSC(matchedPoints1_unique, matchedPoints2_unique,
                          change_form='similarity',
                          error_t=3.0)
    # 8) Show final matches (just for debugging)
    print(c1)
    # 9) Do fusion
    fused = image_fusion(im1, im2, H)
    cv2.imshow("Fused", fused)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H, fused

if __name__ == '__main__':
    path1 = 'DATASET/OSdataset/512/test/sar/sar1.png'
    path2 = 'DATASET/OSdataset/512/test/opt/opt1.png'
    demo_RIFT2(path1, path2)