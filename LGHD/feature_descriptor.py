import numpy as np
import cv2
from .lghd import lghd
from .phasecong3 import phasecong

class FeatureDescriptor:
    """
    FeatureDescriptor: Manage different feature descriptors
    
    This implementation focuses specifically on the LGHD (Log-Gabor Histogram Descriptor)
    for multimodal image registration.
    
    Example: 
    im = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(im, None)
    points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    fd = FeatureDescriptor('LGHD')
    fd_im = fd.compute(im, points)
    
    fd_im['kps'] contains the keypoints after computing the descriptors
    fd_im['des'] contains the descriptors
    """
    
    def __init__(self, descriptor_name='LGHD'):
        """Constructor. Sets the descriptor"""
        self.supported_descriptors = ['LGHD']
        self.set_descriptor_name(descriptor_name)
    
    def set_descriptor_name(self, descriptor_name):
        """Sets the descriptor"""
        if descriptor_name.upper() in self.supported_descriptors:
            self.descriptor_name = descriptor_name.upper()
        else:
            self.descriptor_name = self.supported_descriptors[0]
            print(f'Descriptor {descriptor_name} is not supported')
            print(f'Using {self.supported_descriptors[0]} instead')
        self.set_parameters({})
    
    def set_parameters(self, parameters):
        """Sets descriptor parameters"""
        if not parameters:
            if self.descriptor_name == 'LGHD':
                self.parameters = self.lghd_default_parameters()
        else:
            for key, value in parameters.items():
                if key in self.parameters:
                    self.parameters[key] = value
                else:
                    print(f'The parameter {key} does not exist')
    
    def get_descriptors_available(self):
        """Get the list of the available descriptors"""
        return self.supported_descriptors
    
    def compute(self, im, kps):
        """Computes descriptors for given image and keypoints"""
        return self.compute_lghd(im, kps.T)
    
    def compute_lghd(self, im, kps):
        """Compute Log-Gabor Histogram Descriptor"""
        # Call phasecong to get the phase congruency data
        # Using parameters that work well with your implementation
        M, m, ori, ft, PC, EO, T = phasecong(im, nscale=4, norient=6, 
                                            minWaveLength=3, mult=2.1, 
                                            sigmaOnf=0.55, k=2.0, 
                                            cutOff=0.5, g=10)
        
        # Initialize arrays
        eh = np.zeros((384, kps.shape[1]))
        kps_to_ignore = np.zeros(kps.shape[1], dtype=bool)
        
        for i in range(kps.shape[1]):
            # Get keypoint location
            x = round(kps[0, i])
            y = round(kps[1, i])
            
            # Check if point is within image bounds
            if x < 0 or y < 0 or x >= im.shape[1] or y >= im.shape[0]:
                kps_to_ignore[i] = True
                continue
            
            # Define patch boundary
            x1 = max(0, x-self.parameters['patch_size']//2)
            y1 = max(0, y-self.parameters['patch_size']//2)
            x2 = min(x+self.parameters['patch_size']//2, im.shape[1])
            y2 = min(y+self.parameters['patch_size']//2, im.shape[0])
            
            # Skip if patch is too small
            if (y2-y1) != self.parameters['patch_size'] or (x2-x1) != self.parameters['patch_size']:
                kps_to_ignore[i] = True
                continue
            
            # Process each scale
            descriptor_parts = []
            for s in range(4):  # 4 scales
                scale_patches = []
                # Extract patches for all orientations at this scale
                for o in range(6):  # 6 orientations
                    if s < len(EO) and o < len(EO[s]):
                        # Extract the filter response for this scale and orientation
                        patch = EO[s][o][y1:y2, x1:x2]
                        scale_patches.append(patch)
                    else:
                        # If data not available, use zeros
                        patch = np.zeros((y2-y1, x2-x1), dtype=complex)
                        scale_patches.append(patch)
                
                # Compute LGHD for this scale
                scale_descriptor = lghd(scale_patches)
                descriptor_parts.append(scale_descriptor)
            
            # Concatenate descriptors from all scales
            full_descriptor = np.concatenate(descriptor_parts)
            eh[:, i] = full_descriptor
        
        # Return only valid keypoints and descriptors
        valid_indices = ~kps_to_ignore
        return {
            'kps': kps[:, valid_indices].T,
            'des': eh[:, valid_indices].T
        }
    
    @staticmethod
    def lghd_default_parameters():
        """Default parameters for LGHD"""
        return {'patch_size': 100}