# Image Registration Characterization for Remote Sensing
In this repository are present all the opencv method for Feature detection and description.
Also the **RIFT**, **OS-SIFT**, **SAR-SIFT**, **PSO-SIFT**, **LGHD** and **HOPC** are implemented in python.

## Current state
-  SIFT : Implemented
-  SURF : Implemented
-  ORB : Implemented
-  AKAZE : Implemented
-  RIFT : Implemented
-  HOPC : More analysis needed
-  LGHD : To port in Python
-  OS-SIFT : To port in Python
-  ASIFT : To port in Python
-   ~~POS-GIFT ~~ : Code obfuscated
-   ~~CoFSM ~~ : Code obfuscated

## Metrics
The extracted metrics are
- Number of Match (**NM**)
- Number of correct match (**NCM**)
- Total correct point ratio (**NM/NCM**)
- RMSE
- Median execution time
- Mean execution time

## Credits
### Methods
- Li, Jiayuan, et al. "RIFT2: Speeding-up RIFT with a new rotation-invariance technique."
- Hao Li, Xinyue Gao, Xing Li, and Bin Wu "A modified HOPC algorithm for SAR image registration", Proc. SPIE 12129, International Conference on Environmental Remote Sensing and Big Data (ERSBD 2021), 121290B (9 December 2021); https://doi.org/10.1117/12.2625569
- https://github.com/Pyxel0524/HOPC-Optical-to-SAR-registration
- Y. Xiang, F. Wang and H. You, "OS-SIFT: A Robust SIFT-Like Algorithm for High-Resolution Optical-to-SAR Image Registration in Suburban Areas," in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 6, pp. 3078-3090, June 2018, doi: 10.1109/TGRS.2018.2790483.
- W. Ma et al., "Remote Sensing Image Registration With Modified SIFT and Enhanced Feature Matching," in IEEE Geoscience and Remote Sensing Letters, vol. 14, no. 1, pp. 3-7, Jan. 2017, doi: 10.1109/LGRS.2016.2600858.
- Lowe, David G. "Distinctive image features from scale-invariant keypoints."
- Bay, Herbert, Tinne Tuytelaars, and Luc Van Gool. "Surf: Speeded up robust features." Computer Vision–ECCV 2006: 9th European Conference on Computer Vision, Graz, Austria, May 7-13, 2006.
- E. Rublee, V. Rabaud, K. Konolige and G. Bradski, "ORB: An efficient alternative to SIFT or SURF," 2011 International Conference on Computer Vision, Barcelona, Spain, 2011, pp. 2564-2571, doi: 10.1109/ICCV.2011.6126544.
- Alcantarilla, Pablo F., and T. Solutions. "Fast explicit diffusion for accelerated features in nonlinear scale spaces."

### Dataset used
- Jiang, X., Ma, J., Xiao, G., Shao, Z., & Guo, X. (2021). A review of multimodal image matching: Methods and applications. Information Fusion, 73, 22–71.
- Y. Yao, Y. Zhang, Y. Wan, X. Liu, X. Yan and J. Li, "Multi-Modal Remote Sensing Image Matching Considering Co-Occurrence Filter," in IEEE Transactions on Image Processing, vol. 31, pp. 2584-2597, 2022, doi: 10.1109/TIP.2022.3157450.
- Y. Xiang, R. Tao, F. Wang, H. You and B. Han, "Automatic Registration of Optical and SAR Images Via Improved Phase Congruency Model," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 5847-5861, 2020, doi: 10.1109/JSTARS.2020.3026162.
- Huang, Meiyu, et al. "The QXS-SAROPT dataset for deep learning in SAR-optical data fusion."
- https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain?resource=download










