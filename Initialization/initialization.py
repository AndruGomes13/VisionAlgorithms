import os
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

'''
This file contains the code to initialize the continuous pipeline.
Each dataset is initialized using a different function.
'''

current_path = os.path.dirname(os.path.abspath(__file__)) # Get the current path
parent_path = os.path.dirname(current_path)
data_path = os.path.join(parent_path, 'data')

# Additional functions used in the initialization functions


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum suppression of a
    (2r + 1)*(2r + 1) box around the current maximum.

    INPUT:
        - scores (n, m): cornerness scores of every point in the image
        - num (int 1): number of strongest keypoints to extract
        - r (int 1): radius for non-maximum suppression

    OUTPUT:
        - keypoints (2, num): 2D coordinates of extracted keypoints
    """
    pass
    keypoints = np.zeros([2, num])
    temp_scores = np.pad(scores, [(r, r), (r, r)], mode='constant', constant_values=0)
    for i in range(num):
        kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)
        keypoints[:, i] = np.array(kp) - r
        temp_scores[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)] = 0

    return keypoints


def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix
    containing the keypoint coordinates

    INPUTS:
        - img (n, m): image
        - keypoints(2, num): identified keypoints in the image
        - r (int 1): radius of the descriptor's patch
    """
    pass
    N = keypoints.shape[1]
    desciptors = np.zeros([(2*r+1)**2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(N):
        kp = keypoints[:, i].astype(int) + r
        desciptors[:, i] = padded[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)].flatten()

    return desciptors


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor.

    INPUTS:
        - query_desciptors (M, num): descriptors of etracted keypoints from second frame
        - database_descriptors (M, num): descriptors of etracted keypoints from first frame
        - match_lambda (int 1): paramter for checking SSD

    OUTPUTS:
        - matches: matches(i) will be -1 if there is no database descriptor with an SSD < lambda * min(SSD).
                   No elements of matches will be equal except for the -1 elements.
    """
    pass
    dists = cdist(query_descriptors.T, database_descriptors.T, 'euclidean')
    matches = np.argmin(dists, axis=1)
    dists = dists[np.arange(matches.shape[0]), matches]
    min_non_zero_dist = dists.min()

    matches[dists >= match_lambda * min_non_zero_dist] = -1

    # remove double matches
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]

    return unique_matches


def plotMatches(matches, query_keypoints, database_keypoints):
    """
    Function to plot matches between contiguous frames

    INPUTS:
        - matches: match correspondances between query and database keypoints, as defined in the above function
        - query_keypoints (2, num): 2D-coordinates of extracted keypoints in the current frame
        - database_keypoints (2, num): 2D-coordinates of extracted keypoints in the previous frame

    OUTPUT: No numeric output

    """
    query_indices = np.nonzero(matches >= 0)[0]
    match_indices = matches[query_indices]

    x_from = query_keypoints[0, query_indices]
    x_to = database_keypoints[0, match_indices]
    y_from = query_keypoints[1, query_indices]
    y_to = database_keypoints[1, match_indices]

    for i in range(x_from.shape[0]):
        plt.plot([y_from[i], y_to[i]], [x_from[i], x_to[i]], 'g-', linewidth=2)


def cross2Matrix(v):
    """
    Build skew-symmetric matrix from a 3x1 vector.

    INPUT:
        - v: 3x1 NumPy array or list representing the vector

    OUTPUT:
        - 3x3 NumPy array representing the skew-symmetric matrix
    """
    if len(v) != 3:
        raise ValueError("Input vector must have length 3.")

    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     INPUT:
        - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
        - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
        - M1 np.ndarray(3, 4): projection matrix corresponding to first image
        - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     OUTPUT:
        - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    pass
    assert(p1.shape == p2.shape), "Input points dimension mismatch"
    assert(p1.shape[0] == 3), "Points must have three columns"
    assert(M1.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"
    assert(M2.shape == (3,4)), "Matrix M1 must be 3 rows and 4 columns"

    num_points = p1.shape[1]
    P = np.zeros((4, num_points))

    # Linear Algorithm
    for i in range(num_points):
        # Build matrix of linear homogeneous system of equations
        A1 = cross2Matrix(p1[:, i]) @ M1
        A2 = cross2Matrix(p2[:, i]) @ M2
        A = np.r_[A1, A2]

        # Solve the homogeneous system of equations
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:, i] = vh.T[:,-1]

    # Dehomogenize (P is expressed in homoegeneous coordinates)
    P /= P[3,:]

    return P


##############

def init_kitti(test):
    """
    Loads image 1 and 3 from the KITTI dataset and extracts a set of 2D-3D correspondences through Harris
    feature detector and patch descriptor, the relative pose between the frames and triangulates 3D landmarks

    INPUT:
        - test: boolean variable for testing correct functioning and assessing results

    OUTPUT:
        - P np array (3, N): 3D coordinates of N triangulated points

    """
    # Additional variables for Harris detector and descriptor
    corner_patch_size = 9
    harris_kappa = 0.08
    num_keypoints = 200
    nonmaximum_supression_radius = 8
    descriptor_radius = 9
    match_lambda = 4

    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                  [0, 7.188560000000e+02, 1.852157000000e+02],
                  [0, 0, 1]])

    # Loading first image 000000.png
    kitti_path = os.path.join(data_path, 'kitti', '05', 'image_0')
    img0 = cv2.imread(os.path.join(kitti_path, '000000.png'), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        print("Error: Couldn't open the image.")
        return

        # Check the data type of the image
    if not isinstance(img0, np.ndarray):
        print("Error: Image data type is not numpy.ndarray.")
        return

    # Loading third image 000002.png
    img1 = cv2.imread(os.path.join(kitti_path, '000002.png'), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        print("Error: Couldn't open the image.")
        return

        # Check the data type of the image
    if not isinstance(img1, np.ndarray):
        print("Error: Image data type is not numpy.ndarray.")
        return

    # Harris feature detection in img0
    dst_0 = cv2.cornerHarris(np.float32(img0), blockSize=corner_patch_size, ksize=3, k=harris_kappa)
    keypoints_0 = selectKeypoints(dst_0, num_keypoints, nonmaximum_supression_radius)

    # Harris feature detection in img1
    dst_1 = cv2.cornerHarris(np.float32(img1), blockSize=corner_patch_size, ksize=3, k=harris_kappa)
    keypoints_1 = selectKeypoints(dst_1, num_keypoints, nonmaximum_supression_radius)

    # Patch descriptors
    descriptors_0 = describeKeypoints(img0, keypoints_0, descriptor_radius)
    descriptors_1 = describeKeypoints(img1, keypoints_1, descriptor_radius)
    matches = matchDescriptors(descriptors_1, descriptors_0, match_lambda)

    if test:
        plt.clf()
        plt.close()
        plt.imshow(img1, cmap='gray')
        plt.plot(keypoints_1[1, :], keypoints_1[0, :], 'ro', markersize=2)
        plotMatches(matches, keypoints_1, keypoints_0)
        plt.tight_layout()
        plt.axis('off')
        plt.suptitle('Image 1, identified keypoints and their movement', fontsize=16, y=0.82)
        plt.title('The matched keypoints move back as the camera moves forward', fontsize=9)
        plt.show()

    F, mask = cv2.findFundamentalMat(
        keypoints_0[:, matches[matches > 0]].T,
        keypoints_1[:, matches > 0].T,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.99
    )
    E = K.T @ F @ K
    _, R_1_W, T_1_W, _ = cv2.recoverPose(E, keypoints_0[:, matches[matches > 0]].T, keypoints_1[:, matches > 0].T, K, K)
    if test:
        print('Fundamental matrix:')
        print(F)
        print('\nRelative rotation matrix:')
        print(R_1_W)
        print('\nRelative translation vector:')
        print(T_1_W)

    # Triangulate points
    M0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    M1 = K @ np.hstack([R_1_W, T_1_W])
    points_homogeneous = cv2.triangulatePoints(M0, M1, keypoints_0[:, matches[matches > 0]], keypoints_1[:, matches > 0]) # Homogenous coordinates
    points_3D = points_homogeneous[:3, :] / points_homogeneous[3, :]

    if test:
        # Plotting of 3D points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points_3D[0, :], points_3D[1,:], points_3D[2, :], c='b', marker='o')
        # Plotting of camera positions
        ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='Origin (0, 0, 0)')
        ax.scatter(T_1_W[0], T_1_W[1], T_1_W[2], c='g', marker='o', s=100, label='Point at T_1_W')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Landmarks in World Frame')

        plt.show()
    # Note: this last part of the triangulation doesn't seem to work. There is either a problem with the plot
    # or a problem in the computation of either the triangulated points or the relative pose and translation
    return points_3D


# True: see plots and intermediate values of variables to assess correctness
# False: just retrieve the 3D points found
P = init_kitti(True)
