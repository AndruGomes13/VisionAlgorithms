# Imports
import numpy as np
import cv2
from scipy.optimize import least_squares

# Internal Imports
from point_cloud import Point_Cloud
from pose import Pose

class Visual_Odometry:
    """
    
    
    
    """

    def __init__(self, params):
        """
        Initializes the Vision Odometry object.
        
        Args:
            feature_detector (cv2.Feature2D): A feature detector object.
            matcher (cv2.DescriptorMatcher): A descriptor matcher object.
            K (numpy.ndarray): The camera matrix.
        """
        self.feature_detector = params["Feature_Detector"]
        self.matcher = params["Feature_Matcher"]
        self.K = params["K"]


    def bootstrap(self, image_1: np.ndarray, image_2: np.ndarray) -> None:
        """
        Bootstraps the Vision Odometry object.

        Generates the initial features, descriptor, second keyframe and initial point cloud.

        NOTE: For now it only computes the second keypoing pose and initial point cloud coordinates.
        TODO: Build the actual point cloud and store the needed data.

        Args:
            image_1 (numpy.ndarray): The first image.
            image_2 (numpy.ndarray): The second image.
        
        Returns:
            TODO: Define what to return.
        """
        # Find the matches between the two images
        src_pts, dst_pts, src_des, dst_des = self.find_matches(image_1, image_2)

        # Get the pose of the second image
        T, inlier_mask = self.get_pose(src_pts, dst_pts)
   

        # Update the points to only contain the inlier points
        src_pts = src_pts[inlier_mask.ravel() == 1]
        dst_pts = dst_pts[inlier_mask.ravel() == 1]
        src_des = src_des[inlier_mask.ravel() == 1]
        dst_des = dst_des[inlier_mask.ravel() == 1]

        # Get the 3D points (in the first camera frame)
        points_3d = self.get_3d_points(src_pts, dst_pts, T)

        # Remove points that are too far away
        MAX_DISTANCE = 300 #TODO: Make this a parameter that can be set by the user
        points_3d_distance = np.linalg.norm(points_3d, axis=1)
        points_3d = points_3d[points_3d_distance < MAX_DISTANCE]

        # Build Pose
        pose = Pose(T, points_3d, src_pts)

        # Build Point Cloud
        point_cloud = Point_Cloud(points_3d, src_des, descriptor_size=src_des.shape[1])

        return point_cloud, (src_pts, dst_pts, src_des, dst_des, pose)
    
        
    def find_matches(self, img_1: np.ndarray, img_2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Finds and matches the features of two images.

        NOTE: This function is currently coded for SIFT detector and descriptor.
        TODO: Make it work for any detector and descriptor (maybe make it a function that calls other functions designed for other descriptors).
        
        Args:
            img_1 (numpy.ndarray): The first image.
            img_2 (numpy.ndarray): The second image.

        Returns:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.
            d1 (numpy.ndarray) (Nxd): The descriptors of the matched features in the first image.
            d2 (numpy.ndarray) (Nxd): The descriptors of the matched features in the second image.

            Where:
                N: Number of matched features.
                d: Dimension of the descriptor.
        """
        # Find the keypoints and descriptors
        kp_1, des_1 = self.feature_detector.detectAndCompute(img_1, None)
        kp_2, des_2 = self.feature_detector.detectAndCompute(img_2, None)

        # Match the features
        matches = self.matcher.knnMatch(des_1, des_2, k=2)

        # Filter the matches
        THRESHOLD = 0.7 #TODO: Make this a parameter that can be set by the user
        good_matches = [m for m,n in matches if m.distance < THRESHOLD*n.distance]

        # Image points of the matched features
        q1 = np.float32([kp_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        q2 = np.float32([kp_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Descriptor of the matched features
        d1 = np.float32([des_1[m.queryIdx] for m in good_matches])
        d2 = np.float32([des_2[m.trainIdx] for m in good_matches])

        return q1, q2, d1, d2

    def get_3d_points(self, q1:np.ndarray, q2:np.ndarray, T:np.ndarray) -> np.ndarray:

        """
        Computes the 3D points (defined in reference frame 1) from the matched points q1 and q2 and the homogeneous transformation matrix T of 
        the second pose (in which the q2 points are defined).

        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.
            T (numpy.ndarray) (4x4): The homogeneous transformation matrix of the second pose (in which the q2 points are defined).

        Returns:
            points_3d (numpy.ndarray) (Nx3): The 3D points in the first camera frame.

        """

        K = self.K

        P1 = K @ np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
        P2 = K @ T[:3, :]

        points_4d_1 = cv2.triangulatePoints(P1, P2, q1.T, q2.T) # Homogeneous coordinates in the first camera frame

        # Convert to un-homogeneous coordinates
        points_4d_1 = points_4d_1 / points_4d_1[3]

        points_3d = points_4d_1[:3].T

        return points_3d
            
    def get_pose(self, q1:np.ndarray, q2:np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Computes the pose of the second camera frame with respect to the first camera frame (Using the the 5 point algorithm).

        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.

        Returns:
            T (numpy.ndarray) (4x4): The homogeneous transformation matrix of the second pose (in which the q2 points are defined).

        """
        K = self.K

        # Get essential matrix
        E, inliers_mask = cv2.findEssentialMat(q1, q2, K, cv2.RANSAC, 0.999, 1.0) #NOTE: Make the parameters be set by the user
        
        # Update the inlier points
        q1 = q1[inliers_mask.ravel() == 1]
        q2 = q2[inliers_mask.ravel() == 1]
        
        # Decompose the essential matrix to get the possible poses
        # TODO: Check cv.recoverPose() function. This seems to do everything that is done here.
        R1, R2, t = cv2.decomposeEssentialMat(E)
        # Get the correct pose from the 4 possible poses (using Least Square Approximation)
        T, mask = self._get_correct_pose(R1, R2, t, q1, q2)

        print(mask.shape, mask.sum(), mask.sum()/mask.shape[0])
        # g, R, t, mask = cv2.recoverPose(E, q1, q2)
        # print("R: ", R)
        # print("t: ", t)
        # print("g: ", g)
        # print("mask: ", (mask==255).sum())

        # T = self.pose_RT(R,t)

        # Update inlier mask
        inliers_mask[inliers_mask == 1] = mask.ravel()
        # Update the inlier points
        q1 = q1[mask.ravel() == 1]
        q2 = q2[mask.ravel() == 1]

        # Apply non-linear refinement
        NON_LINEAR_REFINEMENT = True #TODO: Make this a parameter that can be set by the user
        if NON_LINEAR_REFINEMENT:
            T = self.get_pose_refinement(q1, q2, T)

        return T, inliers_mask

    def _get_correct_pose(self, R1:np.ndarray, R2:np.ndarray, t:np.ndarray, q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
        """
        Finds the correct pose from the 4 possible poses (using Least Square Approximation).

        Args:
            R1 (numpy.ndarray) (3x3): The first possible rotation matrix.
            R2 (numpy.ndarray) (3x3): The second possible rotation matrix.
            t (numpy.ndarray) (3x1): The translation vector.
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.

        Returns:
            T (numpy.ndarray) (4x4): The correct transformation matrix.
        """


        K = self.K

        # Get the 4 possible transformations matrices
        T1 = np.concatenate((np.concatenate((R1, t.reshape(-1,1)), axis=1), np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)
        T2 = np.concatenate((np.concatenate((R1, -t.reshape(-1,1)), axis=1), np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)
        T3 = np.concatenate((np.concatenate((R2, t.reshape(-1,1)), axis=1), np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)
        T4 = np.concatenate((np.concatenate((R2, -t.reshape(-1,1)), axis=1), np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)
        
        transformations = [T1, T2, T3, T4]

        # Triangulate a few points and check if the depth is positive
        max_points = -1
        for i, T in enumerate(transformations):
            P1 = K @ np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
            P2 = K @ T[:3, :]

            # Triangulate the points
            points_4d_1 = cv2.triangulatePoints(P1, P2, q1.T, q2.T) # Homogeneous coordinates in the first camera frame
            points_4d_2 = T @ points_4d_1 # Homogeneous coordinates in the second camera frame

            # Convert to un-homogeneous coordinates
            points_4d_1 = points_4d_1 / points_4d_1[3]
            points_4d_2 = points_4d_2 / points_4d_2[3]

            # Find number of points with positive depth
            inliers_mask = np.logical_and(points_4d_1[2] > 0, points_4d_2[2] > 0)
            n_points = np.sum(inliers_mask)

            # Update to the transformation with the most points with positive depth
            if n_points > max_points:
                max_points = n_points
                best_T = T

        return best_T, inliers_mask
            
    def get_pose_refinement(self, q1:np.ndarray, q2:np.ndarray, T:np.ndarray) -> np.ndarray:

        """
        Refines the pose using Reprojection Error (Levenberg-Marquardt).
        TODO: Implement other error metrics (e.g. Epipolar Line Distance, etc)

        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.
            T (numpy.ndarray) (4x4): Initial guess of the homogeneous transformation matrix of the second pose.

        Returns:
            T (numpy.ndarray) (4x4): The refined homogeneous transformation matrix of the second pose.
        """

        K = self.K

        # Refine the pose using non-linear least squares
        func = self._get_squared_reprojection_error_func(q1, q2)

        # Format the initial guess as a 1-D array
        r_vec = cv2.Rodrigues(T[:3,:3])[0]
        t_vec = T[:3,3]
        x_0 = np.concatenate((r_vec.ravel(), t_vec.ravel()))

        # Run the optimization (Levenberg-Marquardt method)
        res = least_squares(func, x_0, method='lm')

        # Build the final transformation matrix
        R = cv2.Rodrigues(res.x[:3])[0]
        t = res.x[3:]
        
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        
        return T

    def _get_squared_reprojection_error_func(self, q1, q2):
        """
        Returns a function that calculates the squared reprojection error of set of points q1 and q2, and the homogeneous transformation matrix.
        
        The return function receives a 1-D array (x) of the parameters of the transformation matrix and returns a 1-D array of the residuals.
        
        The first 3 parameters of x are the rotation vector (r_vec) and the last 3 parameters are the translation vector (t_vec).
        
        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.

        Returns:
            func (function): A function that calculates the squared reprojection error.
        
        """
        K = self.K

        def func(x):
            """
            Calculates the squared reprojection error of set of points q1 and q2, with the homogeneous transformation matrix.

            Args:
                x (numpy.ndarray) (6): The parameters of the transformation matrix.
                Where x[:3] are the rotation vector (r_vec) and x[3:] are the translation vector (t_vec).

            Returns:
                residual (numpy.ndarray) (2*N): The squared reprojection error of the points.
            
            """

            # Build T from x
            T = np.concatenate((cv2.Rodrigues(x[:3])[0], x[3:].reshape(-1,1)), axis=1)
            T = np.concatenate((T, np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)

            # Triangulate the points
            points_3d = self.get_3d_points(q1, q2, T)

            # Project the points to the image plane
            points_2d_1 = cv2.projectPoints(points_3d, np.eye(3), np.zeros((3,1)), K, None)[0].reshape(-1,2)
            points_2d_2 = cv2.projectPoints(points_3d, T[:3,:3], T[:3,3], K, None)[0].reshape(-1,2)

            # Calculate the reprojection error (distance between the projected points and the actual points)
            error_1 = np.linalg.norm(points_2d_1 - q1, axis=1)
            error_2 = np.linalg.norm(points_2d_2 - q2, axis=1)

            # Create 1-D residual vector
            residual = np.concatenate((error_1.ravel(), error_2.ravel()))

            return residual

        return func

    
    def pose_RT(self, R,t):
        return np.concatenate((np.concatenate((R, t.reshape(-1,1)), axis=1), np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)