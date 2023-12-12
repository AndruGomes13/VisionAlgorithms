# Imports
from typing import Any, Union, Tuple, List
import numpy as np
import cv2
from scipy.optimize import least_squares

# Internal Imports
from point_cloud import Point_Cloud
from pose import Pose


class Visual_Odometry_2:
    def __init__(self, params) -> None:
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

        self.descriptor_size = 128 #TODO: This is hardcoded for SIFT. Make it work for any descriptor.

        ## State initialization of the Vision Odometry
        # Keypoints array
        self.S = np.zeros((0, 2), dtype=np.float32)
        # 3D points array
        self.P = np.zeros((0, 3), dtype=np.float32)
        # Candidate keypoints array
        self.C = np.zeros((0, 2), dtype=np.float32)
        # First observation of candidate keypoints array
        self.F = np.zeros((0, 2), dtype=np.float32)
        # Camera poses of first observation of candidate keypoints array
        self.T = np.zeros((0,4,4), dtype=np.float32)

        # Last image
        self.prev_image= None


        # Parameters
        # Lucas-Kanade parameters
        self.klt_params = {"winSize" : (10, 10), 
                    "maxLevel" : 3, 
                    "criteria" : (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.05)}


        # PnP RAMSAC parameters
        self.pnp_ransac_params = {"iterationsCount": 100,
                    "reprojectionError": 2.0,
                    "confidence": 0.99,
                    "distCoeffs": None,
                    "flags": cv2.SOLVEPNP_ITERATIVE,
                    "useExtrinsicGuess": False
                    }
        
        self.base_line_angle = 30/180 * np.pi 


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
        self.prev_image = image_1
        
        # Find keypoints and descriptors of the images      
        kp_des = self.get_image_keypoints_and_descriptors((image_1, image_2))
        kp_1, des_1 = kp_des[0]
        kp_2, des_2 = kp_des[1]
        # Indicies of the inlier points
        kp_1_id = np.arange(len(kp_1))
        kp_2_id = np.arange(len(kp_2))

        T, points_3d_acceptable, matches_1_id, matches_2_id = self.get_relative_pose_and_3d_points_from_features(kp_1, des_1, kp_2, des_2)


        # Update the inlier points indices
        kp_1_id = kp_1_id[matches_1_id]
        kp_2_id = kp_2_id[matches_2_id]

        # Update state of the vision
        self.S = kp_2[kp_2_id]
        self.P = points_3d_acceptable


        assert self.S.shape[0] == self.P.shape[0], "The number of keypoints and 3D points must be equal"
        assert self.S.shape[0] != 0, "The number of keypoints and 3D points must be greater than zero"
        assert self.S.shape[1] == 2, "The keypoints must be 2D"
        assert self.P.shape[1] == 3, "The 3D points must be 3D"
        
        return self.S, self.P
    
    def process_new_frame(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a new frame.

        Args:
            image (numpy.ndarray): The new image.

        """

        #TODO: Sometimes some keypoints are outside the image. Check why and fix it.
        #TODO: Sometimes there is a large percentage of keypoints that are not inliers. Check why and fix it.




        ## Track new keypoints from old keypoints using KLT
        new_kp, mask, err = cv2.calcOpticalFlowPyrLK(self.prev_image, image, self.S, None, **self.klt_params)
        # Filter the keypoints
        mask = mask.ravel() == 1
        # self.S = self.S[mask]
        self.S = new_kp[mask]
        self.P = self.P[mask]

        aux = mask.sum()/len(mask)
        if aux < 0.5:
            print("KLT keypoints:", aux)

        # Find pose with PnP
        T, inliers = self.get_pose_3d_2d(self.P, self.S)
        # Update keypoints with inliers
        self.S = self.S[inliers]
        self.P = self.P[inliers]

        aux = inliers.sum()/len(inliers)
        if aux < 0.5:
            print("Inlier keypoints:", inliers.sum()/len(inliers))

        # TODO: Refine pose 
        T = self.pose_refinement_3d_2d(T, self.P, self.S)



        ## Track candidate keypoints from old candidate keypoints using KLT
        if len(self.C) != 0:
            new_kp, mask, err = cv2.calcOpticalFlowPyrLK(self.prev_image, image, self.C, None, **self.klt_params)

            # Filter the candidates
            mask = mask.ravel() == 1 # Mask of the candidates that were tracked
            self.C = new_kp[mask] # Update the candidate keypoints
            self.F = self.F[mask]
            self.T = self.T[mask]

            # Check which candidates could be added to the keypoints
            # Triangulate the points and check the baseline angle
            points_3d, angle_mask = self.check_candidate_keypoints(T, self.base_line_angle)

            # Update keypoint with the new candidates
            self.S = np.concatenate((self.S, self.C[angle_mask]), axis=0)
            self.P = np.concatenate((self.P, points_3d), axis=0)

            # Remove the candidates that were added to the keypoints
            self.C = self.C[~angle_mask]
            self.F = self.F[~angle_mask]
            self.T = self.T[~angle_mask]


        ## Find new candidate keypoints
        # Initialize new cadidate keypoints with Harris
        new_corners = cv2.goodFeaturesToTrack(image,1500,0.01,10).reshape(-1,2) #TODO: Make this a parameter that can be set by the user
       
        
        # Filter new candidate keypoints that are too close to previous keypoints
        previous_keypoints = np.concatenate((self.S, self.C), axis=0)
        new_corners = self.filter_new_keypoints(previous_keypoints, new_corners, 10)
    
        # Update candidate keypoints
    
        self.C = np.concatenate((self.C, new_corners), axis=0)
        self.F = np.concatenate((self.F, new_corners), axis=0)
        self.T = np.concatenate((self.T, np.tile(T, (new_corners.shape[0], 1, 1))), axis=0)
        
        self.prev_image = image
        # print("Keypoints:", self.S.shape)
        # print("Candidate Keypoints:", self.C.shape)
        return T

    def check_candidate_keypoints(self, T: np.ndarray, baseline_angle = 10/180 * np.pi) -> np.ndarray:
        """
        Checks which candidate keypoints can be added to the keypoints.

        Args:
            T (numpy.ndarray): The camera poses of the current observation of the candidate keypoints.

        Returns:
            mask (numpy.ndarray): The mask of the candidate keypoints that can be added to the keypoints.
        """
        # Triangulate the points
        points_3d = np.zeros((len(self.C),3))
        for i in range(len(self.C)):
            point_3d = self.get_3d_points(self.F[[i]], self.C[[i]], self.T[i], T)
            points_3d[i] = point_3d
            # point_3d = self.T[i] @ np.concatenate((point_3d, np.array([1]).reshape(1,-1)), axis=1).T
            # points_3d[i] = point_3d.T[0,:3]

        # Check the baseline angle
        vector_1 = points_3d - T[:3,3]
        vector_2 = points_3d - self.T[:,:3,3]



        # angle = np.arccos(np.tensordot(vector_1, vector_2, axes=(1,1)) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))
        angle = np.arccos(np.sum(vector_1*vector_2, axis=1) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))

        angle_mask = np.abs(angle) > baseline_angle
        points_3d = points_3d[angle_mask]

        assert points_3d.shape[0] == angle_mask.sum(), "The number of points and mask selected points must be equal"

        return points_3d, angle_mask 



    def filter_new_keypoints(self, prev_keypoints, curr_keypoints, distance_threshold=1.0):
        """Filters out new keypoints that are too close to previous keypoints.
        Args:
            prev_keypoints: Numpy array of previous keypoints of shape (N, 2).
            curr_keypoints: Numpy array of current keypoints of shape (N, 2).
            distance_threshold: Distance threshold below which a new keypoint will
                be removed.
        Returns:
            Array of filtered new keypoints of shape (N, 2).
        """
        acceptable_keypoints = []

        for keypoint in curr_keypoints:
            if np.min(np.linalg.norm(prev_keypoints - keypoint, axis=1)) > distance_threshold:
                acceptable_keypoints.append(keypoint)
        out = np.array(acceptable_keypoints, dtype=np.float32).reshape(-1, 2)

        return out


    def get_pose_3d_2d(self, points_3d:np.ndarray, kp:np.ndarray) -> np.ndarray:
        assert points_3d.shape[0] != 0, "The number of points must be greater than zero"
        assert kp.shape[0] == points_3d.shape[0], "The number of points must be equal"
        assert kp.shape[0] >= 3, "The number of points must be greater than 8" #TODO: Set a minimum number of points
        
        K = self.K
        # TODO: Understand function parameters
        success, R_vec, t, inliers = cv2.solvePnPRansac(points_3d, kp, K, **self.pnp_ransac_params)
        R = cv2.Rodrigues(R_vec)[0]
        T = self.RT_to_pose(R, t)

        # assert inliers.shape[0] == kp.shape[0], "The number of inliers and keypoints must be equal"
        # asse, "The number of inliers and keypoints must be greater than zero"
        # print("Inliers:", inliers)
        print(points_3d)

        assert success == True, "PnP failed"

        return T, inliers.ravel()
    
    def get_image_keypoints_and_descriptors(self, images: Union[np.ndarray, Tuple[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Finds the keypoints and descriptors of a set of images.

        Args:
            images (numpy.ndarray | tuple(numpy.ndarray)): The images.

        Returns:
            keypoints and descriptors (numpy.ndarray | tuple(numpy.ndarray)): The keypoints and descriptors of the images (Keypoint, Descriptor).


        '''
        
        if not (isinstance(images, tuple) or isinstance(images, list)):
            images = (images,)
            

        output = []
        for image in images:
            keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)
            keypoints = np.float32([keypoint.pt for keypoint in keypoints]) # Convert keypoints to numpy array
            output.append((keypoints, descriptors))

        return output if len(output) > 1 else output[0] # Return a tuple if there is more than one image

    def get_relative_pose_and_3d_points_from_features(self, kp_1, des_1, kp_2, des_2):
        
        # Indicies of the inlier points
        kp_1_id = np.arange(len(kp_1))
        kp_2_id = np.arange(len(kp_2))
        
        # Find the matches between the two images
        matches_1_id, matches_2_id = self.find_2d_2d_matches(des_1, des_2)
        
        # Update the inlier points indices
        kp_1_id = kp_1_id[matches_1_id]
        kp_2_id = kp_2_id[matches_2_id]

        # Get the relative pose of the second image frame
        T_2, inlier_id_1, inlier_id_2 = self.get_pose_2d_2d(kp_1[kp_1_id], kp_2[kp_2_id])
        
        # Update the inlier points indices
        kp_1_id = kp_1_id[inlier_id_1]
        kp_2_id = kp_2_id[inlier_id_2]

        # Get the 3D points (in the first camera frame)
        T_1 = np.eye(4) # The first pose is the origin
        points_3d = self.get_3d_points(kp_1[kp_1_id], kp_2[kp_2_id], T_1, T_2)

        # Remove points that are too far away
        MAX_DISTANCE = 300 #TODO: Make this a parameter that can be set by the user
        points_3d_distance = np.linalg.norm(points_3d, axis=1)
        acceptable_points_mask = points_3d_distance < MAX_DISTANCE

        # Update the inlier points indices
        kp_1_id = kp_1_id[acceptable_points_mask]
        kp_2_id = kp_2_id[acceptable_points_mask]
        points_3d_acceptable = points_3d[acceptable_points_mask]

        return T_2, points_3d_acceptable, kp_1_id, kp_2_id

    def find_2d_2d_matches(self, des_1: np.ndarray, des_2: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        
        # Match the features
        # matches = self.matcher.knnMatch(des_1, des_2, k=2)
        matches = self.matcher.match(des_1, des_2)

        # Filter the matches
        # THRESHOLD = 0.7 #TODO: Make this a parameter that can be set by the user
        # good_matches = [m for m,n in matches if m.distance < THRESHOLD*n.distance]
        good_matches = matches
        # Index of keypoints/descriptors of the matched features
        matches_1_id = np.array([m.queryIdx for m in good_matches], dtype=int)
        matches_2_id = np.array([m.trainIdx for m in good_matches], dtype=int)

     
        return matches_1_id, matches_2_id

    def get_3d_points(self, kp_1:np.ndarray, kp_2:np.ndarray, T_1:np.ndarray, T_2:np.ndarray) -> np.ndarray:

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
        assert kp_1.shape == kp_2.shape, "The number of dimensions must be equal"
        assert kp_1.shape[1] == 2, "The points must be 2D"
        assert kp_1.shape[0] != 0, "The number of points must be greater than zero" 
        
        K = self.K

        # Build the projection matrices
        P1 = K @ T_1[:3, :]
        P2 = K @ T_2[:3, :]

        # Triangulate the points
        points_4d_1 = cv2.triangulatePoints(P1, P2, kp_1.T, kp_2.T) # Homogeneous coordinates in the first camera frame

        # Convert to un-homogeneous coordinates
        points_4d_1 = points_4d_1 / points_4d_1[3]

        points_3d = points_4d_1[:3].T

        return points_3d
    
    def get_pose_2d_2d(self, kp_1:np.ndarray, kp_2:np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Computes the pose of the second camera frame with respect to the first camera frame (Using the the 5 point algorithm).

        Args:
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.

        Returns:
            T (numpy.ndarray) (4x4): The homogeneous transformation matrix of the second pose (in which the q2 points are defined).

        """
        assert kp_1.shape == kp_2.shape, "The number of points must be equal"
        assert kp_1.shape[1] == 2, "The points must be 2D"
        assert kp_1.shape[0] > 8, "The number of points must be greater than 8" #TODO: Set a minimum number of points

        K = self.K

        # Indicies of the inlier points
        kp_1_id = np.arange(len(kp_1))
        kp_2_id = np.arange(len(kp_2))

        # Find the essential matrix
        E, mask_E = cv2.findEssentialMat(kp_1, kp_2, K, cv2.RANSAC, 0.999, 1.0) #NOTE: Make the parameters be set by the user (And discover what they do)
        
        # Update the inlier points indices
        kp_1_id = kp_1_id[mask_E.ravel() == 1]
        kp_2_id = kp_2_id[mask_E.ravel() == 1]

        # Get the correct pose
        success, R, t, mask_P = cv2.recoverPose(E, kp_1[kp_1_id], kp_2[kp_2_id])

        # Update the inlier points indices
        kp_1_id = kp_1_id[mask_P.ravel() == 255] # 255 is the value of the inlier mask (Ask OpenCV why)
        kp_2_id = kp_2_id[mask_P.ravel() == 255]

        # Build the transformation matrix
        T_2 = self.RT_to_pose(R,t) # Relative pose of the second camera frame with respect to the first camera frame

        # Apply non-linear refinement
        NON_LINEAR_REFINEMENT = True #TODO: Make this a parameter that can be set by the user
        if NON_LINEAR_REFINEMENT:
            T_2 = self.get_pose_refinement(T_2, kp_1[kp_1_id], kp_2[kp_2_id])

        return T_2, kp_1_id, kp_2_id

    def get_pose_refinement(self, T_i:np.ndarray, kp_1:np.ndarray, kp_2:np.ndarray) -> np.ndarray:

        """
        Refines the pose using Reprojection Error (Levenberg-Marquardt).
        TODO: Implement other error metrics (e.g. Epipolar Line Distance, etc)

        Args:
            T_i (numpy.ndarray) (4x4): Initial guess of the homogeneous transformation matrix of the second pose.
            q1 (numpy.ndarray) (Nx2): The image points of the matched features in the first image.
            q2 (numpy.ndarray) (Nx2): The image points of the matched features in the second image.

        Returns:
            T (numpy.ndarray) (4x4): The refined homogeneous transformation matrix of the second pose.
        """
        assert kp_1.shape == kp_2.shape, "The number of points and dimension must be equal"
        assert kp_1.shape[1] == 2, "The points must be 2D"
        assert kp_1.shape[0] != 0, "The number of points must be greater than zero"


        K = self.K

        # Refine the pose using non-linear least squares
        func = self._get_squared_reprojection_error_func(kp_1, kp_2)

        # Format the initial guess as a 1-D array
        r_vec = cv2.Rodrigues(T_i[:3,:3])[0] # Convert the rotation matrix to a rotation vector
        t_vec = T_i[:3,3]
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
            T_1 = np.eye(4) # TODO: Making this an arguement could be more versatile
            T_2 = T
            points_3d = self.get_3d_points(q1, q2, T_1, T_2)

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

    def pose_refinement_3d_2d(self, T_i:np.ndarray, points_3d:np.ndarray, kp:np.ndarray) -> np.ndarray:
            
            """
            Refines the pose using Reprojection Error (Levenberg-Marquardt).
            """

            assert kp.shape[0] == points_3d.shape[0], "The number of points must be equal"
            assert kp.shape[1] == 2, "The points must be 2D"
            assert points_3d.shape[1] == 3, "The 3D points must be 3D"
            assert kp.shape[0] != 0, "The number of points must be greater than zero"


            K = self.K

            # Refine the pose using non-linear least squares
            func = self._get_squared_reprojection_error_func_3d_2d(points_3d, kp)

            # Format the initial guess as a 1-D array
            R, t = self.pose_to_RT(T_i)
            r_vec = cv2.Rodrigues(R)[0] # Convert the rotation matrix to a rotation vector

            x_0 = np.concatenate((r_vec.ravel(), t.ravel()))

            # Run the optimization (Levenberg-Marquardt method)
            res = least_squares(func, x_0, method='lm')

            # Build the final transformation matrix
            R = cv2.Rodrigues(res.x[:3])[0]
            t = res.x[3:]

            T = self.RT_to_pose(R,t)

            return T



    def _get_squared_reprojection_error_func_3d_2d(self, points_3d, kp):
        K = self.K

        def fun(x):
            # Build T from x
            T = np.concatenate((cv2.Rodrigues(x[:3])[0], x[3:].reshape(-1,1)), axis=1)
            T = np.concatenate((T, np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)

            # Project the 3d points to the image plane
            points_projected = cv2.projectPoints(points_3d, T[:3,:3], T[:3,3], K, None)[0].reshape(-1,2)

            # Calculate the reprojection error (distance between the projected points and the actual points)
            error = np.linalg.norm(points_projected - kp, axis=1)

            # Create 1-D residual vector
            residual = error.ravel()

            return residual
        
        return fun

    # Utility Functions
    def RT_to_pose(self, R,t):
        """
        Builds the homogeneous transformation matrix from the rotation matrix and translation vector.
        """
        return np.concatenate((np.concatenate((R, t.reshape(-1,1)), axis=1), np.array([0, 0, 0, 1]).reshape(1,-1)), axis=0)
    
    def pose_to_RT(self, T):
        """
        Extracts the rotation matrix and translation vector from the homogeneous transformation matrix.
        """
        return T[:3,:3], T[:3,3]

    def _mask_from_indices(self, indices: np.ndarray, size: int) -> np.ndarray:
        """
        Creates a mask from a list of indices.

        Args:
            indices (numpy.ndarray) (N): The indices of the mask.
            size (int): The size of the mask.

        Returns:
            mask (numpy.ndarray) (size): The mask.
        """
        mask = np.zeros((size), dtype=bool)
        mask[indices] = True

        return mask

    def _indices_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Creates a list of indices from a mask.

        Args:
            mask (numpy.ndarray) (size): The mask.

        Returns:
            indices (numpy.ndarray) (N): The indices of the mask.
        """
        return np.arange(0, mask.shape[0], dtype=int)[mask]