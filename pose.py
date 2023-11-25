# External Imports
import numpy as np




class Pose:
    
    def __init__(self, pose: np.ndarray, points_3d_ix: np.ndarray = np.zeros((0,1)), points_2d: np.ndarray = np.zeros((0,2))):

        self.T = pose
        self.points_3d_ix = points_3d_ix    # The indices of the 3D points (with respect to the point cloud) that are seen by the pose.
        self.points_2d = points_2d          # The 2D points in the image that correspond to the 3D points

    



