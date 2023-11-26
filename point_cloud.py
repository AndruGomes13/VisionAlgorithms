# External Imports
import numpy as np

# Internal Imports
from pose import Pose
class Point_Cloud:

    def __init__(self, points: np.ndarray = np.zeros((0,3)), descriptors: np.ndarray = None, descriptor_size=128):
        
        self.points = points
        self.descriptors = np.zeros((0,descriptor_size)) if descriptors is None else descriptors

        self.descriptor_size = descriptor_size 
        self.latest_cloud_mask = np.zeros((0,1)) # Indices of the points that were added to the point cloud in the last keyframe


    def add_points(self, new_points, new_descriptors) -> np.ndarray:
        '''
        Adds new 3D points (in the origin frame) to the point cloud and returns the indices of the points that were added.

        Args:
            new_points: The new 3D points to be added to the point cloud
            new_descriptors: The descriptors of the new 3D points
            new_pose: The pose of the camera that observed the new 3D points

        Returns:
            The indices of the points that were added to the point cloud
        
        '''
        assert new_points.shape[0] == new_descriptors.shape[0], "The number of points and descriptors must be equal"
        assert new_points.shape[1] == 3, "The points must be 3D"
        assert new_descriptors.shape[1] == self.descriptor_size, "The descriptors must have the correct size"

        indices = np.arange(self.points.shape[0], self.points.shape[0] + new_points.shape[0]) 
        
        self.points = np.concatenate((self.points, new_points), axis=0)
        self.descriptors = np.concatenate((self.descriptors, new_descriptors), axis=0)
        

        self.latest_cloud_mask = indices

        # Return the indices of the points that were added to the point cloud
        return indices 

    def get_latest_cloud(self):
        """
        Returns the latest point cloud that was added to the point cloud
        """

        return self.points[self.latest_cloud_mask], self.descriptors[self.latest_cloud_mask]



    