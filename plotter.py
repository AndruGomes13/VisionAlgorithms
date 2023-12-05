# main imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import networkx as nx #NOTE: I just removed this because it was giving me an error and we don't need it yet.
import time


from matplotlib.lines import Line2D
# internal imports
from visual_odometry import Visual_Odometry


class Plotter:
    def __init__(self) -> None:
        self.estimate_size = 6 # size of the state estimate: x,y,z,theta,phi,psi 
        self.first_pose    = np.array([0,0,0,1]) # in generalized coordinates
        self.estimated_pose_history = np.zeros((3,20)) # can grow to be more. 

        self.last_50_poses = np.empty((self.estimate_size, 50))
        self.lidar_cmap = plt.get_cmap('hsv')


    def plot_pointcloud_3d(self, pointCloud): # TODO: pointcloud Object assert, Current placeholders for pointcloud object
        '''
        Inputs: A pointcloud object made of [[x,y,z,dist],[]...]
        Shows an image
        '''
        x = pointCloud[:,0]
        y = pointCloud[:,1]
        z = pointCloud[:,2]
        dist = pointCloud[:,3]

        fig_pointCloud = plt.figure()
        ax  = fig_pointCloud.add_subplot(projection='3d')
        img = ax.scatter(x, y, z, c= dist, cmap=self.lidar_cmap)
        fig_pointCloud.colorbar(img)

        plt.show()


    def plot_pointcloud_topdown(self, pointCloud):
        '''
        Inputs: A pointcloud object made of [[x,y,z,dist], []..]
        Shows an image
        '''
        x = pointCloud[:,0]
        y = pointCloud[:,1]
        z = pointCloud[:,2]
        dist = pointCloud[:,3]

        fig_pointCloud = plt.figure()
        ax  = fig_pointCloud.add_subplot()
        img = ax.scatter(x, y, c= dist, cmap=self.lidar_cmap)
        fig_pointCloud.colorbar(img)

        plt.show()

    def plot_overlay(self, img, features):
        '''
        Inputs: img: the image as a video_dataset[index]. I.e. the full image is passed to the method.
                features: the features as a list of [u,v] coordinates. 
        '''
        raise NotImplementedError   
    
    def plot_overlay_3d(self, img, features, proj_mat):
        '''
        Inputs: img: the image as a video_dataset[index]. I.e. the full image is passed to the method.
                features: the features as a list of [x,y,z] coordinates.  Projected in the 3d world frame. 
                proj_mat: the 3d -> to uv projection matrix. 
        '''

        raise NotImplementedError


    def plot_tracked_features(self, img, features):
        print("I am printing the tracked features!")

        raise NotImplementedError

    


    def plot_path_global(self, poses):
        '''
        Inputs: list of poes, in the global frame. 
        Shows an image. 
        '''
        raise NotImplementedError

    def plot_path_local(self, poses):
        '''
        Inputs: list of poes, in the global frame. 
        Shows an image. 
        '''
        raise NotImplementedError
    
class Error:
    def __init__(self) -> None:
        self.ground_truth = []
        self.estimate_short_horizon = []
        self.estimate_global = []
    
    def calculate_local_error(self):
        raise NotImplementedError

    def calculate_global_error(self):
        raise NotImplementedError




