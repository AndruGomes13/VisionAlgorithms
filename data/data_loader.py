import os
import numpy as np
import cv2

'''
This file contains the code to load the data from the datasets.
Each dataset is loaded using a different function.
'''

current_path = os.path.dirname(os.path.abspath(__file__)) # Get the current path
data_path = current_path  # Get the data directory


def load_kitti() -> (np.ndarray, np.ndarray, int):
    """
    Loads the KITTI dataset.

    #TODO: Select which data to return and update the docstring.
    Returns:
        numpy.ndarray: K matrix (3x3).
        numpy.ndarray: Ground truth Homogeneous Pose Matrices (Nx4x4).
        int: Last frame number.

    """
    kitti_path = os.path.join(data_path, 'kitti')
    assert 'kitti_path' in locals(), "kitti_path variable is not defined"

    # Load the ground truth pose values (3x4 matrix)
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth.reshape(-1, 3, 4)

    # Convert the ground truth pose values to homogeneous coordinates (4x4 matrix)
    ground_truth_homogeneous = np.zeros((ground_truth.shape[0], 4, 4))
    ground_truth_homogeneous[:, :3, :] = ground_truth
    ground_truth_homogeneous[:, 3, 3] = 1


    last_frame = 4540

    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                    [0, 7.188560000000e+02, 1.852157000000e+02],
                    [0, 0, 1]])
    

    return K, ground_truth_homogeneous, last_frame

def load_malaga():
    """
    Loads the Malaga dataset.

    #TODO: Fix docstring and select which data to return.
    Returns:
        numpy.ndarray: K matrix (3x3).
        numpy.ndarray: Left camera image set (Nx600x800).
        int: Last frame number.

    """
    malaga_path = os.path.join(data_path, 'malaga-urban-dataset-extract-07')
    malaga_image_path = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    assert 'malaga_path' in locals(), "malaga_path variable is not defined"

    left_images = [cv2.imread(os.path.join(malaga_image_path, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(malaga_image_path) if filename.endswith('left.jpg')]
    left_images = np.array(left_images)
    
    last_frame = len(left_images)
    K = np.array([[621.18428, 0, 404.0076],
                  [0, 621.18428, 309.05989],
                  [0, 0, 1]])
    
    return K, left_images, last_frame


def load_parking():
    """
    Loads the Parking dataset.

    #TODO: Fix docstring and select which data to return.
    Returns:
        numpy.ndarray: K matrix (3x3).
        numpy.ndarray: Left camera image set (Nx600x800).
        int: Last frame number.

    """
    parking_path = os.path.join(data_path, 'parking')
    assert 'parking_path' in locals(), "parking_path variable is not defined"

    last_frame = 598
    # Side note: the following code is a bit convoluted, but it didn't work with just np.loadtxt
    K_rows = []  # List to accumulate rows of K

    with open(os.path.join(parking_path, 'K.txt'), 'r') as file:
        lines = file.readlines()

        for line in lines:
            try:
                values = list(map(float, line.strip().rstrip(',').split(',')))
                K_rows.append(values)
            except ValueError as e:
                print(f"Error converting line '{line.strip()}' to float: {e}")

    K = np.array(K_rows)
    # Load the ground truth pose values (3x4 matrix)
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth.reshape(-1, 3, 4)

    # Convert the ground truth pose values to homogeneous coordinates (4x4 matrix)
    ground_truth_homogeneous = np.zeros((ground_truth.shape[0], 4, 4))
    ground_truth_homogeneous[:, :3, :] = ground_truth
    ground_truth_homogeneous[:, 3, 3] = 1

    return K, ground_truth_homogeneous, last_frame