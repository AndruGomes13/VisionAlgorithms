# Vision Algorithms for Mobile Robotics final project

import os
import numpy as np

# Setup
ds = 1  # 0: KITTI, 1: Malaga, 2: parking
# To make the following code work place a folder containing
# the code together with the folders containing the datasets
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)  # Get the parent directory
print(f"Parent Path: {parent_path}")

# Datasets paths from parent directory
kitti_path = os.path.join(parent_path, 'kitti')
malaga_path = os.path.join(parent_path, 'malaga-urban-dataset-extract-07')
parking_path = os.path.join(parent_path, 'parking')

if ds == 0:
    assert 'kitti_path' in locals(), "kitti_path variable is not defined"
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    last_frame = 4540
    K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                  [0, 7.188560000000e+02, 1.852157000000e+02],
                  [0, 0, 1]])
elif ds == 1:
    assert 'malaga_path' in locals(), "malaga_path variable is not defined"
    left_images = [filename for filename in os.listdir(os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')) if filename.endswith('.png')]
    last_frame = len(left_images)
    K = np.array([[621.18428, 0, 404.0076],
                  [0, 621.18428, 309.05989],
                  [0, 0, 1]])
elif ds == 2:
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

    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
else:
    assert False

# Print the K matrix to see if the code works properly
print(K)
