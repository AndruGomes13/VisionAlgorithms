# Vision Algorithms for Mobile Robotics final project

import os
import numpy as np
from data.data_loader import load_kitti, load_malaga, load_parking

# Setup
ds = 1  # 0: KITTI, 1: Malaga, 2: parking
# To make the following code work place a folder containing
# the code together with the folders containing the datasets
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)  # Get the parent directory
data_path = os.path.join(current_path, 'data')  # Get the data directory

print(f"Parent Path: {parent_path}")

if ds == 0:
    print("Loading KITTI")
    K, ground_truth, last_frame = load_kitti()
    print("Finished loading")
elif ds == 1:
    print("Loading Malaga")
    K, left_images, last_frame = load_malaga()
    print("Finished loading")
elif ds == 2:
    print("Loading Parking")
    K, ground_truth, last_frame = load_parking()
    print("Finished loading")
else:
    assert False

# Print the K matrix to see if the code works properly
print(len(left_images))
