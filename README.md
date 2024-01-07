# Monocular Visual Odometry Project

## Course Information

![Image of the ETH Zurich logo](https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/ETH_Zurich_-_Zeichen.svg/250px-ETH_Zurich_-_Zeichen.svg.png){width=200px}
![Image of the University of Zurich logo](https://upload.wikimedia.org/wikipedia/en/5/57/University_of_Zurich_logo.svg){width=200px}


**Institution:** University of Zurich (UZH) and Swiss Federal Institute of Technology (ETH)  
**Course:** Vision Algorithms for Mobile Robotics  
**Instructors:** [Instructor Names]

## Project Overview

This project is a part of the Vision Algorithms for Mobile Robotics course, focusing on implementing and understanding the fundamentals of monocular visual odometry. The goal is to develop an algorithm capable of estimating the 3D motion of a single camera moving through a static environment. This technique is critical in the domains of robotics and autonomous vehicles, where understanding the movement relative to the environment is crucial.

## Objectives

* **Implement feature detection, matching, and tracking.** The algorithm utilizes feature detection algorithms, such as SIFT or SURF, to identify and match keypoints across consecutive images. This process allows for the establishment of correspondences between points in different frames, which is essential for estimating camera motion.

* **Estimate camera motion from a sequence of images.** Based on the matched features and their relative positions in the images, the algorithm estimates the 3D motion of the camera. This involves calculating the camera's translation and rotation between frames, enabling the reconstruction of the camera's path through the environment.

* **Understand the principles of monocular visual odometry and its applications.** The project delves into the theoretical foundations of monocular visual odometry, exploring the underlying principles and challenges involved in estimating camera motion from a single perspective. This knowledge provides a deeper understanding of the algorithm's capabilities and limitations.

## Installation

### Prerequisites

* **Python 3.x:** The project is developed using Python 3.x. Ensure Python 3.x is installed on your system.

* **OpenCV:** OpenCV is a powerful library for computer vision. Install OpenCV using your system's package manager or by downloading the official OpenCV installer.

* **Numpy:** Numpy is a fundamental library for scientific computing in Python. Install Numpy using your system's package manager or by downloading the official Numpy installer.

### Setup

1. **Clone the repository:** Clone the project's repository using the following command:

```sh
git clone [repository URL]

pip install -r requirements.txt


## How to Run

1. **Open the main.py file:** Open the `main.py` file in a text editor.

2. **Adjust the image_files list:** Place the unmodified image sequences folders inside the 'data' folder

3. **Execute the main script:** Run the following command in the terminal to execute the main script:

sh
python main.py


The script will process the specified image sequence and display the estimated camera trajectory and 3D reconstruction.
