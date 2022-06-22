import numpy as np
import cv2

import src

data = np.load('hw3.npy', allow_pickle = True)

verts = data[()]['verts']
vertex_colors = data[()]['vertex_colors']
face_indices = data[()]['face_indices']
depth = data[()]['depth']
cam_eye = data[()]['cam_eye']
cam_up = data[()]['cam_up']
cam_lookat = data[()]['cam_lookat']
ka = data[()]['ka']
kd = data[()]['kd']
ks = data[()]['ks']
n = data[()]['n']
light_positions = data[()]['light_positions']
light_intensities = data[()]['light_intensities']
Ia = data[()]['Ia']
M = data[()]['M']
N = data[()]['N']
W = data[()]['W']
H = data[()]['H']
bg_color = data[()]['bg_color']